from typing import List, Iterable

import numpy as np
import torch
import torchvision
from diva.component import ActivityClassifier
from diva.data_store import FrameStore
from diva.data_type import ActivityProposal, Activity, BoundingBox
from frater.dataset import dataset_factory
from frater.stream import OutputStream, InputStream
from torch.nn import DataParallel

from .zz_config import ARNetConfig
from .zz_arnet import ARNet
from ..builders import build_activities_from_outputs_and_proposals
from ..ops import transforms

__all__ = ['ARNetActivityClassifier']


class ARNetActivityClassifier(ActivityClassifier):
    def __init__(self, config: ARNetConfig, input_stream: InputStream, output_stream: OutputStream):
        super(ARNetActivityClassifier, self).__init__(config, input_stream, output_stream)

        self.model = ARNet(n_class=config.num_categories)
        self.frame_store = FrameStore(config.frame_store_config)
        self.model.load_state_dict(torch.load(config.weights))
        self.model = DataParallel(self.model, device_ids=config.gpus)

    @property
    def dataset(self):
        return dataset_factory[self.config.dataset]

    def transform(self):
        return torchvision.transforms.Compose([
            transforms.GroupScale(int(self.model.scale_size)),
            transforms.GroupCenterCrop(self.model.input_size),
            transforms.Stack(
                roll=(self.config.arch in ['BNInception', 'InceptionV3'])),
            transforms.ToTorchFormatTensor(
                div=(self.config.arch not in ['BNInception', 'InceptionV3'])),
            transforms.GroupNormalize(self.model.input_mean, self.model.input_std),
        ])

    def process(self, batch: List[ActivityProposal]) -> Iterable[Activity]:
        batch = self.clip_proposals(batch)
        transformed_proposals = self._transform_proposals(batch)
        unfiltered_indices, filtered_indices = self.filter_proposals(transformed_proposals)

        initial_outputs = [[1.0] + [0.0] * self.config.num_categories] * len(filtered_indices)
        initial_proposals = [batch[i] for i in filtered_indices]
        activities = build_activities_from_outputs_and_proposals(initial_outputs, initial_proposals, self.dataset)
        if len(unfiltered_indices) > 0:
            inputs = torch.stack([transformed_proposals[i] for i in unfiltered_indices])
            outputs = self.model(inputs)
            unfiltered_proposals = [batch[i] for i in unfiltered_indices]
            activities.extend(build_activities_from_outputs_and_proposals(outputs, unfiltered_proposals, self.dataset))

        return activities

    @staticmethod
    def filter_proposals(transformed_proposals):
        unfiltered_indices = list()
        filtered_indices = list()
        for i, proposal in enumerate(transformed_proposals):
            if proposal is None:
                filtered_indices.append(i)
            else:
                unfiltered_indices.append(i)

        return unfiltered_indices, filtered_indices

    def _transform_proposals(self, proposals: List[ActivityProposal]) -> List[torch.FloatTensor]:
        return [self._transform_proposal(proposal) for proposal in proposals]

    def _transform_proposal(self, proposal: ActivityProposal) -> torch.FloatTensor:
        # make sure proposal is within bounds
        num_frames = len(proposal.temporal_range)
        indices = self._sample_indices(num_frames, proposal.start_frame)
        boxes: List[BoundingBox] = [proposal[int(index)] for index in indices]

        frames = [self.frame_store.get_frame(proposal.source_video, box.frame_index,
                                             self.config.modality_type).crop(box).image for box in boxes]
        try:
            return self.transform(frames)
        except ZeroDivisionError:
            return []

    def clip_proposals(self, proposals: List[ActivityProposal]):
        return [self.clip_proposal(proposal) for proposal in proposals]

    def clip_proposal(self, proposal: ActivityProposal):
        min_frame = max(self.frame_store.get_min_frame(proposal.source_video), proposal.start_frame)
        max_frame = min(self.frame_store.get_max_frame(proposal.source_video), proposal.end_frame)
        return proposal[min_frame:max_frame]

    def _sample_indices(self, num_frames: int, start_frame: int = 0):
        if self.config.dense_sample:
            sample_pos = max(1, 1 + num_frames - 64)
            t_stride = 64 // self.config.num_segments
            start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % num_frames for idx in range(self.config.num_segments)]
            return np.array(offsets) + start_frame
        else:
            tick = (num_frames - self.config.new_length + 1) / float(self.config.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.config.num_segments)])
            return offsets + start_frame