import torch
import torch.utils.data as data
from PIL import Image
import os
import numpy as np

class VideoRecord(object):
    def __init__(self, row):
        self._data = row
        self._labels = torch.tensor([-1, -1, -1])
        labels=sorted(list(set([int(x) for x in self._data[2:]])))
        for i,l in enumerate(labels):
            self._labels[i]=l

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return self._labels


class ARNetDataSet(data.Dataset):
    def __init__(self, root_path, list_file, num_segments=3,
                 image_tmpl='img_{:05d}.jpg', transform=None, args=None):

        self.root_path = root_path
        self.list_file = \
            ".".join(list_file.split(".")[:-1]) + args.filelist_suffix + "."+ list_file.split(".")[-1] #TODO
        self.num_segments = num_segments
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.args=args
        self.root_path += self.args.folder_suffix
        self._parse_list()

    def _load_image(self, directory, idx):
        try:
            return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
        except Exception:
            print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
            exit()

    def _parse_list(self):
        splitter="," if self.args.dataset in ["actnet", "fcvid"] else " "
        tmp = [x.strip().split(splitter) for x in open(self.list_file)]

        if any(len(items)>=3 for items in tmp) and self.args.dataset=="minik":
            tmp = [[splitter.join(x[:-2]), x[-2], x[-1]] for x in tmp]

        self.video_list = [VideoRecord(item) for item in tmp]

        if self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
            for v in self.video_list:
                v._data[1] = int(v._data[1]) / 2
        print('video number:%d' % (len(self.video_list)))

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments:
            tick = record.num_frames  / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.array(list(range(record.num_frames)) + [record.num_frames-1]*(self.num_segments - record.num_frames))
        return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]

        if self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
            file_name = self.image_tmpl.format(int(record.path), 'x', 1)
            full_path = os.path.join(self.root_path, '{:06d}'.format(int(record.path)), file_name)
        else:
            file_name = self.image_tmpl.format(1)
            full_path = os.path.join(self.root_path, record.path, file_name)

        if not os.path.exists(full_path):
            print('################## Not Found:', os.path.join(self.root_path, record.path, file_name))
            exit()

        segment_indices = self._get_val_indices(record)
        return self.get(record, segment_indices)

    def get(self, record, indices):
        images = list()
        for seg_ind in indices:
            images.extend(self._load_image(record.path, int(seg_ind)))

        process_data = self.transform(images)
        if self.args.ada_reso_skip:
            return_items = [process_data]
            rescaled = [self.rescale(process_data, (x,x)) for xi,x in enumerate(self.args.reso_list[1:])
                        if (xi + 1 < len(self.args.backbone_list) or xi + 1==self.args.policy_input_offset)]
            return_items = return_items + rescaled + [record.label]
            return tuple(return_items)
        else:
            if self.args.rescale_to == 224:
                rescaled = process_data
            else:
                x = self.args.rescale_to
                rescaled = self.rescale(process_data, (x, x))

            return rescaled, record.label

    def rescale(self, input_data, size):
        return torch.nn.functional.interpolate(input_data.unsqueeze(1), size=size, mode='nearest').squeeze(1)

    def __len__(self):
        return len(self.video_list)
