from dataclasses import dataclass, field
from typing import List

from diva.data_store import FrameStoreConfig
from frater.component import BatchComponentConfig


@dataclass
class ARNetConfig(BatchComponentConfig):
    weights: str = ''
    frame_store_config: FrameStoreConfig = field(
        default_factory=lambda: FrameStoreConfig(frame_filename_format='%08d%s', extension='.png'))
    num_categories: int = 0
    dataset: str = 'something_something'
    gpus: List[int] = field(default_factory=list)