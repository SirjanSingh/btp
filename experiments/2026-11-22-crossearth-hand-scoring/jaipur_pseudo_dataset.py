"""mmseg dataset for Jaipur tiles with teacher pseudo-labels (0 Building, 1 Background, 255 ignore).
Copy to CrossEarth/rs_dataset/jaipur_pseudo.py and add `from .jaipur_pseudo import JaipurPseudoDataset`
to CrossEarth/rs_dataset/__init__.py."""
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class JaipurPseudoDataset(BaseSegDataset):
    METAINFO = dict(classes=['Building', 'Background'],
                    palette=[(0, 0, 0), (255, 255, 255)])

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.png', seg_map_suffix='.png',
                         reduce_zero_label=False, **kwargs)
