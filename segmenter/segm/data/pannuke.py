from pathlib import Path

from segm.data.base import BaseMMSeg
from segm.data import utils
from segm.config import dataset_dir
from mmseg.datasets import DATASETS

PANNUKE_CONFIG_PATH = Path(__file__).parent / "config" / "pannuke.py"
PANNUKE_CATEGORIES_PATH = Path(__file__).parent / "config" / "pannuke.yml"

class PannukeSegmentation(BaseMMSeg):
    def __init__(self, image_size, crop_size, split, **kwargs):
        super().__init__(
            image_size,
            crop_size,
            split,
            PANNUKE_CONFIG_PATH,
            **kwargs,
        )
        self.names, self.colors = utils.dataset_cat_description(PANNUKE_CATEGORIES_PATH)
        self.n_cls = 6
        self.reduce_zero_label = False

    def update_default_config(self, config):
        root_dir = dataset_dir()
	print("in pannuke.py: root_dir from env variable = ",root_dir)
        path = Path(config.data_root)
	print("in pannuke.py: data_root from config/pannuke.py = ",config.data_root) 
        config.data_root = path
        if self.split == "train":
            config.data.train.data_root = path
        elif self.split == "trainval":
            config.data.trainval.data_root = path
        elif self.split == "val":
            config.data.val.data_root = path
        elif self.split == "test":
            config.data.test.data_root = path
        config = super().update_default_config(config)
        return config

    def test_post_process(self, labels):
        return labels + 1
