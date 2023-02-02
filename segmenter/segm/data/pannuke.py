import os
import numpy as np
from pathlib import Path

from setuptools import glob
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from PIL import Image
from torch import from_numpy

from segm.data import utils
from segm.config import dataset_dir


class PannukeDataset(Dataset):
    def __init__(
        self,
        root_dir,
        image_size=256,
        crop_size=256,
        split="train",
        normalization="vit",
    ):
        super().__init__()
        assert image_size[0] == image_size[1]

        # self.img_path = Path(root_dir) / split / 'images'
        # self.mask_path = Path(root_dir) / split / 'masks'
        self.img_path = glob.glob(os.path.join(root_dir, 'image', '*.npy'))
        self.mask_path = glob.glob(os.path.join(root_dir, 'mask', '*.npy'))
        self.crop_size = crop_size
        self.image_size = image_size
        self.split = split
        self.normalization = normalization

        if split == "train":
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(self.crop_size, interpolation=3),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(image_size[0] + 32, interpolation=3),
                    transforms.CenterCrop(self.crop_size),
                    transforms.ToTensor(),
                ]
            )

        # self.base_dataset = datasets.ImageFolder(self.path, self.transform)
        self.n_cls = 6

    @property
    def unwrapped(self):
        return self

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        im = from_numpy(np.load(self.img_path[idx]))#.ToTensor()
        target = from_numpy(np.load(self.mask_path[idx]))
        # im, target = self.base_dataset[idx]
        im = utils.rgb_normalize(im, self.normalization)
        return dict(im=im, target=target)

    def get_gt_seg_maps(self):
        dataset = self.dataset
        gt_seg_maps = {}
        for img_info in dataset.img_infos:
            seg_map = Path(dataset.ann_dir) / img_info["ann"]["seg_map"]
            gt_seg_map = mmcv.imread(seg_map, flag="unchanged", backend="pillow")
            gt_seg_map[gt_seg_map == self.ignore_label] = IGNORE_LABEL
            if self.reduce_zero_label:
                gt_seg_map[gt_seg_map != IGNORE_LABEL] -= 1
            gt_seg_maps[img_info["filename"]] = gt_seg_map
        return gt_seg_maps