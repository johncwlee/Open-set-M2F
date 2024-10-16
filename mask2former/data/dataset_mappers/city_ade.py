"""Data module for Cityscapes dataset with ADE20k anomaly mixing.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.tv_tensors import Mask
from torchvision.datasets import Cityscapes as _CityscapesDataset
from albumentations import Compose

from anomalib.data.base import AnomalibDataset
from anomalib.data.utils import read_image

from .base import BaseDataModule, FusedTransforms, InputNormalizationMethod
from .utils import read_mask
from .ade import ADE

logger = logging.getLogger(__name__)


class CityADEDataset(_CityscapesDataset):
    """Dataset class for Cityscapes+ADE20k Mix dataset."""
    CLASS_NAMES=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                'traffic light', 'traffic sign', 'vegetation', 'terrain',
                'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                'motorcycle', 'bicycle')
    CLASS_INDICES = (7, 8, 11, 12, 13, 17, 19, 20, 21, 
                        22, 23, 24, 25, 26, 27, 28, 31, 32, 33)
    VOID_INDICES = (0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1)
    def __init__(
        self,
        ade_root: Path | str,
        transform: FusedTransforms,
        ignore_index: int = 255,
        **kwargs
    ) -> None:
        """Initialize the Cityscapes with ADE20K mixing dataset."""
        super().__init__(**kwargs)

        self.transform = transform
        self.ignore_index = ignore_index

        #* ADE dataset
        self.ade = ADE(ade_root)

    def encode_target_city(self, target: Mask) -> Mask:
        """Modify the target to contain only valid classes."""
        for void_index in self.VOID_INDICES:
            target[target == void_index] = self.ignore_index
        for i, cls_index in enumerate(self.CLASS_INDICES):
            target[target == cls_index] = i
        return target

    def _paste_anomaly(self, x, label, ood_patch, ood_lbl, ood_id):
        p_h, p_w = ood_patch.shape[-2:]
        h, w = x.shape[-2:]
        pos_i = torch.randint(0, h - p_h, (1,)).item()
        pos_j = torch.randint(0, w - p_w, (1,)).item()
        for i in range(3):
            x[i, pos_i: pos_i + p_h, pos_j: pos_j + p_w] = x[i, pos_i: pos_i + p_h, pos_j: pos_j + p_w] \
                                                            * (1 - ood_lbl) + ood_lbl * ood_patch[i, :, :]
        label[pos_i: pos_i + p_h, pos_j: pos_j + p_w][ood_lbl == 1] = ood_id
        return x, label

    def __getitem__(self, index: int) -> dict[str, str | torch.Tensor]:
        """Get dataset item for the index ``index``.

        Args:
            index (int): Index to get the item.

        Returns:
            dict[str, str | torch.Tensor]: Dict of image tensor during training. Otherwise, Dict containing image path,
                target path, image tensor, label and transformed bounding box.
        """
        city_img_pth = self.images[index]
        if self.target_type[0] != "semantic":
            raise ValueError(f"Target type {self.target_type} is not supported for this Cityscapes dataset class.")
        city_tgt_pth = self.targets[index][0]
        ade_idx = self.ade.get_idx()
        ade_img_pth = self.ade.images[ade_idx]
        ade_tgt_pth = self.ade.targets[ade_idx]

        city_img, city_tgt = read_image(city_img_pth, as_tensor=True), \
                                self.encode_target_city(read_mask(city_tgt_pth, as_tensor=True))
        ade_img, ade_tgt = read_image(ade_img_pth, as_tensor=True), read_mask(ade_tgt_pth, as_tensor=True)

        #? Randomly rescale ADE image
        ade_size = torch.randint(96, 500, (1,)).item()
        factor = ade_size / max(ade_tgt.shape)
        if factor < 1.:
            ade_img = F.interpolate(ade_img.unsqueeze(0), scale_factor=factor).squeeze(0)
            ade_tgt = F.interpolate(ade_tgt[None, None, ...], scale_factor=factor, mode='nearest').squeeze()

        #? Mix Cityscapes and ADE
        unique_ade_cls = ade_tgt.unique()
        for c in np.random.choice(unique_ade_cls, 3):   #* Randomly select 3 classes from ADE
            binary_ade_tgt = torch.zeros_like(ade_tgt)
            binary_ade_tgt[ade_tgt == c] = 1
            image, mask = self._paste_anomaly(city_img, city_tgt, ade_img, binary_ade_tgt, 19)

        image, mask = self.transform(image, mask)
        return {
            "id_image_path": city_img_pth,
            "ood_image_path": ade_img_pth,
            "image": image,
            "mask": mask,
        }


class CityADE(BaseDataModule):
    """Data module for Cityscapes+ADE Mix dataset."""
    def __init__(
        self,
        city_root: Path | str = "./data/ALLO",
        ade_root: Path | str = "./data/ADEChallengeData2016",
        fishy_root: Path | str = "./data/fishyscapes",
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        inference_batch_size: int = 8,
        num_workers: int = 8,
        image_size: tuple[int, int] = (1024, 2048),
        train_transform: Compose | None = None,
        eval_transform: Compose | None = None,
        normalization: InputNormalizationMethod = InputNormalizationMethod.IMAGENET,
        pad_value: float = 0.0,
        divisor: int | None = 16,
        seed: int | None = None,
        **kwargs
    ) -> None:
        """Initialize the Cityscapes+ADE Mix datamodule.
        """
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            train_transform=train_transform,
            eval_transform=eval_transform,
            num_workers=num_workers,
            image_size=image_size,
            normalization=normalization,
            pad_value=pad_value,
            divisor=divisor,
            seed=seed,
            **kwargs
        )
        self.city_root = Path(city_root)
        self.ade_root = Path(ade_root)
        self.fishy_root = Path(fishy_root)

        self.inference_batch_size = inference_batch_size

    def _setup(self, _stage: str | None = None) -> None:
        del _stage  # Unused

        self.train_data = CityADEDataset(
            root=self.city_root,
            split="train",
            mode="fine",
            target_type="semantic",
            ade_root=self.ade_root,
            transform=self.train_transform
        )
        #* Use Fishyscapes for validation
        self.val_data = FishyscapesDataset(
            root=self.fishy_root,
            transform=self.eval_transform
        )
        self.test_data = self.val_data

    def setup(self, stage: str | None = None) -> None:
        self._setup(stage)
