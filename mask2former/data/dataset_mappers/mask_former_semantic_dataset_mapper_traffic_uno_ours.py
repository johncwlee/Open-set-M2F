# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging

import numpy as np
import torch
from torch.nn import functional as F
import glob
import random

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.tv_tensors import Mask
from torchvision.datasets import Cityscapes as _CityscapesDataset
from albumentations import Compose
import albumentations as A

from anomalib.data.base import AnomalibDataset
from anomalib.data.utils import read_image

from .base import BaseDataModule, FusedTransforms, InputNormalizationMethod
from .city_ade import CityADEDataset
from .utils import read_mask
from .ade import ADE

__all__ = ["MaskFormerSemanticDatasetMapperWithUNOOurs"]

from cityscapesscripts.helpers.labels import labels as _labels

import json
import os
import torch


class CityADE(BaseDataModule):
    def __init__(
        self,
        **kwargs
    ) -> None:
        """
        """
        super().__init__(**kwargs)

    def _setup(self, _stage: str | None = None) -> None:
        del _stage  # Unused

    def setup(self, stage: str | None = None) -> None:
        self._setup(stage)


class MaskFormerSemanticDatasetMapperWithUNOOurs:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for semantic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """
    CLASS_INDICES = (7, 8, 11, 12, 13, 17, 19, 20, 21, 
                        22, 23, 24, 25, 26, 27, 28, 31, 32, 33)
    VOID_INDICES = (0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1)
    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")

        # self.vistas_mapper = create_vistas_to_cityscapes_mapper('./datasets/mapillary-vistas')

        root = '/home/johnl/data/ADEChallengeData2016'
        self.ood_images = sorted(glob.glob(root + '/images' + '/training/*.jpg'))
        self.ood_annotations = sorted(glob.glob(root + '/annotations' + '/training/*.png'))

        self.ood_classes_per_item = 3
        transform = Compose([
            A.LongestMaxSize(
                max_size=[1024, 1228, 1433, 1638, 1843, 2048, 2252, 2457, 2662, 2867, 3072, 3276, 3481, 3686, 3891, 4096],
                interpolation=1,
                p=1.0
            ),
            A.PadIfNeeded(min_height=512, min_width=1024, border_mode=0, value=0.0, mask_value=255, always_apply=True, p=1.0),
            A.RandomCrop(height=512, width=1024, always_apply=True, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.12, contrast=0.0, saturation=0.0, hue=0.0, p=0.5),
            A.ColorJitter(brightness=0.0, contrast=0.5, saturation=0.0, hue=0.0, p=0.5),
            A.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.5, hue=0.0, p=0.5),
            A.HueSaturationValue(hue_shift_limit=18, sat_shift_limit=0, val_shift_limit=0, p=0.5),
        ])
        
        basemodule = CityADE(
            train_batch_size=4,
            eval_batch_size=4,
            train_transform=transform,
            eval_transform=None,
            num_workers=4,
            image_size=None,
            normalization=InputNormalizationMethod.IMAGENET,
            pad_value=0.0,
            divisor=None,
            seed=42
        )
        self.transform = basemodule.train_transform
        
        self.ade = ADE(root)
        self.ignore_index = ignore_label

    def encode_target_city(self, target: Mask) -> Mask:
        """Modify the target to contain only valid classes."""
        for void_index in self.VOID_INDICES:
            target[target == void_index] = self.ignore_index
        for i, cls_index in enumerate(self.CLASS_INDICES):
            target[target == cls_index] = i
        return target

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        augs.append(T.RandomFlip())

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
        }
        return ret

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

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "MaskFormerSemanticDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        city_img_pth = dataset_dict["file_name"]
        city_tgt_pth = dataset_dict["sem_seg_file_name"]
        ade_idx = self.ade.get_idx()
        ade_img_pth = self.ade.images[ade_idx]
        ade_tgt_pth = self.ade.targets[ade_idx]

        city_img, city_tgt = read_image(city_img_pth, as_tensor=True), \
                                read_mask(city_tgt_pth, as_tensor=True)
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

        dataset_dict["image"] = image
        dataset_dict["sem_seg"] = mask

        # Prepare per-category binary masks
        sem_seg_gt = mask
        if sem_seg_gt is not None:
            sem_seg_gt = sem_seg_gt.numpy()
            instances = Instances(image.shape[-2:])
            classes = np.unique(sem_seg_gt)
            # remove ignored region
            classes = classes[classes != self.ignore_label]
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            masks = []
            for class_id in classes:
                masks.append(sem_seg_gt == class_id)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor

            dataset_dict["instances"] = instances

        return dataset_dict
