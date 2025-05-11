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

__all__ = ["MaskFormerALLOFBSegSemanticDatasetMapperWithUNO"]


ade_classes=('wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road',
            'bed ', 'windowpane', 'grass', 'cabinet', 'sidewalk',
            'person', 'earth', 'door', 'table', 'mountain', 'plant',
            'curtain', 'chair', 'car', 'water', 'painting', 'sofa',
            'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair',
            'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp',
            'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
            'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
            'skyscraper', 'fireplace', 'refrigerator', 'grandstand',
            'path', 'stairs', 'runway', 'case', 'pool table', 'pillow',
            'screen door', 'stairway', 'river', 'bridge', 'bookcase',
            'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill',
            'bench', 'countertop', 'stove', 'palm', 'kitchen island',
            'computer', 'swivel chair', 'boat', 'bar', 'arcade machine',
            'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
            'chandelier', 'awning', 'streetlight', 'booth',
            'television receiver', 'airplane', 'dirt track', 'apparel',
            'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle',
            'buffet', 'poster', 'stage', 'van', 'ship', 'fountain',
            'conveyer belt', 'canopy', 'washer', 'plaything',
            'swimming pool', 'stool', 'barrel', 'basket', 'waterfall',
            'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food',
            'step', 'tank', 'trade name', 'microwave', 'pot', 'animal',
            'bicycle', 'lake', 'dishwasher', 'screen', 'blanket',
            'sculpture', 'hood', 'sconce', 'vase', 'traffic light',
            'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate',
            'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
            'clock', 'flag')
ade_skip_classes = (
    'refrigerator', 'screen door', 'blind', 'bar', 'computer', 'television receiver', 
    'apparel', 'pole', 'washer', 'microwave', 'blanket', 'ashcan', 'monitor', 'shower',
    'oven', 'rug', 'arcade machine'
)
ade_skip_labels = [50, 58, 63, 77, 74, 89, 92, 93, 107, 
                   124, 131, 138, 143, 145, 118, 28, 78]


class MaskFormerALLOFBSegSemanticDatasetMapperWithUNO:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for semantic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
        labels_mapping=None
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
        self.labels_mapping = labels_mapping

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")

        root = '/home/data/ADEChallengeData2016'
        self.ood_images = sorted(glob.glob(root + '/images' + '/training/*.jpg'))
        self.ood_annotations = sorted(glob.glob(root + '/annotations' + '/training/*.png'))

        self.ood_classes_per_item = 3
        self.skip_labels = np.array(ade_skip_labels)

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
        augs.append(T.RandomFlip(horizontal=True, vertical=False))
        augs.append(T.RandomFlip(horizontal=False, vertical=True))

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label
        
        if "labels_mapping" in meta.as_dict():
            labels_mapping = torch.tensor(meta.labels_mapping)
        else:
            labels_mapping = None

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            "labels_mapping": labels_mapping
        }
        return ret

    def _paste_anomaly(self, x, label, ood_patch, ood_lbl, ood_id):
        p_h, p_w, _ = ood_patch.shape
        h, w, _ = x.shape
        pos_i = random.randint(0, h - p_h)
        pos_j = random.randint(0, w - p_w)
        for i in range(3):
            x[pos_i: pos_i + p_h, pos_j: pos_j + p_w, i] = x[pos_i: pos_i + p_h, pos_j: pos_j + p_w, i] * (1 - ood_lbl) + ood_lbl * ood_patch[:, :, i]
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
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format).copy()
        utils.check_image_size(dataset_dict, image)

        sem_seg_file_name = dataset_dict["sem_seg_file_name"] if "sem_seg_file_name" in dataset_dict else None
        if "sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
        else:
            sem_seg_gt = None

        if sem_seg_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        #? Remap to foreground/background labels
        sem_seg_gt[(sem_seg_gt >= 1) & (sem_seg_gt != 255)] = 1
        
        ## paste outlier
        idx = np.random.randint(len(self.ood_images))
        ood_image = utils.read_image(self.ood_images[idx], format=self.img_format)
        ood_lbl = utils.read_image(self.ood_annotations[idx])
        unique_lbls = np.unique(ood_lbl)
        
        #? Only use images with specified classes (comment this if you want to use all classes)
        filtered_lbls = unique_lbls[~np.isin(unique_lbls, self.skip_labels)]
        unique_lbls = filtered_lbls #* comment this if you want to use all classes
        while len(unique_lbls) < self.ood_classes_per_item:
            idx = np.random.randint(len(self.ood_images))
            ood_image = utils.read_image(self.ood_images[idx], format=self.img_format)
            ood_lbl = utils.read_image(self.ood_annotations[idx])
            unique_lbls = np.unique(ood_lbl)
            filtered_lbls = unique_lbls[~np.isin(unique_lbls, self.skip_labels)]
            unique_lbls = filtered_lbls
        #? ----------------------------

        ood_size = np.random.randint(96, 500)
        factor = ood_size / max(ood_lbl.shape)
        if factor < 1.:
            ood_image = torch.nn.functional.interpolate(torch.from_numpy(ood_image).float().permute(2, 0, 1).unsqueeze(0), scale_factor=factor)[0].permute(1, 2, 0).long().numpy()
            ood_lbl = torch.nn.functional.interpolate(torch.from_numpy(ood_lbl).unsqueeze(0).unsqueeze(0), scale_factor=factor, mode='nearest')[0, 0].numpy()
        ood_image = np.uint8(ood_image)
        ood_lbl = ood_lbl.astype("double")

        for c in np.random.choice(unique_lbls, self.ood_classes_per_item):
            binary_ood_lbl = np.zeros_like(ood_lbl)
            binary_ood_lbl[ood_lbl == c] = 1
            binary_ood_lbl = np.uint8(binary_ood_lbl)
            image, sem_seg_gt = self._paste_anomaly(image, sem_seg_gt, ood_image, binary_ood_lbl, 2)
        ##

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image
        sem_seg_gt = aug_input.sem_seg

        # Pad image and segmentation label here!
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - image_size[1],
                0,
                self.size_divisibility - image_size[0],
            ]
            image = F.pad(image, padding_size, value=128).contiguous()
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()

        if "annotations" in dataset_dict:
            raise ValueError("Semantic segmentation dataset should not have 'annotations'.")

        # Prepare per-category binary masks
        if sem_seg_gt is not None:
            sem_seg_gt = sem_seg_gt.numpy()
            instances = Instances(image_shape)
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
