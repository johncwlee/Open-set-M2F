"""Implementation of FusedTransforms class for combining albumentations and torchvision transforms."""


from __future__ import annotations

import logging
from enum import Enum
from abc import ABC
import random

import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms.v2 as v2

from anomalib.data.utils.image import get_image_height_and_width

# from core.utils.pre_processing import generate_polygon

logger = logging.getLogger(__name__)


class InputNormalizationMethod(str, Enum):
    """Normalization method for the input images."""
    IMAGENET = "imagenet"  #* normalization to ImageNet statistics
    ALLO = "allo"  #* normalization to ALLO dataset statistics
    ALLO_CLAHE = "allo_clahe"  #* normalization to ALLO dataset statistics with CLAHE
    ALLO_PRETRAIN = "allo_pretrain" #* normalization to ALLO anomaly pretraining dataset
    ALLO_V2 = "allo_v2"  #* normalization to ALLO v2 dataset statistics

class NormalizationStats(Enum):
    """Statistics for normalization."""
    IMAGENET = {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225)
    }
    # ALLO = {
    #     "mean": (0.2268, 0.2306, 0.2326),
    #     "std": (0.2145, 0.2173, 0.2186)
    # }
    ALLO = {
        "mean": (0.1328, 0.1364, 0.1382),
        "std": (0.2045, 0.2068, 0.2076)
    }
    ALLO_CLAHE = {
        "mean": (0.1811, 0.2089, 0.1871),
        "std": (0.1850, 0.2235, 0.1946)
    }
    ALLO_PRETRAIN = {
        "mean": (0.1296, 0.1329, 0.1373),
        "std": (0.2040, 0.2059, 0.2101)
    }
    ALLO_V2 = {
        "mean": (0.1448, 0.1485, 0.1503),
        "std": (0.2119, 0.2141, 0.2149)
    }


class RandomScaleV2(A.RandomScale):
    """Randomly scaling with predefined scale factors."""
    def __init__(self, scale_factors: list[float] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.scale_factors = scale_factors

    def get_params(self) -> dict[str, float]:
        if self.scale_factors is not None:
            scale = random.choice(self.scale_factors)
            if not isinstance(scale, float):
                raise ValueError(f"Invalid scale factor: {scale}")
            return {"scale": scale}
        
        return super().get_params()

class FusedTransforms(ABC):
    """Class for transformations that combines albumentations and torchvision transforms.
    This already includes the normalization and ToTensor transforms.
    The order of transforms is as follows:
        1. Albumentations transforms (custom; specified by user in config)
        2. Resize transform (LongestMaxSize, PadIfNeeded; based on image_size in config)
        3. Normalize transform
    """
    def __init__(self, 
                transforms: A.Compose | None,
                image_size: int | tuple[int, int] | None,
                normalization: InputNormalizationMethod,
                hist_eq: bool = False,
                pad_value: float | tuple[float] = 0.0,
                mask_pad_value: int = -100,
                divisor: int | None = None,
                synthesize_cfg: dict = {}) -> None:
        self.albu_transforms = transforms if transforms is not None else A.Compose([])
        self.image_size = get_image_height_and_width(image_size) if image_size is not None else None
        self.normalization = normalization
        self.hist_eq = hist_eq
        if isinstance(pad_value, float):
            pad_value = (pad_value, pad_value, pad_value)
        self.pad_value = pad_value
        self.mask_pad_value = mask_pad_value
        self.divisor = divisor
        self.synthesize_cfg = synthesize_cfg
        self._synthesize = False #* disabled by default

    @property
    def synthesize(self) -> bool:
        return self._synthesize

    @synthesize.setter
    def synthesize(self, value: bool) -> None:
        self._synthesize = value

    @property
    def resize(self) -> bool:
        geo_transforms = [A.RandomResizedCrop, A.RandomScale, A.LongestMaxSize, A.RandomCrop]
        return not any(type(t) in geo_transforms for t in self.albu_transforms)

    def __call__(self, image, *args, **kwargs):
        """Apply the transforms to the input image."""
        #? Convert to numpy if tensor
        #* (C, H, W) -> (H, W, C)
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()
        else:
            if not isinstance(image, np.ndarray):
                raise ValueError(f"Unknown input image type: {type(image)}")

        #? CLAHE preprocessing
        if getattr(self, "hist_eq", False):
            #TODO: use CLAHE from albumentations instead
            #TODO: see https://albumentations.ai/docs/api_reference/augmentations/transforms/
            raise NotImplementedError("CLAHE from albumentations is not yet implemented.")
            image = (image * 255).astype("uint8")
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(image)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            v = clahe.apply(v)
            image = cv2.merge((h, s, v))
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
            image = image.astype("float32") / 255

        input_dict = {"image": image}
        if args:
            #? Add mask if provided
            mask = args[0]
            if isinstance(mask, torch.Tensor):
                if mask.ndim == 3:
                    mask = mask[0]
                mask = mask.numpy()
            else:
                if not isinstance(mask, np.ndarray):
                    raise ValueError(f"Unknown input mask type: {type(mask)}")
            input_dict["mask"] = mask
        
        #? Generate synthetic anomaly if specified
        transformed = self.generate_synthetic_anomaly(input_dict) \
                        if self.synthesize else input_dict
        
        #? Apply provided transforms from albumentations
        transformed = self.albu_transforms(**transformed) \
                        if self.albu_transforms is not None else input_dict
        
        #? Resize the image if image_size is provided
        transformed = self.resize_transform(**transformed) \
                        if self.resize_transform is not None else transformed
        
        #? Convert to tensor
        transformed = ToTensorV2()(**transformed)
        
        #? Apply post-transforms
        final_img = self.post_transforms(transformed["image"])
        
        if "mask" in transformed:
            return final_img, transformed["mask"]
        else:
            return final_img

    @property
    def norm_stats(self):
        if self.normalization == InputNormalizationMethod.IMAGENET:
            stats = NormalizationStats.IMAGENET
        elif self.normalization == InputNormalizationMethod.ALLO:
            stats = NormalizationStats.ALLO
        elif self.normalization == InputNormalizationMethod.ALLO_CLAHE:
            stats = NormalizationStats.ALLO_CLAHE
        elif self.normalization == InputNormalizationMethod.ALLO_PRETRAIN:
            stats = NormalizationStats.ALLO_PRETRAIN
        elif self.normalization == InputNormalizationMethod.ALLO_V2:
            stats = NormalizationStats.ALLO_V2
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization}")
        
        mean = stats.value["mean"]
        std = stats.value["std"]
        return mean, std

    @property
    def post_transforms(self) -> v2.Transform:
        """Get final transforms for the images: ToDtype, Normalize"""
        mean, std = self.norm_stats
        return v2.Compose(
            [
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std)
            ]
        )

    @property
    def resize_transform(self) -> A.Compose:
        """Custom resize transform that combines LongestMaxSize and PadIfNeeded.
        Assumes that the width is always longer than the height."""
        transformations = []
        if self.image_size is not None and self.resize:
            transformations.extend([
                A.LongestMaxSize(max_size=self.image_size[1],
                                    interpolation=3,    #* area interpolation for downscaling
                                    always_apply=True),
                # A.PadIfNeeded(min_height=self.image_size[0],
                #                 min_width=self.image_size[1],
                #                 pad_height_divisor=None,
                #                 pad_width_divisor=None,
                #                 border_mode=cv2.BORDER_CONSTANT, #* constant padding
                #                 value=self.pad_value,
                #                 mask_value=self.mask_pad_value,
                #                 always_apply=True),
            ])
        if self.divisor is not None and isinstance(self.divisor, int):
            transformations.append(
                A.PadIfNeeded(min_height=None,
                                min_width=None,
                                pad_height_divisor=self.divisor,
                                pad_width_divisor=self.divisor,
                                border_mode=cv2.BORDER_CONSTANT, #* constant padding
                                value=self.pad_value,
                                mask_value=self.mask_pad_value,
                                always_apply=True)
            )
        return A.Compose(transformations)

    def generate_synthetic_anomaly(self, input_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Generate synthetic anomaly in the image.

        Args:
            input_dict (dict[str, np.ndarray]): Input to be transformed. Has keys "image" and
                "mask" if available.

        Returns:
            dict[str, np.ndarray]: transformed image and mask.
        """
        image = input_dict["image"] #* (H, W, C)
        mask = input_dict["mask"] #* (H, W)
        H, W = mask.shape
        num_objs = np.random.randint(0, int(self.synthesize_cfg["max_num_objs"]) + 1, (1,)).item()
        num_points = int(self.synthesize_cfg["num_points"])
        min_factor, max_factor = float(self.synthesize_cfg["min_factor"]), \
                                    float(self.synthesize_cfg["max_factor"])

        for i in range(num_objs):
            #? Generate polygon
            rr, cc, box_w, box_h = generate_polygon(H, W, num_points, (min_factor, max_factor))
            num_outliers = len(rr)

            #? Sample random noise for the polygon
            mean, std = self.norm_stats
            mean, cov = np.array(mean), np.diag(np.array(std) ** 2)
            noise = np.random.multivariate_normal(mean, cov, (num_outliers,))

            #? Pick random location to place the polygon (top-left corner)
            #* Note: this will flip the polygon vertically since it assumed the origin is bottom-left
            x = np.random.randint(0, W - box_w - 1, (1,))
            y = np.random.randint(0, H - box_h - 1, (1,))
            rr += x
            cc += y

            #? Place the polygon
            image[cc, rr, :] = noise
            mask[cc, rr] = self.synthesize_cfg["anomaly_label"]

        return {"image": image, "mask": mask}