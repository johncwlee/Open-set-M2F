"""Base class for datamodule"""


import logging

from torch.utils.data.dataloader import DataLoader
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms.v2 import Transform
from albumentations import Compose

from anomalib.data.base import AnomalibDataModule
from anomalib.data.base.datamodule import collate_fn
from anomalib.data.utils import TestSplitMode, ValSplitMode

from .transforms import FusedTransforms, InputNormalizationMethod

logger = logging.getLogger(__name__)


class BaseDataModule(AnomalibDataModule):
    """Base data module.
    #TODO: change the docstring
    Args:
        train_batch_size (int): Batch size used by the train dataloader.
        eval_batch_size (int): Batch size used by the val and test dataloaders.
        num_workers (int): Number of workers used by the train, val and test dataloaders.
        val_split_mode (ValSplitMode): Determines how the validation split is obtained.
            Options: [none, same_as_test, from_test, synthetic]
        val_split_ratio (float): Fraction of the train or test images held our for validation.
        test_split_mode (Optional[TestSplitMode], optional): Determines how the test split is obtained.
            Options: [none, from_dir, synthetic].
            Defaults to ``None``.
        test_split_ratio (float): Fraction of the train images held out for testing.
            Defaults to ``None``.
        image_size (tuple[int, int], optional): Size to which input images should be resized.
            Defaults to ``None``.
        transform (Transform, optional): Transforms that should be applied to the input images.
            Defaults to ``None``.
        train_transform (Transform, optional): Transforms that should be applied to the input images during training.
            Defaults to ``None``.
        eval_transform (Transform, optional): Transforms that should be applied to the input images during evaluation.
            Defaults to ``None``.
        seed (int | None, optional): Seed used during random subset splitting.
            Defaults to ``None``.
    """

    def __init__(
        self,
        train_batch_size: int,
        eval_batch_size: int,
        num_workers: int,
        val_split_mode: ValSplitMode | str = ValSplitMode.FROM_TEST,
        val_split_ratio: float = 0.3,
        test_split_mode: TestSplitMode | str | None = None,
        test_split_ratio: float | None = None,
        image_size: tuple[int, int] | None = None,
        train_transform: Compose | None = None,
        eval_transform: Compose | None = None,
        normalization: InputNormalizationMethod = InputNormalizationMethod.IMAGENET,
        hist_eq: bool = False,
        pad_value: float = 0.0,
        mask_pad_value: int = 0,
        divisor: int | None = 16,
        seed: int | None = None,
        shuffle: bool = False,
        debug: bool = False,
        synthesize_cfg: dict = {},
    ) -> None:
        if train_transform is None:
            logger.info("No train_transform provided. Using default transforms (normalization) only.")
        if eval_transform is None:
            logger.info("No eval_transform provided. Using default transforms (normalization) only.")
        
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            image_size=image_size,
            train_transform=self._build_transforms(image_size, 
                                                   train_transform, 
                                                   normalization, 
                                                   hist_eq,
                                                   pad_value,
                                                   mask_pad_value,
                                                   divisor,
                                                   synthesize_cfg),
            eval_transform=self._build_transforms(image_size, 
                                                  eval_transform, 
                                                  normalization, 
                                                  hist_eq,
                                                  pad_value,
                                                  mask_pad_value,
                                                  divisor),
            seed=seed,
        )
        self.shuffle = shuffle
        self.debug = debug

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Get train dataloader."""
        return DataLoader(
            dataset=self.train_data,
            shuffle=True,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers if not self.debug else 0,
            pin_memory=(not self.debug),
            persistent_workers=(not self.debug),
            drop_last=True
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """Get validation dataloader."""
        return DataLoader(
            dataset=self.val_data,
            shuffle=self.shuffle,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers if not self.debug else 0,
            collate_fn=collate_fn,
            pin_memory=(not self.debug),
            persistent_workers=(not self.debug),
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """Get test dataloader."""
        return DataLoader(
            dataset=self.test_data,
            shuffle=self.shuffle,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers if not self.debug else 0,
            collate_fn=collate_fn,
            pin_memory=(not self.debug),
            persistent_workers=False
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        """Get predict dataloader."""
        return DataLoader(
            dataset=self.test_data,
            shuffle=self.shuffle,
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers if not self.debug else 0,
            collate_fn=collate_fn,
            pin_memory=(not self.debug),
            persistent_workers=False
        )

    def _build_transforms(self, 
                            image_size: tuple[int, int] | None, 
                            transforms: Compose | None,
                            normalization: InputNormalizationMethod,
                            hist_eq: bool,
                            pad_value: float,
                            mask_pad_value: int,
                            divisor: int | None,
                            synthesize_cfg: dict = {}) -> FusedTransforms:
        """Build FusedTransforms with given image_size and `Compose` transforms."""
        return FusedTransforms(image_size=image_size, 
                               transforms=transforms,
                               normalization=normalization,
                               hist_eq=hist_eq,
                               pad_value=pad_value,
                               mask_pad_value=mask_pad_value,
                               divisor=divisor,
                               synthesize_cfg=synthesize_cfg)

    @property
    def transform(self) -> Compose | None:
        """Property that returns the user-specified transform for the datamodule, if any.

        This property is accessed by the engine to set the transform for the model. The eval_transform takes precedence
        over the train_transform, because the transform that we store in the model is the one that should be used during
        inference.
        """
        return self._eval_transform

    @property
    def train_transform(self) -> Transform:
        """Get the transforms that will be passed to the train dataset."""
        return self._train_transform

    @property
    def eval_transform(self) -> Transform:
        """Get the transform that will be passed to the val/test/predict datasets."""
        return self._eval_transform
