"""Image Utils."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.v2.functional import to_image
from torchvision.tv_tensors import Mask

def read_mask(path: str | Path, as_tensor: bool = False) -> torch.Tensor | np.ndarray:
    """Read mask from disk.

    Args:
        path (str, Path): path to the mask file
        as_tensor (bool, optional): If True, returns the mask as a tensor. Defaults to False.

    Example:
        >>> mask = read_mask("test_mask.png")
        >>> type(mask)
        <class 'numpy.ndarray'>
        >>>
        >>> mask = read_mask("test_mask.png", as_tensor=True)
        >>> type(mask)
        <class 'torch.Tensor'>
    """
    image = Image.open(path).convert("L")
    if not as_tensor:
        return np.array(image)
    else:
        image = to_image(image).squeeze()
        return Mask(image, dtype=torch.uint8)