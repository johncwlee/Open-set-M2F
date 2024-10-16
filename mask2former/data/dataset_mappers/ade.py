

from pathlib import Path

import numpy as np


class ADE(object):
    """This dataset is only used as an auxiliary dataset for OOD segmentation.
    It does not support the main functionality of a dataset class."""
    train_id_in = 0
    train_id_out = 254

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

        image_dir = self.root / "images" / "training"
        target_dir = self.root / "annotations" / "training"
        self.images = [str(pth) for pth in sorted(image_dir.glob("*.jpg"))]
        self.targets = [str(pth) for pth in sorted(target_dir.glob("*.png"))]

    def get_idx(self):
        """Pick a random index from the dataset."""
        return np.random.randint(0, len(self.images))