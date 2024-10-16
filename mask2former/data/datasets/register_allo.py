# Copyright (c) Facebook, Inc. and its affiliates.
import os
import glob
from pathlib import Path
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

ALLO_SEM_SEG_CATEGORIES = [
    {
        "color": [0, 0, 0],
        "instances": True,
        "readable": "Inliers",
        "name": "inliers",
        "evaluate": True,
    },
    {
        "color": [255, 255, 255],
        "instances": True,
        "readable": "Outlier",
        "name": "outlier",
        "evaluate": True,
    }
]

def _get_allo_meta():
    stuff_classes = [k["readable"] for k in ALLO_SEM_SEG_CATEGORIES if k["evaluate"]]
    stuff_colors = [k["color"] for k in ALLO_SEM_SEG_CATEGORIES if k["evaluate"]]
    ret = {
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret

def load_allo_train(root):
    root_ = Path(root) / "train_v2"
    image_files = [str(img) for img in sorted(root_.glob('**/normal/**/images/*.png'))]
    examples = []

    for im_file in image_files:
        examples.append({
            "file_name": im_file,
            "sem_seg_file_name": im_file.replace('images', 'masks'),
            "height": 1080,
            "width": 1920,
        })
    return examples

def load_allo_val(root):
    root_ = Path(root) / "test_v2"
    image_files = [str(img) for img in sorted(root_.glob('**/images/*.png'))]
    examples = []

    for im_file in image_files:
        examples.append({
            "file_name": im_file,
            "sem_seg_file_name": im_file.replace('images', 'masks'),
            "height": 1080,
            "width": 1920,
        })
    return examples


def register_all_allo(root):
    #TODO: Change root to the correct path depending on system
    root = os.path.join(root, "blender/full")
    meta = _get_allo_meta()

    DatasetCatalog.register(
        "allo_train", lambda x=root: load_allo_train(x)
    )
    DatasetCatalog.register(
        "allo_val", lambda x=root: load_allo_val(x)
    )
    MetadataCatalog.get("allo_train").set(
        evaluator_type="ood_detection",
        ignore_label=255,
        **meta,
    )
    MetadataCatalog.get("allo_val").set(
        evaluator_type="ood_detection",
        ignore_label=255,
        **meta,
    )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_allo(_root)
