# Copyright (c) Facebook, Inc. and its affiliates.
import os
from pathlib import Path
import random
from detectron2.data import DatasetCatalog, MetadataCatalog

ALLO_SEM_SEG_CATEGORIES = [
    {"name": "background"},
    {"name": "robotic_arms"},
    {"name": "solar_arrays"},
    {"name": "pressurized_modules"},
    {"name": "airlock_docking_ports"},
    {"name": "truss_other"},
    {"name": "modules_other"},
    {"name": "celestial_body"},
    {"name": "anomaly"}
]

def _get_allo_meta():
    classes = [k["name"] for k in ALLO_SEM_SEG_CATEGORIES]
    ret = {
        "stuff_classes": classes,
    }
    return ret

def load_allo_train(root):
    root_ = Path(root) / "train_v3"
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

def load_allo_seg_val(root):
    root_ = Path(root) / "test_v3"
    image_files = [str(img) for img in sorted(root_.glob('**/normal/Camera*/images/*.png'))]
    examples = []

    for im_file in image_files:
        examples.append({
            "file_name": im_file,
            "sem_seg_file_name": im_file.replace('images', 'masks'),
            "height": 1080,
            "width": 1920,
        })
    return examples

def load_allo_anomaly_val(root):
    root_ = Path(root) / "test_v3"
    image_files = [str(img) for img in sorted(root_.glob('**/images/*.png'))]
    random.shuffle(image_files)
    image_files = image_files[:200]
    examples = []

    #* Take a subset to not run out of memory during evaluation
    for im_file in image_files:
        examples.append({
            "file_name": im_file,
            "sem_seg_file_name": im_file.replace('images', 'masks'),
            "height": 1080,
            "width": 1920,
        })
    return examples


def register_allo(root):
    #TODO: Change root to the correct path depending on system
    root = os.path.join(root, "blender/full")
    meta = _get_allo_meta()

    DatasetCatalog.register(
        "allo_train", lambda x=root: load_allo_train(x)
    )
    DatasetCatalog.register(
        "allo_seg_val", lambda x=root: load_allo_seg_val(x)
    )
    DatasetCatalog.register(
        "allo_anomaly_val", lambda x=root: load_allo_anomaly_val(x)
    )
    MetadataCatalog.get("allo_train").set(
        ignore_label=255,
        **meta,
    )
    MetadataCatalog.get("allo_seg_val").set(
        evaluator_type="sem_seg",
        ignore_label=255,
        **meta,
    )
    MetadataCatalog.get("allo_anomaly_val").set(
        evaluator_type="ood_detection",
        ignore_label=255,
        **meta,
    )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_allo(_root)
