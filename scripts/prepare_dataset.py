"""
prepare_dataset.py — Set up YOLO segmentation dataset structure.

Usage:
    python scripts/prepare_dataset.py [--dataset-dir data/dataset] [--synthetic-count 20]

If you have a real COCO-format or YOLO-format dataset, place it in data/dataset/
replacing the generated synthetic samples.
"""

import argparse
import os
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


CLASSES = ["dent", "scratch", "peel_paint", "broken"]
SPLITS = ["train", "val", "test"]


def create_directory_structure(base: Path) -> None:
    for split in SPLITS:
        (base / "images" / split).mkdir(parents=True, exist_ok=True)
        (base / "labels" / split).mkdir(parents=True, exist_ok=True)
    print(f"✅ Directory structure created at: {base}")


def generate_synthetic_sample(img_path: Path, label_path: Path, class_id: int) -> None:
    """
    Generate a synthetic damage image with a random polygon mask.
    Clearly marked as synthetic for training experiments.
    """
    W, H = 640, 480
    # Car-like background
    img = Image.new("RGB", (W, H), color=(180 + random.randint(-30, 30),
                                           170 + random.randint(-30, 30),
                                           160 + random.randint(-30, 30)))
    draw = ImageDraw.Draw(img)
    # Body rectangle
    draw.rectangle([80, 140, 560, 360], fill=(70 + random.randint(0, 40),
                                               70 + random.randint(0, 40),
                                               80 + random.randint(0, 40)))

    # Damage polygon (random convex-ish shape)
    cx = random.randint(150, 490)
    cy = random.randint(160, 340)
    points = []
    n_pts = random.randint(5, 8)
    for i in range(n_pts):
        angle = 2 * np.pi * i / n_pts
        r = random.randint(15, 60)
        px = cx + int(r * np.cos(angle))
        py = cy + int(r * np.sin(angle))
        points.append((px, py))

    damage_colors = {
        0: (90, 70, 60),    # dent — darker
        1: (200, 200, 200), # scratch — light streak
        2: (160, 130, 70),  # peel_paint — brownish
        3: (20, 20, 20),    # broken — very dark
    }
    draw.polygon(points, fill=damage_colors.get(class_id, (100, 100, 100)))

    img.save(img_path)

    # YOLO segmentation label: class_id x1 y1 x2 y2 ... (normalised)
    flat = []
    for (px, py) in points:
        flat.extend([px / W, py / H])
    label_line = f"{class_id} " + " ".join(f"{v:.6f}" for v in flat)
    label_path.write_text(label_line)


def write_dataset_yaml(base: Path, yaml_path: Path) -> None:
    content = f"""# Auto-generated YOLO dataset config
path: {base.resolve()}
train: images/train
val: images/val
test: images/test

nc: {len(CLASSES)}
names: {CLASSES}
"""
    yaml_path.write_text(content)
    print(f"✅ Dataset YAML written: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare YOLO Segmentation Dataset")
    parser.add_argument("--dataset-dir", default="data/dataset", help="Root dataset directory")
    parser.add_argument("--synthetic-count", type=int, default=20, help="Synthetic samples per class per split")
    args = parser.parse_args()

    base = Path(args.dataset_dir)
    create_directory_structure(base)

    total = 0
    for split_idx, split in enumerate(SPLITS):
        count = args.synthetic_count if split == "train" else max(4, args.synthetic_count // 4)
        for cls_id, cls_name in enumerate(CLASSES):
            for i in range(count):
                stem = f"synthetic_{cls_name}_{split}_{i:04d}"
                img_path = base / "images" / split / f"{stem}.jpg"
                label_path = base / "labels" / split / f"{stem}.txt"
                generate_synthetic_sample(img_path, label_path, cls_id)
                total += 1

    print(f"✅ Generated {total} synthetic samples.")
    print("⚠️  These are SYNTHETIC images for demo purposes only.")
    print("   Replace data/dataset/ with your real annotated dataset for production training.")

    yaml_path = base / "dataset.yaml"
    write_dataset_yaml(base, yaml_path)


if __name__ == "__main__":
    main()
