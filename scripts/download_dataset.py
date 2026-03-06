"""
Automated Car Damage Dataset Download & YOLO-seg Conversion
-----------------------------------------------------------
Downloads the CarDD dataset from HuggingFace (harpreetsahota/CarDD)
which has COCO-format polygon segmentation annotations,
converts to YOLO-seg format, and places under data/dataset/.

Classes remapped to match our 4-class system:
  Original CarDD:  crack, dent, glass_shatter, lamp_broken, scratch, tire_flat
  Our mapping:
    dent          -> dent   (0)
    scratch       -> scratch (1)
    crack         -> peel_paint (2)  [surface damage, closest match]
    glass_shatter -> broken  (3)
    lamp_broken   -> broken  (3)
    tire_flat     -> broken  (3)

Usage:
  pip install datasets pillow
  python scripts/download_dataset.py
"""

import json
import os
import sys
import shutil
from pathlib import Path
import urllib.request

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "data" / "dataset"

# Class remapping: CarDD original -> our 4-class system index
CLASS_MAP = {
    "dent":          0,   # dent
    "scratch":       1,   # scratch
    "crack":         2,   # peel_paint (closest: surface damage)
    "glass_shatter": 3,   # broken
    "lamp_broken":   3,   # broken
    "tire_flat":     3,   # broken
}

OUR_CLASSES = ["dent", "scratch", "peel_paint", "broken"]


def setup_dirs():
    for split in ["train", "val", "test"]:
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    print(f"[+] Directory structure created at {DATASET_DIR}")


def coco_polygon_to_yolo(segmentation, img_w, img_h):
    """Convert COCO polygon [[x1,y1,x2,y2,...]] to YOLO normalized polygon."""
    all_points = []
    for polygon in segmentation:
        if len(polygon) < 6:
            continue
        pts = []
        for i in range(0, len(polygon), 2):
            x = min(max(polygon[i] / img_w, 0.0), 1.0)
            y = min(max(polygon[i+1] / img_h, 0.0), 1.0)
            pts.extend([x, y])
        all_points.extend(pts)
    return all_points


def convert_coco_to_yolo(coco_json_path, images_src_dir, images_dst_dir, labels_dst_dir, category_map):
    """Convert a COCO-format annotation file to YOLO-seg txt files."""
    with open(coco_json_path) as f:
        coco = json.load(f)

    # Build lookup maps
    img_map = {img["id"]: img for img in coco["images"]}
    cat_map = {cat["id"]: cat["name"].lower().replace(" ", "_") for cat in coco["categories"]}

    # Group annotations by image
    ann_by_img = {}
    for ann in coco["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    converted = 0
    skipped = 0

    for img_id, img_info in img_map.items():
        img_filename = img_info["file_name"]
        img_w = img_info["width"]
        img_h = img_info["height"]

        src_img = images_src_dir / img_filename
        if not src_img.exists():
            skipped += 1
            continue

        # Copy image
        dst_img = images_dst_dir / img_filename
        shutil.copy2(src_img, dst_img)

        # Write label file
        anns = ann_by_img.get(img_id, [])
        label_lines = []
        for ann in anns:
            cat_name = cat_map.get(ann["category_id"], "")
            cls_idx = category_map.get(cat_name)
            if cls_idx is None:
                continue
            if not ann.get("segmentation"):
                continue
            pts = coco_polygon_to_yolo(ann["segmentation"], img_w, img_h)
            if len(pts) < 6:
                continue
            coords_str = " ".join(f"{v:.6f}" for v in pts)
            label_lines.append(f"{cls_idx} {coords_str}")

        if label_lines:
            stem = Path(img_filename).stem
            label_file = labels_dst_dir / f"{stem}.txt"
            label_file.write_text("\n".join(label_lines))
            converted += 1

    print(f"  Converted: {converted} images, Skipped: {skipped}")
    return converted


def try_huggingface_download():
    """Try to download via HuggingFace datasets library."""
    try:
        from datasets import load_dataset
        print("[+] Downloading CarDD dataset from HuggingFace (harpreetsahota/CarDD)...")
        print("    This may take a few minutes (~500MB)...")
        
        dataset = load_dataset("harpreetsahota/CarDD", trust_remote_code=True)
        
        raw_dir = DATASET_DIR / "_raw_hf"
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        split_map = {"train": "train", "validation": "val", "test": "test"}
        
        total = 0
        for hf_split, our_split in split_map.items():
            if hf_split not in dataset:
                continue
            
            split_data = dataset[hf_split]
            img_dst = DATASET_DIR / "images" / our_split
            lbl_dst = DATASET_DIR / "labels" / our_split
            
            print(f"  Processing {hf_split} split ({len(split_data)} samples)...")
            
            for i, sample in enumerate(split_data):
                if i % 100 == 0:
                    print(f"    {i}/{len(split_data)}", end="\r")
                
                # Save image
                img = sample.get("image") or sample.get("img")
                if img is None:
                    continue
                
                img_name = f"img_{our_split}_{i:05d}.jpg"
                img.save(img_dst / img_name)
                
                # Extract segmentation annotations
                objects = sample.get("objects") or sample.get("annotations") or {}
                
                label_lines = []
                if isinstance(objects, dict):
                    categories = objects.get("category", objects.get("label", []))
                    polygons = objects.get("segmentation", objects.get("polygon", []))
                    img_w, img_h = img.size
                    
                    for cat, poly in zip(categories, polygons):
                        if isinstance(cat, int):
                            # Map numeric index
                            cat_names = ["dent", "scratch", "crack", "glass_shatter", "lamp_broken", "tire_flat"]
                            cat_name = cat_names[cat] if cat < len(cat_names) else "dent"
                        else:
                            cat_name = str(cat).lower().replace(" ", "_")
                        
                        cls_idx = CLASS_MAP.get(cat_name)
                        if cls_idx is None:
                            continue
                        
                        if isinstance(poly, list) and len(poly) >= 6:
                            pts = coco_polygon_to_yolo([poly], img_w, img_h)
                            if len(pts) >= 6:
                                coords_str = " ".join(f"{v:.6f}" for v in pts)
                                label_lines.append(f"{cls_idx} {coords_str}")
                
                if label_lines:
                    stem = Path(img_name).stem
                    (lbl_dst / f"{stem}.txt").write_text("\n".join(label_lines))
                    total += 1
            
            print(f"  {our_split}: Done")
        
        print(f"[+] HuggingFace download complete. {total} labeled samples.")
        return True
        
    except Exception as e:
        print(f"[!] HuggingFace download failed: {e}")
        return False


def generate_synthetic_fallback(n_per_split=200):
    """Generate synthetic YOLO-seg data as fallback."""
    import random
    import numpy as np
    
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("[!] Pillow not available, skipping synthetic generation")
        return 0
    
    print(f"[+] Generating synthetic dataset ({n_per_split} images per split)...")
    
    splits = {"train": n_per_split, "val": n_per_split // 5, "test": n_per_split // 10}
    class_colors = {
        0: (180, 160, 150),  # dent - dark area
        1: (200, 200, 200),  # scratch - light streak  
        2: (140, 120, 100),  # peel_paint - brownish
        3: (80, 80, 80),     # broken - dark
    }
    
    total = 0
    for split, count in splits.items():
        img_dst = DATASET_DIR / "images" / split
        lbl_dst = DATASET_DIR / "labels" / split
        
        for i in range(count):
            W, H = random.choice([(640, 480), (800, 600), (1024, 768)])
            
            # Car body background
            bg = np.ones((H, W, 3), dtype=np.uint8)
            r = random.randint(120, 200)
            bg[:] = [r, r+5, r+10]
            
            img = Image.fromarray(bg)
            draw = ImageDraw.Draw(img)
            
            # Draw 1-4 random damage regions
            n_damages = random.randint(1, 4)
            label_lines = []
            
            for _ in range(n_damages):
                cls = random.randint(0, 3)
                
                # Create random convex polygon
                cx = random.randint(W//5, 4*W//5)
                cy = random.randint(H//5, 4*H//5)
                rx = random.randint(20, W//6)
                ry = random.randint(15, H//6)
                
                import math
                n_pts = random.randint(6, 12)
                angles = sorted(random.uniform(0, 2*math.pi) for _ in range(n_pts))
                pts = []
                for ang in angles:
                    x = int(cx + rx * math.cos(ang) * random.uniform(0.7, 1.0))
                    y = int(cy + ry * math.sin(ang) * random.uniform(0.7, 1.0))
                    x = max(0, min(W-1, x))
                    y = max(0, min(H-1, y))
                    pts.append((x, y))
                
                color = class_colors[cls]
                draw.polygon(pts, fill=color)
                
                # YOLO-seg line
                norm_pts = []
                for (px, py) in pts:
                    norm_pts.extend([px/W, py/H])
                coords_str = " ".join(f"{v:.6f}" for v in norm_pts)
                label_lines.append(f"{cls} {coords_str}")
            
            fname = f"synth_{split}_{i:05d}"
            img.save(img_dst / f"{fname}.jpg", quality=85)
            (lbl_dst / f"{fname}.txt").write_text("\n".join(label_lines))
            total += 1
        
        print(f"  {split}: {count} synthetic images generated")
    
    return total


def write_dataset_yaml():
    """Write dataset.yaml for YOLO training."""
    yaml_content = f"""# Car Damage Segmentation Dataset
path: {DATASET_DIR.as_posix()}
train: images/train
val: images/val
test: images/test

nc: {len(OUR_CLASSES)}
names: {OUR_CLASSES}
"""
    yaml_path = DATASET_DIR / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    print(f"[+] dataset.yaml written to {yaml_path}")
    return yaml_path


def count_samples():
    """Print how many images and labeled files exist."""
    for split in ["train", "val", "test"]:
        imgs = list((DATASET_DIR / "images" / split).glob("*.jpg")) + \
               list((DATASET_DIR / "images" / split).glob("*.png"))
        lbls = list((DATASET_DIR / "labels" / split).glob("*.txt"))
        print(f"  {split}: {len(imgs)} images, {len(lbls)} labels")


def main():
    print("=" * 60)
    print("  Car Damage Dataset Download & Conversion")
    print("=" * 60)
    
    setup_dirs()
    
    # Try HuggingFace first
    success = try_huggingface_download()
    
    # Check if we got any data
    train_imgs = list((DATASET_DIR / "images" / "train").glob("*"))
    
    if not success or len(train_imgs) < 10:
        print()
        print("[!] HuggingFace download did not produce enough data.")
        print("[+] Generating synthetic dataset instead...")
        total = generate_synthetic_fallback(n_per_split=300)
        print(f"[+] Generated {total} total samples")
    
    yaml_path = write_dataset_yaml()
    
    print()
    print("[+] Dataset Summary:")
    count_samples()
    
    print()
    print("[+] All done! Now run training:")
    print(f"    python scripts/train_seg.py --data {yaml_path} --epochs 50 --imgsz 640")
    print()
    print("    Or for better accuracy (slower):")
    print(f"    python scripts/train_seg.py --data {yaml_path} --epochs 100 --imgsz 1024")


if __name__ == "__main__":
    main()
