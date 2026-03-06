"""
Convert CarDD_release COCO format annotations to YOLO-seg
and organize into our dataset folder.
"""

import json
import shutil
from pathlib import Path

BASE_DIR = Path(r"E:\antigravity projects\CV-AI-Projects\Project 4 car damage detection\car_damage_seg_estimator")
CARDD_DIR = BASE_DIR / "data" / "CarDD_release" / "CarDD_release" / "CarDD_COCO"
DATASET_DIR = BASE_DIR / "data" / "dataset"

# Map CarDD categories to our 4-class system
CLASS_MAP = {
    "dent": 0,
    "scratch": 1,
    "crack": 2,          # mapped to peel_paint
    "glass shatter": 3,  # broken
    "lamp broken": 3,    # broken
    "tire flat": 3       # broken
}

OUR_CLASSES = ["dent", "scratch", "peel_paint", "broken"]


def coco_polygon_to_yolo(segmentation, img_w, img_h):
    all_points = []
    # COCO segmentation can be a list of lists (multiple polygons per instance)
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


def process_split(split_name, coco_filename, img_folder):
    print(f"\nProcessing {split_name} split...")
    coco_path = CARDD_DIR / "annotations" / coco_filename
    if not coco_path.exists():
        print(f"Skipping {split_name}, {coco_path} not found.")
        return 0

    img_src_dir = CARDD_DIR / img_folder
    
    img_dst_dir = DATASET_DIR / "images" / split_name
    lbl_dst_dir = DATASET_DIR / "labels" / split_name
    
    img_dst_dir.mkdir(parents=True, exist_ok=True)
    lbl_dst_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Categories
    cat_map = {cat["id"]: cat["name"].lower() for cat in coco["categories"]}

    # Images
    img_map = {img["id"]: img for img in coco["images"]}

    # Annotations grouped by image
    ann_by_img = {}
    for ann in coco["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    converted = 0
    skipped = 0

    for img_id, img_info in img_map.items():
        img_filename = img_info["file_name"]
        img_w = img_info["width"]
        img_h = img_info["height"]

        src_img = img_src_dir / img_filename
        if not src_img.exists():
            skipped += 1
            continue

        # Check if the image has any annotations
        anns = ann_by_img.get(img_id, [])
        label_lines = []
        for ann in anns:
            cat_name = cat_map.get(ann["category_id"], "")
            cls_idx = CLASS_MAP.get(cat_name)
            if cls_idx is None:
                continue
            if not ann.get("segmentation"):
                continue
            
            # Skip if RLE format instead of polygons
            if isinstance(ann["segmentation"], dict):
                continue
                
            pts = coco_polygon_to_yolo(ann["segmentation"], img_w, img_h)
            if len(pts) < 6:
                continue
                
            coords_str = " ".join(f"{v:.6f}" for v in pts)
            label_lines.append(f"{cls_idx} {coords_str}")

        # If we have valid labels, copy image and write label file
        if label_lines:
            dst_img = img_dst_dir / img_filename
            # We use copy() so we don't mess up original dataset
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)

            stem = Path(img_filename).stem
            label_file = lbl_dst_dir / f"{stem}.txt"
            label_file.write_text("\n".join(label_lines))
            converted += 1

    print(f"  Converted {converted} images, Skipped {skipped} (missing files or no valid polygons)")
    return converted


def main():
    # Clean previous dataset contents
    print("Cleaning previous dataset directories...")
    if (DATASET_DIR / "images").exists():
        shutil.rmtree(DATASET_DIR / "images")
    if (DATASET_DIR / "labels").exists():
        shutil.rmtree(DATASET_DIR / "labels")

    total = 0
    total += process_split("train", "instances_train2017.json", "train2017")
    total += process_split("val", "instances_val2017.json", "val2017")
    total += process_split("test", "instances_test2017.json", "test2017")

    print(f"\nTotal labeled images correctly processed: {total}")

    # Write yaml
    yaml_content = f"""# Car Damage Segmentation Dataset (CarDD Real)
path: {DATASET_DIR.as_posix()}
train: images/train
val: images/val
test: images/test

nc: {len(OUR_CLASSES)}
names: {OUR_CLASSES}
"""
    yaml_path = DATASET_DIR / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    print(f"Written dataset.yaml to {yaml_path}")


if __name__ == "__main__":
    main()
