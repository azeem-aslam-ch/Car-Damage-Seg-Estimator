"""
infer.py — Run single-image inference via the YOLOv8-seg model and print JSON results.

Usage:
    python scripts/infer.py --image data/sample_images/sample_car.jpg \
                            --model data/weights/best.pt \
                            --conf 0.25
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run car damage inference on a single image")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--model", default="data/weights/best.pt", help="Weights path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--out-dir", default="data/outputs", help="Output directory for overlay")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)

    # ── Add backend to sys.path so we can import services ──────────────────────
    backend_path = Path(__file__).resolve().parent.parent / "backend"
    if str(backend_path) not in sys.path:
        sys.path.insert(0, str(backend_path))

    try:
        from ultralytics import YOLO
        import numpy as np
    except ImportError:
        print("❌ Missing packages. Run: pip install ultralytics numpy")
        sys.exit(1)

    # Resolve model path
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"⚠️  Custom weights not found at {model_path}, using pretrained yolov8s-seg.pt")
        model_path = "yolov8s-seg.pt"

    print(f"🔍 Running inference: {image_path}")
    model = YOLO(str(model_path))
    results = model.predict(
        source=str(image_path),
        conf=args.conf,
        iou=args.iou,
        task="segment",
        verbose=False,
    )

    result = results[0]
    img_h, img_w = result.orig_shape[:2]
    image_area = img_h * img_w
    class_names = result.names

    output = []
    if result.boxes and len(result.boxes) > 0:
        for idx, box in enumerate(result.boxes):
            cls_id = int(box.cls.item())
            cls_name = class_names.get(cls_id, f"class_{cls_id}")
            conf = float(box.conf.item())
            mask_px = 0
            if result.masks and idx < len(result.masks):
                mask_data = result.masks[idx].data.cpu().numpy().squeeze().astype(bool)
                mask_px = int(mask_data.sum())
            damage_pct = mask_px / image_area * 100

            output.append({
                "class": cls_name,
                "confidence": round(conf, 4),
                "mask_area_px": mask_px,
                "image_area_px": image_area,
                "damage_area_percent": round(damage_pct, 4),
            })

    # Save overlay
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = out_dir / f"infer_{image_path.stem}_overlay.jpg"
    result.save(filename=str(overlay_path))

    summary = {
        "image": str(image_path),
        "total_instances": len(output),
        "overlay_saved": str(overlay_path),
        "detections": output,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
