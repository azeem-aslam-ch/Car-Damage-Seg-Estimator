"""
evaluate.py — Evaluate a trained YOLOv8-seg model on validation set.

Usage:
    python scripts/evaluate.py \
        --model data/weights/best.pt \
        --data data/dataset/dataset.yaml \
        --imgsz 1024
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8-seg model")
    parser.add_argument("--model", default="data/weights/best.pt", help="Weights path")
    parser.add_argument("--data", default="data/dataset/dataset.yaml", help="Dataset YAML")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model weights not found: {model_path}")
        print("   Run training first: python scripts/train_seg.py")
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics not installed: pip install ultralytics")
        return

    print(f"📊 Evaluating model: {model_path} on split: {args.split}")
    model = YOLO(str(model_path))

    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        split=args.split,
        verbose=True,
    )

    print("\n─── Evaluation Results ───")
    print(f"  mAP50       : {metrics.seg.map50:.4f}")
    print(f"  mAP50-95    : {metrics.seg.map:.4f}")
    print(f"  Precision   : {metrics.seg.mp:.4f}")
    print(f"  Recall      : {metrics.seg.mr:.4f}")

    # Per-class breakdown
    if hasattr(metrics.seg, "maps"):
        classes = ["dent", "scratch", "peel_paint", "broken"]
        print("\n  Per-class mAP50-95:")
        for cls, ap in zip(classes, metrics.seg.maps):
            print(f"    {cls:<15}: {ap:.4f}")

    print("\n✅ Evaluation complete.")


if __name__ == "__main__":
    main()
