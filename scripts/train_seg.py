"""
train_seg.py — Train YOLOv8-seg on a car damage segmentation dataset.

Usage:
    python scripts/train_seg.py \
        --data data/dataset/dataset.yaml \
        --epochs 100 \
        --imgsz 1024 \
        --model yolov8s-seg.pt \
        --output data/weights

Requirements:
    pip install ultralytics
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8-seg car damage model")
    parser.add_argument("--data", default="data/dataset/dataset.yaml", help="Dataset YAML path")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--model", default="yolov8s-seg.pt", help="Base model weights")
    parser.add_argument("--output", default="data/weights", help="Output weights directory")
    parser.add_argument("--name", default="car_damage_seg", help="Run name")
    parser.add_argument("--device", default="", help="Device: '' for auto, 'cpu', '0', '0,1'")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Dataset YAML not found: {data_path}")
        print("   Run: python scripts/prepare_dataset.py  to generate a demo dataset.")
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics not installed. Run: pip install ultralytics")
        return

    print(f"🚀 Starting training: {args.model} for {args.epochs} epochs at {args.imgsz}px")
    model = YOLO(args.model)

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        device=args.device if args.device else None,
        task="segment",
        cache=True,
        patience=20,
        lr0=0.01,
        lrf=0.01,
        mosaic=1.0,
        augment=True,
        verbose=True,
    )

    # Copy best weights to output dir
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_pt = Path(f"runs/segment/{args.name}/weights/best.pt")
    if best_pt.exists():
        import shutil
        dest = output_dir / "best.pt"
        shutil.copy(best_pt, dest)
        print(f"✅ Best weights saved to: {dest}")
    else:
        print(f"⚠️  best.pt not found. Check runs/segment/{args.name}/weights/")

    print("✅ Training complete.")


if __name__ == "__main__":
    main()
