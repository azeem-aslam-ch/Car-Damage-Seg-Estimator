"""Run YOLOv8-seg training on the downloaded car damage dataset."""
import shutil
from pathlib import Path
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent.parent
DATA_YAML = BASE / "data" / "dataset" / "dataset.yaml"
WEIGHTS_DIR = BASE / "data" / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  YOLOv8-seg Training: Car Damage Detection")
print("=" * 60)
print(f"Dataset: {DATA_YAML}")
print(f"Device:  CPU")
print(f"Epochs:  14")
print(f"Imgsz:   640")
print()

model = YOLO("yolov8s-seg.pt")

results = model.train(
    data=str(DATA_YAML),
    epochs=14,
    imgsz=640,
    batch=8,
    patience=15,
    project=str(WEIGHTS_DIR / "runs"),
    name="car_damage_seg",
    exist_ok=True,
    device="cpu",
    verbose=True,
)

best = WEIGHTS_DIR / "runs" / "car_damage_seg" / "weights" / "best.pt"
if best.exists():
    dest = WEIGHTS_DIR / "best.pt"
    shutil.copy2(best, dest)
    print()
    print(f"[+] best.pt saved to: {dest}")
    print("[+] Restart Docker containers to use the new weights:")
    print("    docker-compose restart")
else:
    for pt in (WEIGHTS_DIR / "runs").rglob("best.pt"):
        shutil.copy2(pt, WEIGHTS_DIR / "best.pt")
        print(f"[+] Found and copied best.pt from {pt}")
        break

print()
print("[+] Training complete!")
