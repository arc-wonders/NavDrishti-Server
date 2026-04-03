"""Export project models to ONNX and place them under the models/ directory."""
from __future__ import annotations

from pathlib import Path
import shutil

import torch
from ultralytics import YOLO

OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Source models to export -> target filenames inside models/
MODEL_SOURCES = [
    # Ultralytics YOLO26 weights are typically named "yolo26l.pt". Accept legacy "yolov26l.pt" too.
    Path("yolo26l.pt"),
    Path("yolov26l.pt"),
    Path("yolov8m.pt"),
    Path("yolov8n.pt"),
    Path("yolov8n-face.pt"),
    Path("models/pothole_seg.pt"),
    Path("models/currency_model.pt"),
]


def export_one(weights_path: Path) -> Path | None:
    if not weights_path.exists():
        print(f"[skip] {weights_path} not found")
        return None

    target_name = f"{weights_path.stem}.onnx"
    target_path = OUTPUT_DIR / target_name

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[export] {weights_path} -> {target_path} (device={device})")

    model = YOLO(str(weights_path))
    exported_path = model.export(
        format="onnx",
        device=device,
        dynamic=True,
        opset=12,
        simplify=True,
        imgsz=model.model.args.get("imgsz", 640) if hasattr(model, "model") else 640,
        nms=True,
        batch=1,
    )

    exported_path = Path(exported_path)

    if exported_path.resolve() != target_path.resolve():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(exported_path), target_path)

    print(f"[done] {target_path}")
    return target_path


def main() -> int:
    created = [p for p in (export_one(path) for path in MODEL_SOURCES) if p]
    if not created:
        print("No models exported.")
        return 1

    print("\nExported ONNX models:")
    for path in created:
        print(f" - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
