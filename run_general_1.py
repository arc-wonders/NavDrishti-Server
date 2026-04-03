import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run general worker 1.")
    parser.add_argument(
        "--model-path",
        default="models/yolo26l.onnx",
        help="Path to the model weights for general worker 1 (ONNX recommended).",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Bypass trigger logic so frames are sent to this worker even if no trigger labels are detected.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = Path(__file__).resolve().parent / model_path

    if not model_path.exists():
        fallback = Path(__file__).resolve().parent / "models/yolov8m.onnx"
        print(
            f"[general_1] WARNING: model not found: {model_path}. Falling back to {fallback}.",
            file=sys.stderr,
            flush=True,
        )
        args.model_path = str(fallback)

    from worker import Worker

    worker = Worker("general_1", model_path=args.model_path, test_mode=args.test_mode)
    worker.run()
