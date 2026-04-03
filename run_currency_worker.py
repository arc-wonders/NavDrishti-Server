import argparse

from worker import Worker


def parse_args():
    parser = argparse.ArgumentParser(description="Run the currency worker.")
    parser.add_argument(
        "--model-path",
        default="models/currency_model.onnx",
        help="Path to the model weights for the currency worker.",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Bypass trigger logic so frames are sent to this worker even if no trigger labels are detected.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    worker = Worker(
        "currency_worker",
        model_path=args.model_path,
        test_mode=args.test_mode,
    )
    worker.run()
