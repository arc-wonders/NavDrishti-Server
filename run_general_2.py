from worker import Worker

if __name__ == "__main__":
    worker = Worker("general_2", model_path="models/pothole_seg.onnx")
    worker.run()
