import json
import time
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
import zmq
from ultralytics import YOLO

# ROAD_WORKER_ID = "road_worker"  # disabled for now while the road specialist is inactive


class Worker:
    def __init__(
        self,
        worker_id,
        server_addr=None,
        model_path="models/yolov8m.onnx",
        test_mode=False,
    ):
        self.worker_id = worker_id
        self.worker_identity = worker_id.encode()
        self.test_mode = bool(test_mode)

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.IDENTITY, self.worker_identity)
        if server_addr is None:
            port = os.getenv("ORCHESTRATOR_PORT", "5555")
            server_addr = f"tcp://localhost:{port}"

        self.socket.connect(server_addr)

        print(f"[{worker_id}] Connected to orchestrator")

        ort_providers = ort.get_available_providers()
        ort_device = "cuda" if "CUDAExecutionProvider" in ort_providers else "cpu"
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"

        resolved_model_path = Path(model_path)
        if not resolved_model_path.is_absolute():
            resolved_model_path = Path(__file__).resolve().parent / resolved_model_path

        print(f"[{worker_id}] Loading model: {resolved_model_path}")
        self.model = YOLO(str(resolved_model_path))
        if resolved_model_path.suffix.lower() == ".onnx":
            self.engine = "onnxruntime"
            self.device = ort_device
            print(
                f"[{worker_id}] Using ONNXRuntime ({self.device}) | providers={ort_providers}"
            )
        else:
            self.engine = "torch"
            self.device = torch_device
            print(f"[{worker_id}] Using torch device: {self.device}")
            self.model.to(self.device)

        ready_payload = {
            "type": "READY",
            "worker_id": self.worker_id,
            "test_mode": self.test_mode,
        }
        self.socket.send_multipart([b"", json.dumps(ready_payload).encode()])

    def compute_centroid_and_angle(self, xyxy, frame_shape):
        x1, y1, x2, y2 = xyxy
        centroid_x = (x1 + x2) / 2
        centroid_y = (y1 + y2) / 2

        frame_height, frame_width = frame_shape[:2]
        frame_center_x = frame_width / 2

        delta_x = centroid_x - frame_center_x
        max_offset_x = max(frame_width / 2, 1)
        normalized_offset = max(-1.0, min(1.0, delta_x / max_offset_x))
        angle_deg = normalized_offset * 90.0

        return int(round(centroid_x)), int(round(centroid_y)), angle_deg

    def format_detections(self, result, frame_shape):
        """Turn model result into summary strings and structured details.

        Also computes bbox area normalized to frame area (0-1) for logging/analysis.
        """
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return "none", 0, [], []

        frame_h, frame_w = frame_shape[:2]
        frame_area = max(frame_w * frame_h, 1)

        detections = []
        classes = []
        details = []
        xyxy_values = boxes.xyxy.tolist()
        for cls_id, conf, xyxy in zip(boxes.cls.tolist(), boxes.conf.tolist(), xyxy_values):
            label = result.names.get(int(cls_id), str(int(cls_id)))
            x1, y1, x2, y2 = xyxy
            bbox_area_norm = max((x2 - x1), 0) * max((y2 - y1), 0) / frame_area
            centroid_x, centroid_y, angle_deg = self.compute_centroid_and_angle(xyxy, frame_shape)

            # Summary string for logs: omit centroid to keep logs concise.
            detections.append(
                f"{label}({conf:.2f}, angle={angle_deg:.1f}deg, area={bbox_area_norm:.4f})"
            )
            classes.append(label)
            details.append(
                {
                    "label": label,
                    "conf": float(conf),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "centroid": [int(centroid_x), int(centroid_y)],
                    "angle_deg": float(angle_deg),
                    "area_norm": float(bbox_area_norm),
                }
            )

        if not detections:
            return "none", 0, [], []

        return ", ".join(detections), len(detections), sorted(set(classes)), details

    def send_summary(self, frame_id, summary, num_det, classes, latency_ms, detections, triggered=True):
        payload = {
            "frame_id": frame_id,
            "worker_id": self.worker_id,
            "summary": summary,
            "num_det": num_det,
            "classes": classes,
            "latency_ms": latency_ms,
            "triggered": triggered,
            "detections": detections,
        }
        self.socket.send_multipart([b"", json.dumps(payload).encode()])

    def run(self):
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)

        while True:
            socks = dict(poller.poll(1000))

            if self.socket in socks:
                _, header_bytes, frame_bytes = self.socket.recv_multipart()
                header = json.loads(header_bytes.decode())
                frame_id = header["frame_id"]

                start = time.time()

                np_arr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is None:
                    summary = "decode_failed"
                    latency = int((time.time() - start) * 1000)
                    print(f"[{self.worker_id}] Frame {frame_id} | {summary} | {latency} ms")
                    self.send_summary(frame_id, summary, 0, [], latency)
                    continue

                predict_kwargs = {"verbose": False}
                if self.engine == "torch":
                    predict_kwargs["device"] = self.device

                results = self.model(frame, **predict_kwargs)
                detection_summary, num_det, classes, det_details = self.format_detections(results[0], frame.shape)
                latency = int((time.time() - start) * 1000)

                print(
                    f"[{self.worker_id}] Frame {frame_id} | {num_det} detections | "
                    f"{detection_summary} | {latency} ms"
                )

                self.send_summary(
                    frame_id,
                    detection_summary,
                    num_det,
                    classes,
                    latency,
                    det_details,
                )

            time.sleep(0.01)
