from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


VIDEO_PATH = Path("/home/arkin-kansra/server/video.mp4")
MODEL_PATH = Path("/home/arkin-kansra/server/yolov8m.pt")
OUTPUT_DIR = Path("/home/arkin-kansra/server/output")


def build_roi_polygon(frame_shape):
    frame_height, frame_width = frame_shape[:2]
    center_x = frame_width // 2

    top_y = int(frame_height * 0.35)
    bottom_y = frame_height - 1
    top_half_width = int(frame_width * 0.12)
    bottom_half_width = int(frame_width * 0.32)

    return np.array(
        [
            [center_x - top_half_width, top_y],
            [center_x + top_half_width, top_y],
            [center_x + bottom_half_width, bottom_y],
            [center_x - bottom_half_width, bottom_y],
        ],
        dtype=np.int32,
    )


def compute_centroid(box_xyxy):
    x1, y1, x2, y2 = box_xyxy
    centroid_x = int(round((x1 + x2) / 2))
    centroid_y = int(round((y1 + y2) / 2))
    return centroid_x, centroid_y


def is_centroid_in_roi(centroid, roi_polygon):
    return cv2.pointPolygonTest(roi_polygon, centroid, False) >= 0


def draw_roi(frame, roi_polygon):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [roi_polygon], (40, 90, 180))
    blended = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)
    cv2.polylines(blended, [roi_polygon], True, (0, 255, 255), 2)
    return blended


def annotate_detection(frame, result, detection_index, centroid):
    boxes = result.boxes
    box_xyxy = boxes.xyxy[detection_index].tolist()
    x1, y1, x2, y2 = [int(round(value)) for value in box_xyxy]
    conf = float(boxes.conf[detection_index].item())
    cls_id = int(boxes.cls[detection_index].item())
    label = result.names.get(cls_id, str(cls_id))

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.circle(frame, centroid, 4, (0, 0, 255), -1)
    cv2.putText(
        frame,
        f"{label} {conf:.2f}",
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"c={centroid}",
        (x1, min(frame.shape[0] - 10, y2 + 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    if result.masks is not None and detection_index < len(result.masks.xy):
        polygon = np.array(result.masks.xy[detection_index], dtype=np.int32)
        if len(polygon) >= 3:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [polygon], (0, 200, 0))
            frame[:] = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
            cv2.polylines(frame, [polygon], True, (0, 255, 0), 2)


def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    model.to(device)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi_polygon = build_roi_polygon(frame.shape)
        annotated_frame = draw_roi(frame.copy(), roi_polygon)
        results = model(frame, verbose=False)
        result = results[0]

        kept_detections = 0
        if result.boxes is not None and len(result.boxes) > 0:
            for detection_index, box_xyxy in enumerate(result.boxes.xyxy.tolist()):
                centroid = compute_centroid(box_xyxy)
                if not is_centroid_in_roi(centroid, roi_polygon):
                    continue

                annotate_detection(annotated_frame, result, detection_index, centroid)
                kept_detections += 1

        cv2.putText(
            annotated_frame,
            f"ROI detections: {kept_detections}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        output_path = OUTPUT_DIR / f"frame_{frame_index:06d}.jpg"
        if not cv2.imwrite(str(output_path), annotated_frame):
            raise RuntimeError(f"Failed to save frame: {output_path}")

        frame_index += 1
        if frame_index % 25 == 0:
            print(f"Saved {frame_index} frames...")

    cap.release()
    print(f"Done. Saved {frame_index} frames to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
