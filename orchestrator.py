import json
import time
import os
import subprocess
import threading
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import zmq

GENERAL_WORKERS = (
    "general_1",
    # "general_2",  # disabled for now
)
SPECIALIST_WORKERS = (
    "currency_worker",
    # "road_worker",  # disabled for now
)
WORKER_COLORS = {
    "general_1": (255, 0, 0),  # blue-ish in BGR
    # "general_2": (0, 165, 255),  # orange, disabled for now
    "currency_worker": (0, 255, 0),  # green
    "road_worker": (255, 0, 255),  # magenta fallback
}
REQUIRED_WORKERS = set(GENERAL_WORKERS) | set(SPECIALIST_WORKERS)
FRAME_FLUSH_COUNT = 2
TARGET_FPS = 10
FRAME_INTERVAL_SEC = 1 / TARGET_FPS
OUTPUT_LOG_PATH = Path("/home/arkin-kansra/server/output.logs")
TTS_OUTPUT_LOG_PATH = Path("/home/arkin-kansra/server/tts_output.logs")
VIDEO_SOURCE = "http://raspberrypi.local:8000/video"
FALLBACK_VIDEO_SOURCE = "test.h264"
TEST_MODE = True
STREAM = True
# Scale factor for the local display window (only affects cv2.imshow).
STREAM_SCALE = 2.5
# Rotate video frames before processing/streaming. Supported values: 0 (none), 180 (upside down).
VIDEO_ROTATION_DEG = 180
ORCHESTRATOR_PORT = int(os.getenv("ORCHESTRATOR_PORT", "5555"))
TRIGGER_CLASSES = {
    "currency_worker": {"person"},
    # "road_worker": {"red light", "traffic light"},
}
# Toggle for audio playback via tts_layer.py. Set to False to disable launching the TTS pipeline.
TTS_ENABLED = True
# When True, write spoken warnings to tts_output.logs and print them to stdout.
# Only lines that indicate a spoken warning ("[WARN] ...") are logged/printed.
TTS_SPOKEN_LOGGING = True


def parse_worker_message(message_bytes):
    try:
        return json.loads(message_bytes.decode())
    except (json.JSONDecodeError, UnicodeDecodeError):
        try:
            return {"type": message_bytes.decode()}
        except UnicodeDecodeError:
            return {"type": ""}


def resolve_video_source(source=VIDEO_SOURCE):
    if isinstance(source, int):
        return source

    if isinstance(source, Path):
        return str(source)

    if isinstance(source, str):
        stripped = source.strip()
        if stripped.isdigit():
            return int(stripped)
        return stripped

    return source


def describe_video_source(source):
    if isinstance(source, int):
        return f"webcam (device {source})"

    parsed = urlparse(str(source))
    if parsed.scheme in {"http", "https", "rtsp", "rtmp"}:
        return f"stream ({parsed.scheme})"

    return "video file"


def open_capture(source=VIDEO_SOURCE):
    resolved_source = resolve_video_source(source)

    if isinstance(resolved_source, int):
        cap = cv2.VideoCapture(resolved_source)
    else:
        cap = cv2.VideoCapture(resolved_source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(resolved_source)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def get_active_video_source():
    if TEST_MODE:
        return resolve_video_source(FALLBACK_VIDEO_SOURCE)
    return resolve_video_source(VIDEO_SOURCE)


def reconnect_capture(cap, source, reason):
    source_description = describe_video_source(source)
    print(f"[Orchestrator] {reason} Reconnecting to {source_description}...")
    cap.release()
    time.sleep(1)
    return open_capture(source)


def read_latest_frame(cap):
    for _ in range(FRAME_FLUSH_COUNT):
        if not cap.grab():
            return False, None

    return cap.retrieve()


def draw_overlay(frame, frame_state):
    """Draw worker outputs on a copy of the frame."""
    overlay = frame.copy()

    # Draw detections
    for det in frame_state.get("detections", []):
        bbox = det["bbox"]
        x1, y1, x2, y2 = map(int, bbox)
        cx, cy = det["centroid"]
        angle = det["angle_deg"]
        worker_id = det["worker_id"]
        color = WORKER_COLORS.get(worker_id, (255, 255, 255))

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.circle(overlay, (cx, cy), 4, color, -1)
        label = f"{det['label']} {det['conf']:.2f} | {angle:+.1f}°"
        cv2.putText(
            overlay,
            label,
            (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    y = 30
    line_height = 25
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6

    for worker_id in ("general_1", "currency_worker"):
        status = frame_state["results"].get(worker_id, "pending")
        text = f"{worker_id}: {status}"
        cv2.putText(overlay, text, (10, y), font, scale, (0, 255, 0), 2, cv2.LINE_AA)
        y += line_height

    return overlay


def maybe_rotate_frame(frame):
    if VIDEO_ROTATION_DEG == 0:
        return frame
    if VIDEO_ROTATION_DEG == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)

    # Fallback for unsupported values: no rotation but log once.
    print(f"[Orchestrator] Unsupported VIDEO_ROTATION_DEG={VIDEO_ROTATION_DEG}; using no rotation.")
    return frame


def show_stream(pending_frames):
    """Display the most recent frame with overlays if streaming is enabled."""
    if not STREAM or not pending_frames:
        return

    now = time.time()
    stale_ids = [
        fid
        for fid, state in pending_frames.items()
        if state.get("finalized") and now - state.get("finalized_at", now) > 1.0
    ]
    for fid in stale_ids:
        pending_frames.pop(fid, None)

    if not pending_frames:
        return

    latest_frame_id = max(pending_frames.keys())
    frame_state = pending_frames.get(latest_frame_id)
    frame = frame_state.get("frame")
    if frame is None:
        return

    annotated = draw_overlay(frame, frame_state)
    if STREAM_SCALE and STREAM_SCALE != 1.0:
        annotated = cv2.resize(
            annotated,
            None,
            fx=STREAM_SCALE,
            fy=STREAM_SCALE,
            interpolation=cv2.INTER_LINEAR,
        )
    cv2.imshow("Orchestrator Stream", annotated)
    cv2.waitKey(1)


def _serialize_detections(frame_state):
    by_worker = {}
    for det in frame_state.get("detections", []):
        worker = det.get("worker_id")
        if not worker:
            continue
        by_worker.setdefault(worker, []).append(
            {
                "label": det.get("label"),
                "confidence": det.get("conf", det.get("confidence", 0.0)),
                "centroid": det.get("centroid"),
                "angle": det.get("angle_deg", det.get("angle", 0.0)),
            }
        )
    return by_worker


def emit_tts_payload(frame_id, frame_state):
    tts_stream = frame_state.get("tts_stream")
    if tts_stream is None:
        return True

    payload = {
        "frame": frame_id,
        "timestamp": time.time(),
        "source_format": "orchestrator",
        "workers": _serialize_detections(frame_state),
    }
    try:
        tts_stream.write(json.dumps(payload) + "\n")
        tts_stream.flush()
    except (BrokenPipeError, ValueError):
        print("[Orchestrator] TTS stream closed; disabling audio output.")
        return False
    return True


def _run_tts_output_reader(pipe, stop_event):
    if pipe is None:
        return

    TTS_OUTPUT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TTS_OUTPUT_LOG_PATH.open("a", encoding="utf-8") as log_file:
        while not stop_event.is_set():
            line = pipe.readline()
            if not line:
                break
            stripped = line.strip()
            if not stripped.startswith("[WARN]"):
                continue
            log_file.write(stripped + "\n")
            log_file.flush()
            print(stripped, flush=True)


def write_log_line(frame_id, frame_state):
    results_by_worker = frame_state["results"]
    line = (
        f"frame {frame_id} : "
        f"general_1 : {results_by_worker['general_1']} , "
        # f"general_2 : {results_by_worker['general_2']} , "
        f"currency_worker : {results_by_worker['currency_worker']}"
        # , f"road_worker : {results_by_worker['road_worker']}"
    )
    with OUTPUT_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(line + "\n")

    return emit_tts_payload(frame_id, frame_state)


def send_frame(socket, worker_id, frame_id, frame_bytes):
    header = json.dumps({"frame_id": frame_id}).encode()
    socket.send_multipart([worker_id.encode(), b"", header, frame_bytes])


def maybe_finalize_frame(frame_id, pending_frames):
    frame_state = pending_frames.get(frame_id)
    if frame_state is None:
        return

    if not frame_state["awaiting"]:
        still_open = write_log_line(frame_id, frame_state)
        if not still_open:
            frame_state["tts_stream"] = None
        if STREAM:
            frame_state["finalized"] = True
            frame_state["finalized_at"] = time.time()
        else:
            del pending_frames[frame_id]


def handle_worker_response(socket, response, pending_frames, worker_configs):
    frame_id = response["frame_id"]
    worker_id = response["worker_id"]

    frame_state = pending_frames.get(frame_id)
    if frame_state is None:
        return

    frame_state["results"][worker_id] = response["summary"]
    for det in response.get("detections", []) or []:
        det["worker_id"] = worker_id
        frame_state["detections"].append(det)
    frame_state["awaiting"].discard(worker_id)

    if worker_id in GENERAL_WORKERS:
        frame_state["general_classes"].update(response["classes"])

        generals_done = all(general not in frame_state["awaiting"] for general in GENERAL_WORKERS)
        if not frame_state["specialists_dispatched"] and generals_done:
            if not frame_state.get("tts_general_sent"):
                still_open = emit_tts_payload(frame_id, frame_state)
                frame_state["tts_general_sent"] = still_open
                if not still_open:
                    frame_state["tts_stream"] = None
            frame_state["specialists_dispatched"] = True

            for specialist in SPECIALIST_WORKERS:
                trigger_labels = TRIGGER_CLASSES[specialist]
                worker_test_mode = worker_configs.get(specialist, {}).get("test_mode", False)
                triggered = worker_test_mode or any(label in frame_state["general_classes"] for label in trigger_labels)

                if triggered:
                    send_frame(socket, specialist, frame_id, frame_state["frame_bytes"])
                    frame_state["awaiting"].add(specialist)
                else:
                    frame_state["results"][specialist] = "skipped"

    maybe_finalize_frame(frame_id, pending_frames)


def main():
    OUTPUT_LOG_PATH.write_text("", encoding="utf-8")
    if TTS_SPOKEN_LOGGING:
        TTS_OUTPUT_LOG_PATH.write_text("", encoding="utf-8")
    source = get_active_video_source()
    source_description = describe_video_source(source)

    context = zmq.Context()

    socket = context.socket(zmq.ROUTER)
    socket.bind(f"tcp://*:{ORCHESTRATOR_PORT}")

    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    connected_workers = set()
    pending_frames = {}
    frame_id = 0

    print("[Orchestrator] Waiting for workers...")

    worker_configs = {}

    while connected_workers != REQUIRED_WORKERS:
        socks = dict(poller.poll(1000))

        if socket in socks:
            identity, _, message = socket.recv_multipart()
            worker_id = identity.decode()

            payload = parse_worker_message(message)
            message_type = payload.get("type")

            if message_type == "READY":
                worker_configs[worker_id] = {"test_mode": bool(payload.get("test_mode", False))}
                connected_workers.add(worker_id)
                print(f"[Orchestrator] {worker_id} connected")

    print("[Orchestrator] All required workers connected!")

    print(f"[Orchestrator] Connecting to {source_description}: {source}")
    cap = open_capture(source)

    while True:
        ret, frame = read_latest_frame(cap)
        if ret:
            frame = maybe_rotate_frame(frame)
            print(f"[Orchestrator] {source_description.capitalize()} ready!")
            break

        print(f"[Orchestrator] Waiting for {source_description} frames...")
        cap = reconnect_capture(cap, source, "Startup frame read failed.")

    # Start TTS pipeline after we have a valid frame shape (for width/height hints)
    tts_process = None
    tts_stream = None
    tts_reader_thread = None
    tts_reader_stop = threading.Event()
    if TTS_ENABLED:
        script_path = Path(__file__).with_name("tts_layer.py")
        frame_height, frame_width = frame.shape[:2]
        tts_cmd = [
            "python3",
            str(script_path),
            "--frame-width",
            str(frame_width),
            "--frame-height",
            str(frame_height),
        ]
        try:
            tts_stdout = subprocess.DEVNULL
            if TTS_SPOKEN_LOGGING:
                tts_stdout = subprocess.PIPE
            tts_process = subprocess.Popen(
                tts_cmd,
                stdin=subprocess.PIPE,
                stdout=tts_stdout,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            tts_stream = tts_process.stdin
            print(f"[Orchestrator] TTS pipeline started: {' '.join(tts_cmd)}")

            if TTS_SPOKEN_LOGGING:
                tts_reader_thread = threading.Thread(
                    target=_run_tts_output_reader,
                    args=(tts_process.stdout, tts_reader_stop),
                    name="tts-output-reader",
                    daemon=True,
                )
                tts_reader_thread.start()
        except OSError as exc:
            print(f"[Orchestrator] Failed to start TTS pipeline: {exc}")
            tts_process = None
            tts_stream = None

    print("[Orchestrator] Starting main loop")

    try:
        while True:
            ret, frame = read_latest_frame(cap)
            if not ret:
                cap = reconnect_capture(cap, source, "Frame read failed.")
                continue

            frame = maybe_rotate_frame(frame)

            frame_id += 1
            _, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()

            pending_frames[frame_id] = {
                "frame_bytes": frame_bytes,
                "frame": frame,
                "results": {
                    "general_1": "pending",
                    # "general_2": "pending",
                    "currency_worker": "pending",
                    # "road_worker": "pending",
                },
                "awaiting": set(GENERAL_WORKERS),
                "general_classes": set(),
                "specialists_dispatched": False,
                "detections": [],
                "tts_stream": tts_stream,
                "tts_general_sent": False,
            }

            for worker in GENERAL_WORKERS:
                send_frame(socket, worker, frame_id, frame_bytes)
                print(f"[Orchestrator] Sent frame {frame_id} to {worker}")

            frame_deadline = time.time() + FRAME_INTERVAL_SEC
            while time.time() < frame_deadline:
                socks = dict(poller.poll(5))

                if socket in socks:
                    try:
                        identity, _, response_bytes = socket.recv_multipart(zmq.NOBLOCK)
                        response = json.loads(response_bytes.decode())
                        handle_worker_response(socket, response, pending_frames, worker_configs)
                    except zmq.Again:
                        pass

            # If some workers didn’t respond before the deadline, mark them as timed out
            frame_state = pending_frames.get(frame_id)
            if frame_state and frame_state["awaiting"]:
                for missing_worker in list(frame_state["awaiting"]):
                    frame_state["results"][missing_worker] = "timeout"
                    frame_state["awaiting"].discard(missing_worker)
                maybe_finalize_frame(frame_id, pending_frames)

            show_stream(pending_frames)
            time.sleep(max(0, frame_deadline - time.time()))
    finally:
        tts_reader_stop.set()
        if tts_stream is not None:
            try:
                tts_stream.close()
            except Exception:
                pass
        if tts_process is not None:
            tts_process.terminate()
            try:
                tts_process.wait(timeout=1.0)
            except Exception:
                pass
        if tts_reader_thread is not None:
            tts_reader_thread.join(timeout=0.5)


if __name__ == "__main__":
    main()
