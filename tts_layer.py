#!/usr/bin/env python3
"""Continuously parse orchestrator detections and emit audio-safe warnings.

This script is designed for a blind smart stick pipeline where multiple workers
produce detections per frame. It reads newline-delimited input from stdin and
accepts either:

1. JSON lines, for example:
   {"frame": 42, "workers": {"general_worker_1": [{"label": "car", "confidence": 0.8, "centroid": [320, 180], "angle": 0.1}]}}

2. Legacy plain-text lines similar to:
   frame 42 : general_worker_1 : car(0.80, centroid=(320,180), angle=0.1deg) , general_worker_2 : none , currency_worker : none

The engine ranks detections by danger, suppresses duplicates, and emits at most
one spoken warning per detection type in a 20-second window.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Deque, Dict, List, Optional, Tuple


DEFAULT_FRAME_WIDTH = 640
DEFAULT_FRAME_HEIGHT = 360

DEFAULT_AREA_MIN_RATIO = 0.002  # fraction of frame area
DEFAULT_MAX_AREA_RATIO = 0.18   # used for area score normalization
DEFAULT_RISK_THRESHOLD = 92.0
DEFAULT_AMBIGUITY_RATIO = 1.2

# A conservative FOV trapezium (top narrower than bottom), expressed as ratios.
DEFAULT_TRAPEZIUM = (
    (0.28, 0.22),  # top-left
    (0.72, 0.22),  # top-right
    (0.96, 1.00),  # bottom-right
    (0.04, 1.00),  # bottom-left
)


DETECTION_RE = re.compile(
    r"""
    (?P<label>.+?)\(
        (?P<confidence>\d+(?:\.\d+)?)
        ,\s*centroid=\((?P<x>-?\d+),(?P<y>-?\d+)\)
        ,\s*angle=(?P<angle>-?\d+(?:\.\d+)?)deg
    \)
    """,
    re.VERBOSE,
)


WORKER_BLOCK_RE = re.compile(
    r"(?P<worker>[a-zA-Z0-9_]+)\s*:\s*(?P<content>.*?)(?=\s*,\s*[a-zA-Z0-9_]+\s*:|$)"
)


SPECIAL_CLASS_PRIORITIES: Dict[str, int] = {
    "car": 100,
    "bus": 98,
    "truck": 98,
    "motorcycle": 95,
    "motorbike": 95,
    "train": 92,
    "bicycle": 88,
    "person": 84,
    "traffic light": 82,
    "stop sign": 80,
    "pothole": 96,
    "bench": 46,
    "potted plant": 22,
    "animal": 60,
    "dog": 62,
    "cat": 40,
    "bird": 25,
    "horse": 72,
    "cow": 70,
    "sheep": 58,
    "elephant": 70,
    "bear": 74,
    "zebra": 64,
    "giraffe": 54,
    "tv": 18,
    "refrigerator": 18,
    "chair": 18,
    "couch": 16,
    "bed": 16,
    "toilet": 15,
}


COCO_DEFAULT_PRIORITIES: Dict[str, int] = {
    "person": 84,
    "bicycle": 88,
    "car": 100,
    "motorcycle": 95,
    "airplane": 12,
    "bus": 98,
    "train": 92,
    "truck": 98,
    "boat": 10,
    "traffic light": 82,
    "fire hydrant": 26,
    "stop sign": 80,
    "parking meter": 22,
    "bench": 46,
    "bird": 25,
    "cat": 40,
    "dog": 62,
    "horse": 72,
    "sheep": 58,
    "cow": 70,
    "elephant": 70,
    "bear": 74,
    "zebra": 64,
    "giraffe": 54,
    "backpack": 14,
    "umbrella": 14,
    "handbag": 14,
    "tie": 10,
    "suitcase": 16,
    "frisbee": 5,
    "skis": 10,
    "snowboard": 10,
    "sports ball": 8,
    "kite": 8,
    "baseball bat": 10,
    "baseball glove": 10,
    "skateboard": 18,
    "surfboard": 10,
    "tennis racket": 8,
    "bottle": 14,
    "wine glass": 8,
    "cup": 8,
    "fork": 6,
    "knife": 12,
    "spoon": 6,
    "bowl": 8,
    "banana": 4,
    "apple": 4,
    "sandwich": 4,
    "orange": 4,
    "broccoli": 4,
    "carrot": 4,
    "hot dog": 4,
    "pizza": 4,
    "donut": 4,
    "cake": 4,
    "chair": 18,
    "couch": 16,
    "potted plant": 22,
    "bed": 16,
    "dining table": 18,
    "toilet": 15,
    "tv": 18,
    "laptop": 12,
    "mouse": 8,
    "remote": 8,
    "keyboard": 10,
    "cell phone": 10,
    "microwave": 10,
    "oven": 10,
    "toaster": 8,
    "sink": 10,
    "refrigerator": 18,
    "book": 6,
    "clock": 8,
    "vase": 8,
    "scissors": 12,
    "teddy bear": 4,
    "hair drier": 6,
    "toothbrush": 4,
    "pothole": 96,
    "currency": 40,
}


WORKER_PRIORITY_BONUS = {
    "general_1": 18,
    "general_2": 8,
    "general_worker_1": 18,  # legacy naming
    "general_worker_2": 8,   # legacy naming
    "currency_worker": -18,
}

INDOOR_CLASSES = {"chair", "tv", "bed", "couch", "refrigerator", "toilet"}
OUTDOOR_CLASSES = {"car", "bus", "truck", "motorcycle", "motorbike", "train", "bicycle"}


@dataclass
class Detection:
    label: str
    confidence: float
    x: float
    y: float
    angle: float
    worker: str
    area: Optional[float] = None


@dataclass
class FramePacket:
    frame: int
    timestamp: Optional[float]
    source_format: str
    workers: Dict[str, List[Detection]]


@dataclass
class WarningCandidate:
    key: str
    message: str
    score: float
    category: str
    label: str
    direction: str


class AudioOutput:
    def __init__(
        self,
        enabled: bool = True,
        queue_mode: bool = True,
        backend_name: str = "auto",
        voice: Optional[str] = None,
        rate: Optional[int] = None,
        clips_dir: Optional[str] = None,
        clips_metadata: Optional[str] = None,
        prefer_clips: bool = True,
        allow_tts_fallback: bool = False,
    ) -> None:
        self.enabled = enabled
        self.backend_name = backend_name
        self.voice = voice
        self.rate = rate
        self.clips_dir = Path(clips_dir) if clips_dir else None
        self.clips_metadata = Path(clips_metadata) if clips_metadata else None
        self.prefer_clips = prefer_clips
        self.allow_tts_fallback = allow_tts_fallback
        self.clip_map = self._load_clip_map()
        self.clip_player = self._discover_clip_player()
        self.backend = self._discover_backend() if enabled else None
        self.queue_mode = queue_mode and enabled and (self.backend is not None or self.clip_player is not None)
        self.queue: Optional[Queue[Tuple[str, str]]] = Queue() if self.queue_mode else None
        self.stop_event = threading.Event()
        self.worker: Optional[threading.Thread] = None

        if self.queue_mode and self.queue is not None:
            self.worker = threading.Thread(target=self._run, name="tts-worker", daemon=True)
            self.worker.start()
            atexit.register(self.close)

    def _discover_backend(self) -> Optional[List[str]]:
        candidates: List[List[str]]
        if self.backend_name == "auto":
            candidates = [["spd-say"], ["espeak"], ["say"]]
        else:
            candidates = [[self.backend_name]]

        for candidate in candidates:
            if shutil.which(candidate[0]):
                return candidate
        return None

    def _discover_clip_player(self) -> Optional[List[str]]:
        for candidate in (
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"],
            ["aplay", "-q"],
            ["paplay"],
            ["play", "-q"],
        ):
            if shutil.which(candidate[0]):
                return candidate
        return None

    def _load_clip_map(self) -> Dict[str, Path]:
        if not self.clips_dir or not self.clips_metadata or not self.clips_metadata.exists():
            return {}

        clip_map: Dict[str, Path] = {}
        with self.clips_metadata.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                text = row.get("text")
                filename = row.get("file")
                if not text or not filename:
                    continue
                clip_map[text] = self.clips_dir / filename
        return clip_map

    def speak(self, message: str, *, frame_id: Optional[int] = None, label: Optional[str] = None) -> None:
        if frame_id is not None or label:
            prefix_parts: List[str] = []
            if frame_id is not None:
                prefix_parts.append(f"frame {frame_id}")
            if label:
                prefix_parts.append(str(label))
            prefix = " | ".join(prefix_parts)
            print(f"[WARN] {prefix} | {message}", flush=True)
        else:
            print(f"[WARN] {message}", flush=True)
        if not self.enabled:
            return
        if self.queue_mode and self.queue is not None:
            clip_path = self.clip_map.get(message)
            self._discard_pending_audio()
            if self.prefer_clips and clip_path and clip_path.exists():
                self.queue.put(("clip", str(clip_path)))
            elif self.allow_tts_fallback:
                self.queue.put(("tts", message))
            return
        self._speak_now(message)

    def _discard_pending_audio(self) -> None:
        if self.queue is None:
            return
        while True:
            try:
                item_type, _ = self.queue.get_nowait()
            except Empty:
                break
            if item_type == "__STOP__":
                self.queue.put(("__STOP__", ""))
                break
            self.queue.task_done()

    def _speak_now(self, message: str) -> None:
        clip_path = self.clip_map.get(message)
        if self.prefer_clips and clip_path and clip_path.exists():
            self._play_clip_now(clip_path)
            return
        if not self.allow_tts_fallback:
            return
        if not self.backend:
            return
        try:
            subprocess.Popen(self._build_command(message))
        except OSError:
            pass

    def _play_clip_now(self, clip_path: Path) -> None:
        if not self.clip_player:
            return
        try:
            subprocess.Popen(self.clip_player + [str(clip_path)])
        except OSError:
            pass

    def _build_command(self, message: str) -> List[str]:
        assert self.backend is not None
        command = list(self.backend)

        if command[0] == "spd-say":
            if self.rate is not None:
                command.extend(["-r", str(self.rate)])
            if self.voice:
                command.extend(["-t", self.voice])
            command.append(message)
            return command

        if command[0] == "espeak":
            if self.rate is not None:
                command.extend(["-s", str(self.rate)])
            if self.voice:
                command.extend(["-v", self.voice])
            command.append(message)
            return command

        if command[0] == "say":
            if self.voice:
                command.extend(["-v", self.voice])
            command.append(message)
            return command

        command.append(message)
        return command

    def describe(self) -> str:
        if not self.enabled:
            return "disabled"
        if self.prefer_clips and self.clip_player is not None and self.clip_map:
            return f"clips via {self.clip_player[0]}"
        if self.prefer_clips and not self.allow_tts_fallback:
            return "clips only"
        if self.backend is None:
            return "no backend found"
        return self.backend[0]

    def _run(self) -> None:
        assert self.queue is not None
        while not self.stop_event.is_set():
            try:
                item_type, payload = self.queue.get(timeout=0.2)
            except Empty:
                continue
            if item_type == "__STOP__":
                self.queue.task_done()
                break
            if item_type == "clip":
                self._play_clip_now(Path(payload))
            else:
                self._speak_now(payload)
            self.queue.task_done()

    def close(self) -> None:
        if not self.queue_mode or self.queue is None or self.worker is None:
            return
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        self.queue.put(("__STOP__", ""))
        self.worker.join(timeout=1.0)


class WarningEngine:
    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        per_warning_cooldown: float,
        global_cooldown: float,
        min_persistence_frames: int,
        min_streak_frames: int,
        min_confidence: float,
        area_min: float,
        max_area: float,
        risk_threshold: float,
        ambiguity_ratio: float,
        trapezium: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        audio: AudioOutput,
    ) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.per_warning_cooldown = per_warning_cooldown
        self.global_cooldown = global_cooldown
        self.min_persistence_frames = min_persistence_frames
        self.min_streak_frames = min_streak_frames
        self.min_confidence = min_confidence
        self.area_min = area_min
        self.max_area = max_area
        self.risk_threshold = risk_threshold
        self.ambiguity_ratio = ambiguity_ratio
        self.trapezium = trapezium
        self.audio = audio

        self.last_spoken_at: Dict[str, float] = {}
        self.last_global_warning_at = 0.0
        self.history: Dict[str, Deque[float]] = {}
        # Tracks consecutive-frame streaks per detection key: key -> (last_frame_id, streak_length)
        self.streaks: Dict[str, Tuple[int, int]] = {}

    def process_frame(self, frame_id: int, detections: List[Detection]) -> Optional[WarningCandidate]:
        now = time.time()
        if now - self.last_global_warning_at < self.global_cooldown:
            return None

        detections = [item for item in detections if self.is_valid_detection(item)]
        detections = context_filter(detections)

        candidates: List[WarningCandidate] = []
        general_present = any(item.worker.startswith("general") for item in detections)
        hazard_present = any(
            not normalize_label(item.label).isdigit() and class_priority(normalize_label(item.label)) >= 60
            for item in detections
        )

        for detection in detections:
            if detection.confidence < self.min_confidence:
                continue
            candidate = self._build_candidate(detection, frame_id, now)
            if candidate is None:
                continue
            if detection.worker == "currency_worker" and general_present:
                continue
            if candidate.category == "currency" and hazard_present:
                continue
            if not self._can_speak(candidate.key, now):
                continue
            candidates.append(candidate)

        if not candidates:
            return None

        candidates.sort(key=lambda item: item.score, reverse=True)
        chosen = candidates[0]
        if chosen.score < self.risk_threshold:
            return None
        if len(candidates) > 1 and chosen.score < candidates[1].score * self.ambiguity_ratio:
            return None

        self.last_spoken_at[chosen.key] = now
        self.last_global_warning_at = now
        self.audio.speak(chosen.message, frame_id=frame_id, label=chosen.label)
        return chosen

    def _streak_for(self, key: str, frame_id: int) -> int:
        last_frame, streak = self.streaks.get(key, (None, 0))
        if last_frame is not None and frame_id == last_frame + 1:
            streak += 1
        else:
            streak = 1
        self.streaks[key] = (frame_id, streak)
        return streak

    def _stability_score(self, key: str, frame_id: int, now: float, confidence: float, proximity: str) -> float:
        # Time-based smoothing (kept for legacy support)
        history = self.history.setdefault(key, deque(maxlen=32))
        history.append(now)

        streak = self._streak_for(key, frame_id)
        required = self._required_streak(key, confidence, proximity)
        stability_score = min(streak / max(required, 1), 1.2)

        # Drop single-frame blips unless very confident.
        if streak < 2 and confidence < 0.92:
            return 0.0

        if confidence >= 0.92:
            return stability_score

        if streak >= self.min_streak_frames or confidence >= 0.85:
            return stability_score

        recent = [stamp for stamp in history if now - stamp <= 2.0]
        if len(recent) >= self.min_persistence_frames:
            return stability_score
        return 0.0

    def _required_streak(self, key: str, confidence: float, proximity: str) -> int:
        if proximity == "very close":
            return 2
        if confidence >= 0.90:
            return 3
        if confidence >= 0.80:
            return max(3, self.min_streak_frames - 1)
        return self.min_streak_frames

    def _can_speak(self, key: str, now: float) -> bool:
        if now - self.last_spoken_at.get(key, 0.0) < self.per_warning_cooldown:
            return False
        return True

    def is_valid_detection(self, detection: Detection) -> bool:
        # 1. Minimum bbox area (tiny = noise)
        if detection.area is not None and detection.area < self.area_min:
            return False

        # 2. Must be inside trapezium (FOV)
        if not inside_trapezium(detection.x, detection.y, self.frame_width, self.frame_height, self.trapezium):
            return False

        # 3. Class-specific constraints
        label = normalize_label(detection.label)
        if label == "car" and detection.y < 0.4 * self.frame_height:
            return False  # car too high → likely false

        return True

    def _collision_risk(
        self, priority: float, x: float, y: float, area: Optional[float], confidence: float, stability: float
    ) -> float:
        proximity_level = proximity_level_score(y, self.frame_height)  # 1..3
        centrality = centrality_score(x, self.frame_width)  # 0..1
        area_component = area_score(area, self.max_area)  # 0..1
        risk = (
            priority * 1.0
            + proximity_level * 2.0
            + centrality * 1.5
            + area_component * 2.5
            + confidence * 2.0
        )
        return risk * max(stability, 0.0)

    def _build_candidate(self, detection: Detection, frame_id: int, now: float) -> Optional[WarningCandidate]:
        label = normalize_label(detection.label)
        direction = angle_direction(
            detection.angle,
            relative_direction(detection.x, self.frame_width),
        )
        proximity = proximity_band(detection.y, self.frame_height)
        priority = class_priority(label)

        if label.isdigit():
            message = build_currency_message(label, direction, proximity)
            key = f"currency:{label}:{direction}"
            stability = self._stability_score(key, frame_id, now, detection.confidence, proximity)
            if stability <= 0.0:
                return None
            score = self._collision_risk(
                priority=18,
                x=detection.x,
                y=detection.y,
                area=detection.area,
                confidence=detection.confidence,
                stability=stability,
            ) + WORKER_PRIORITY_BONUS.get(detection.worker, 0)
            return WarningCandidate(
                key=key,
                message=message,
                score=score,
                category="currency",
                label=label,
                direction=direction,
            )

        if priority < 20:
            return None

        category = danger_category(label)
        intent_message = build_intent_message(label, direction, proximity, category)
        if intent_message is None:
            return None
        key = f"{label}:{direction}"
        stability = self._stability_score(key, frame_id, now, detection.confidence, proximity)
        if stability <= 0.0:
            return None
        worker_bonus = WORKER_PRIORITY_BONUS.get(detection.worker, 0)
        score = self._collision_risk(
            priority=float(priority),
            x=detection.x,
            y=detection.y,
            area=detection.area,
            confidence=detection.confidence,
            stability=stability,
        ) + worker_bonus

        return WarningCandidate(
            key=key,
            message=intent_message,
            score=score,
            category=category,
            label=label,
            direction=direction,
        )


def normalize_label(label: str) -> str:
    return " ".join(label.strip().lower().split())


def class_priority(label: str) -> int:
    if label in SPECIAL_CLASS_PRIORITIES:
        return SPECIAL_CLASS_PRIORITIES[label]
    if label in COCO_DEFAULT_PRIORITIES:
        return COCO_DEFAULT_PRIORITIES[label]
    if label.isdigit():
        return 18
    return 20


def danger_category(label: str) -> str:
    if label == "pothole":
        return "surface"
    if label in {"car", "bus", "truck", "motorcycle", "motorbike", "train", "bicycle"}:
        return "vehicle"
    if label == "person":
        return "pedestrian"
    if label in {"traffic light", "stop sign"}:
        return "navigation"
    if label in {"bench", "chair", "couch", "refrigerator", "tv", "potted plant"}:
        return "obstacle"
    return "object"


def relative_direction(x: float, frame_width: int) -> str:
    normalized = x / max(frame_width, 1)
    if normalized < 0.38:
        return "left"
    if normalized > 0.62:
        return "right"
    return "ahead"


def angle_direction(angle: float, fallback_direction: str) -> str:
    if angle <= -1.0:
        return "left"
    if angle >= 1.0:
        return "right"
    return fallback_direction


def proximity_band(y: float, frame_height: int) -> str:
    normalized = y / max(frame_height, 1)
    if normalized >= 0.72:
        return "very close"
    if normalized >= 0.52:
        return "close"
    return "ahead"


def proximity_score(band: str) -> int:
    return {
        "very close": 22,
        "close": 12,
        "ahead": 4,
    }.get(band, 0)


def build_warning_message(label: str, direction: str, proximity: str, category: str) -> str:
    location_phrase = combine_location_phrase(direction, proximity)

    if category == "vehicle":
        if proximity == "very close":
            return f"Caution. {label} {location_phrase}."
        return f"Warning. {label} {location_phrase}."
    if category == "surface":
        return f"Warning. Pothole {location_phrase}."
    if category == "pedestrian":
        return f"Person {location_phrase}."
    if category == "navigation":
        return f"{label.capitalize()} detected {location_phrase}."
    if category == "obstacle":
        return f"Obstacle {location_phrase}. {label} detected."
    return f"{label.capitalize()} detected {location_phrase}."


def build_currency_message(label: str, direction: str, proximity: str) -> str:
    location_phrase = combine_location_phrase(direction, proximity)
    return f"{label} currency note detected {location_phrase}."

def build_intent_message(label: str, direction: str, proximity: str, category: str) -> Optional[str]:
    # "Human-like" intents: actionable, minimal label leakage.
    if proximity == "ahead":
        return None

    if proximity == "very close" and direction == "ahead":
        return "Stop. Obstacle ahead."

    if direction == "ahead":
        return "Caution. Obstacle ahead."

    side_phrase = direction_to_phrase(direction)
    if proximity == "very close":
        return f"Obstacle {side_phrase}."
    return f"Obstacle {side_phrase}."


def direction_to_phrase(direction: str) -> str:
    if direction == "left":
        return "on your left"
    if direction == "right":
        return "on your right"
    return "ahead"


def combine_location_phrase(direction: str, proximity: str) -> str:
    direction_phrase = direction_to_phrase(direction)
    if direction == "ahead":
        if proximity == "ahead":
            return "ahead"
        return f"{proximity} ahead"
    if proximity == "ahead":
        return direction_phrase
    return f"{proximity} {direction_phrase}"

def proximity_level_score(y: float, frame_height: int) -> int:
    normalized = y / max(frame_height, 1)
    if normalized > 0.75:
        return 3
    if normalized > 0.55:
        return 2
    return 1


def centrality_score(x: float, frame_width: int) -> float:
    half = max(frame_width, 1) / 2.0
    dist = abs(x - half) / half
    return max(0.0, min(1.0, 1.0 - dist))


def area_score(area: Optional[float], max_area: float) -> float:
    if area is None or max_area <= 0:
        return 0.0
    return max(0.0, min(1.0, float(area) / max_area))


def inside_trapezium(
    x: float,
    y: float,
    frame_width: int,
    frame_height: int,
    trapezium: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]],
) -> bool:
    points = [(px * frame_width, py * frame_height) for px, py in trapezium]
    return point_in_polygon(x, y, points)


def point_in_polygon(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
    # Ray casting algorithm; poly can be convex or concave.
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersects = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / max((yj - yi), 1e-9) + xi)
        if intersects:
            inside = not inside
        j = i
    return inside


def context_filter(detections: List[Detection]) -> List[Detection]:
    indoor_present = any(normalize_label(item.label) in INDOOR_CLASSES for item in detections)
    if not indoor_present:
        return detections
    filtered: List[Detection] = []
    for item in detections:
        if normalize_label(item.label) in OUTDOOR_CLASSES:
            continue
        filtered.append(item)
    return filtered


def parse_legacy_line(line: str) -> Tuple[Optional[int], List[Detection]]:
    line = line.strip()
    if not line.lower().startswith("frame "):
        return None, []

    frame_match = re.match(r"frame\s+(\d+)\s*:\s*(.*)$", line, re.IGNORECASE)
    if not frame_match:
        return None, []

    frame_id = int(frame_match.group(1))
    rest = frame_match.group(2)
    detections: List[Detection] = []

    for match in WORKER_BLOCK_RE.finditer(rest):
        worker = match.group("worker").strip()
        content = match.group("content").strip()
        if not content or content.lower() == "none":
            continue
        for detection_text in split_detection_list(content):
            parsed = parse_detection_text(detection_text, worker)
            if parsed:
                detections.append(parsed)

    return frame_id, detections


def group_detections_by_worker(detections: List[Detection]) -> Dict[str, List[Detection]]:
    workers: Dict[str, List[Detection]] = {}
    for detection in detections:
        workers.setdefault(detection.worker, []).append(detection)
    return workers


def split_detection_list(content: str) -> List[str]:
    parts: List[str] = []
    current: List[str] = []
    depth = 0

    for char in content:
        if char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
        if char == "," and depth == 0:
            piece = "".join(current).strip()
            if piece:
                parts.append(piece)
            current = []
            continue
        current.append(char)

    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def parse_detection_text(text: str, worker: str) -> Optional[Detection]:
    match = DETECTION_RE.search(text.strip())
    if not match:
        return None
    return Detection(
        label=match.group("label").strip(),
        confidence=float(match.group("confidence")),
        x=float(match.group("x")),
        y=float(match.group("y")),
        angle=float(match.group("angle")),
        worker=worker,
    )


def parse_json_line(line: str) -> Tuple[Optional[int], List[Detection]]:
    payload = json.loads(line)
    frame_id = payload.get("frame")
    workers = payload.get("workers", {})
    detections: List[Detection] = []

    for worker, items in workers.items():
        if items in (None, "none"):
            continue
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            continue
        for item in items:
            detection = parse_json_detection(item, worker)
            if detection:
                detections.append(detection)

    return frame_id, detections


def normalize_frame_packet(line: str) -> Optional[FramePacket]:
    stripped = line.strip()
    if not stripped:
        return None

    if stripped.startswith("{"):
        payload = json.loads(stripped)
        frame_id, detections = parse_json_line(stripped)
        if frame_id is None:
            return None
        return FramePacket(
            frame=frame_id,
            timestamp=payload.get("timestamp"),
            source_format="json",
            workers=group_detections_by_worker(detections),
        )

    frame_id, detections = parse_legacy_line(stripped)
    if frame_id is None:
        return None
    return FramePacket(
        frame=frame_id,
        timestamp=None,
        source_format="legacy",
        workers=group_detections_by_worker(detections),
    )


def parse_json_detection(item: object, worker: str) -> Optional[Detection]:
    if isinstance(item, str):
        return parse_detection_text(item, worker)
    if not isinstance(item, dict):
        return None

    label = item.get("label") or item.get("class") or item.get("name")
    confidence = item.get("confidence", item.get("score"))
    centroid = item.get("centroid", item.get("center"))
    angle = item.get("angle", 0.0)
    area = item.get("area")
    if area is None:
        area = extract_area(item)

    if label is None or confidence is None or centroid is None:
        return None

    if isinstance(centroid, dict):
        x = centroid.get("x")
        y = centroid.get("y")
    else:
        try:
            x, y = centroid
        except (TypeError, ValueError):
            return None

    if x is None or y is None:
        return None

    return Detection(
        label=str(label),
        confidence=float(confidence),
        x=float(x),
        y=float(y),
        angle=float(angle),
        worker=worker,
        area=float(area) if area is not None else None,
    )

def extract_area(item: dict) -> Optional[float]:
    # Common shapes:
    # - area: number
    # - bbox/box/xyxy: [x1,y1,x2,y2]
    # - xywh/bbox: [x,y,w,h]
    # - width/height or w/h
    candidates = (
        item.get("bbox"),
        item.get("box"),
        item.get("xyxy"),
        item.get("xywh"),
    )
    for candidate in candidates:
        area = area_from_box(candidate)
        if area is not None:
            return area

    width = item.get("width", item.get("w"))
    height = item.get("height", item.get("h"))
    if width is not None and height is not None:
        try:
            width_f = float(width)
            height_f = float(height)
        except (TypeError, ValueError):
            return None
        if width_f >= 0 and height_f >= 0:
            return width_f * height_f
    return None


def area_from_box(candidate: object) -> Optional[float]:
    if candidate is None:
        return None
    if isinstance(candidate, dict):
        x1 = candidate.get("x1")
        y1 = candidate.get("y1")
        x2 = candidate.get("x2")
        y2 = candidate.get("y2")
        if None not in (x1, y1, x2, y2):
            try:
                w = float(x2) - float(x1)
                h = float(y2) - float(y1)
            except (TypeError, ValueError):
                return None
            if w > 0 and h > 0:
                return w * h
        w = candidate.get("w", candidate.get("width"))
        h = candidate.get("h", candidate.get("height"))
        if w is not None and h is not None:
            try:
                w_f = float(w)
                h_f = float(h)
            except (TypeError, ValueError):
                return None
            if w_f >= 0 and h_f >= 0:
                return w_f * h_f
        return None

    if isinstance(candidate, (list, tuple)) and len(candidate) == 4:
        try:
            a, b, c, d = (float(candidate[0]), float(candidate[1]), float(candidate[2]), float(candidate[3]))
        except (TypeError, ValueError):
            return None
        # Prefer xyxy if it looks like it (positive width/height after subtraction).
        w_xyxy = c - a
        h_xyxy = d - b
        if w_xyxy > 0 and h_xyxy > 0:
            return w_xyxy * h_xyxy
        # Fall back to xywh.
        if c >= 0 and d >= 0:
            return c * d
    return None


def parse_line(line: str) -> Tuple[Optional[int], List[Detection]]:
    packet = normalize_frame_packet(line)
    if packet is None:
        return None, []

    detections: List[Detection] = []
    for items in packet.workers.values():
        detections.extend(items)
    return packet.frame, detections


def dump_priority_table() -> str:
    ordered = sorted(COCO_DEFAULT_PRIORITIES.items(), key=lambda item: (-item[1], item[0]))
    lines = ["Class priority table:"]
    for label, priority in ordered:
        lines.append(f"{label}: {priority}")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Blind smart stick warning engine")
    parser.add_argument("--frame-width", type=int, default=DEFAULT_FRAME_WIDTH)
    parser.add_argument("--frame-height", type=int, default=DEFAULT_FRAME_HEIGHT)
    parser.add_argument("--per-warning-cooldown", type=float, default=20.0)
    parser.add_argument("--global-cooldown", type=float, default=1.2)
    parser.add_argument("--min-persistence-frames", type=int, default=2)
    parser.add_argument("--min-streak-frames", type=int, default=4, help="Minimum consecutive frames before a warning is spoken")
    parser.add_argument("--min-confidence", type=float, default=0.30)
    parser.add_argument("--area-min", type=float, help="Minimum bbox area in pixels (tiny = noise)")
    parser.add_argument("--area-min-ratio", type=float, default=DEFAULT_AREA_MIN_RATIO, help="Fallback area-min as ratio of frame area")
    parser.add_argument("--max-area-ratio", type=float, default=DEFAULT_MAX_AREA_RATIO, help="Area score normalization ratio of frame area")
    parser.add_argument("--risk-threshold", type=float, default=DEFAULT_RISK_THRESHOLD, help="Minimum risk required to speak")
    parser.add_argument("--ambiguity-ratio", type=float, default=DEFAULT_AMBIGUITY_RATIO, help="Suppress if top-2 risks are too close")
    parser.add_argument("--mute", action="store_true", help="Do not call a TTS backend")
    parser.add_argument("--verbose", action="store_true", help="Print parsed frame details")
    parser.add_argument(
        "--tts-backend",
        choices=["auto", "spd-say", "espeak", "say"],
        default="auto",
        help="Text to speech backend to use",
    )
    parser.add_argument("--tts-voice", help="Optional voice name for the selected TTS backend")
    parser.add_argument("--tts-rate", type=int, help="Optional speaking rate for the selected TTS backend")
    parser.add_argument("--clips-dir", default="tts_clips", help="Directory containing generated audio clips")
    parser.add_argument("--clips-metadata", default="tts_clips/metadata.csv", help="CSV mapping of phrase text to clip filename")
    parser.add_argument(
        "--allow-live-tts-fallback",
        action="store_true",
        help="Use live TTS only when a matching clip is not found",
    )
    parser.add_argument("--print-priority-table", action="store_true", help="Print the full ranked class table and exit")
    parser.add_argument("--dump-normalized-json", action="store_true", help="Print each parsed frame as normalized JSON")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    if args.print_priority_table:
        print(dump_priority_table())
        return 0

    frame_area = max(args.frame_width, 1) * max(args.frame_height, 1)
    area_min = float(args.area_min) if args.area_min is not None else frame_area * float(args.area_min_ratio)
    max_area = frame_area * float(args.max_area_ratio)

    engine = WarningEngine(
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        per_warning_cooldown=args.per_warning_cooldown,
        global_cooldown=args.global_cooldown,
        min_persistence_frames=args.min_persistence_frames,
        min_streak_frames=args.min_streak_frames,
        min_confidence=args.min_confidence,
        area_min=area_min,
        max_area=max_area,
        risk_threshold=args.risk_threshold,
        ambiguity_ratio=args.ambiguity_ratio,
        trapezium=DEFAULT_TRAPEZIUM,
        audio=AudioOutput(
            enabled=not args.mute,
            queue_mode=True,
            backend_name=args.tts_backend,
            voice=args.tts_voice,
            rate=args.tts_rate,
            clips_dir=args.clips_dir,
            clips_metadata=args.clips_metadata,
            prefer_clips=True,
            allow_tts_fallback=args.allow_live_tts_fallback,
        ),
    )

    if args.verbose:
        print(f"[TTS] {engine.audio.describe()}", file=sys.stderr, flush=True)

    try:
        for raw_line in sys.stdin:
            try:
                packet = normalize_frame_packet(raw_line)
            except json.JSONDecodeError as exc:
                print(f"[ERROR] Invalid JSON line: {exc}", file=sys.stderr, flush=True)
                continue

            if packet is None:
                continue

            detections: List[Detection] = []
            for items in packet.workers.values():
                detections.extend(items)

            if args.verbose:
                print(
                    f"[FRAME {packet.frame}] parsed {len(detections)} detections from {packet.source_format}",
                    file=sys.stderr,
                    flush=True,
                )

            if args.dump_normalized_json:
                print(
                    json.dumps(
                        {
                            "frame": packet.frame,
                            "timestamp": packet.timestamp,
                            "source_format": packet.source_format,
                            "workers": {
                                worker: [
                                    {
                                        "label": item.label,
                                        "confidence": item.confidence,
                                        "centroid": [item.x, item.y],
                                        "angle": item.angle,
                                        "area": item.area,
                                    }
                                    for item in items
                                ]
                                for worker, items in packet.workers.items()
                            },
                        }
                    ),
                    flush=True,
                )

            engine.process_frame(packet.frame, detections)
    finally:
        engine.audio.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
