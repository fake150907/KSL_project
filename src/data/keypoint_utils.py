"""Keypoint helpers for fixed MediaPipe sequences.

The current project standard is:
pose33 + left_hand21 + right_hand21.

WORD uses [x, y, z] per point: 75 x 3 = 225 features.
SENTENCE v2 uses [x, y, z, c] per point: 75 x 4 = 300 features.
"""

from __future__ import annotations

from typing import Any

import numpy as np


COORD_KEYS = ("x", "y", "z")
POSE_POINTS = 33
HAND_POINTS = 21
TOTAL_POINTS = POSE_POINTS + HAND_POINTS + HAND_POINTS
FEATURE_DIMS = 3
FEATURE_COUNT = TOTAL_POINTS * FEATURE_DIMS
SENTENCE_FEATURE_DIMS = 4
SENTENCE_FEATURE_COUNT = TOTAL_POINTS * SENTENCE_FEATURE_DIMS
SENTENCE_T_MAX = 128
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


def _is_number_list(obj: Any) -> bool:
    return isinstance(obj, list) and bool(obj) and all(isinstance(v, (int, float)) for v in obj)


def _collect_points(obj: Any) -> list[list[float]]:
    points: list[list[float]] = []
    if isinstance(obj, dict):
        if all(k in obj for k in ("x", "y")):
            points.append([float(obj.get(k, 0.0) or 0.0) for k in COORD_KEYS])
        else:
            for value in obj.values():
                points.extend(_collect_points(value))
    elif isinstance(obj, list):
        if _is_number_list(obj):
            step = 3 if len(obj) >= 3 else 2
            for idx in range(0, len(obj) - step + 1, step):
                xyz = [float(v) for v in obj[idx : idx + step]]
                while len(xyz) < 3:
                    xyz.append(0.0)
                points.append(xyz[:3])
        else:
            for value in obj:
                points.extend(_collect_points(value))
    return points


def extract_frames(record: Any) -> list[np.ndarray]:
    if isinstance(record, dict):
        people = record.get("people")
        if isinstance(people, dict):
            points: list[list[float]] = []
            for key in ("pose_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"):
                points.extend(_collect_points(people.get(key, [])))
            if points:
                return [np.asarray(points, dtype=np.float32)]
        for key in ("frames", "data", "annotations", "keypoints"):
            value = record.get(key)
            if isinstance(value, list) and value and not _is_number_list(value):
                frames = [np.asarray(_collect_points(item), dtype=np.float32) for item in value]
                frames = [f for f in frames if f.size > 0]
                if frames:
                    return frames
    points = np.asarray(_collect_points(record), dtype=np.float32)
    return [points] if points.size else []


def _fit_frame_to_layout(frame: np.ndarray, feature_dims: int = FEATURE_DIMS) -> np.ndarray:
    fixed = np.zeros((TOTAL_POINTS, feature_dims), dtype=np.float32)
    if frame.size == 0:
        return fixed
    current = np.nan_to_num(frame.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if current.ndim == 1:
        source_dims = feature_dims
        if current.size % feature_dims != 0:
            source_dims = SENTENCE_FEATURE_DIMS if current.size % SENTENCE_FEATURE_DIMS == 0 else FEATURE_DIMS
        current = current.reshape(-1, source_dims)
    cols = min(feature_dims, current.shape[1])
    rows = min(TOTAL_POINTS, current.shape[0])
    fixed[:rows, :cols] = current[:rows, :cols]
    if feature_dims >= SENTENCE_FEATURE_DIMS and current.shape[1] < SENTENCE_FEATURE_DIMS:
        valid_xyz = np.any(fixed[:rows, :FEATURE_DIMS] != 0.0, axis=1)
        fixed[:rows, 3] = valid_xyz.astype(np.float32)
    return fixed


def normalize_frame(frame: np.ndarray, feature_dims: int = FEATURE_DIMS) -> np.ndarray:
    frame = _fit_frame_to_layout(frame, feature_dims=feature_dims)
    left = frame[LEFT_SHOULDER].copy()
    right = frame[RIGHT_SHOULDER].copy()

    if not np.any(left) or not np.any(right):
        return frame

    center = (left + right) / 2.0
    scale = float(np.linalg.norm(left[:2] - right[:2]))
    if scale < 1e-6:
        return frame

    valid = np.any(frame != 0.0, axis=1)
    normalized = frame.copy()
    normalized[valid, :FEATURE_DIMS] = (normalized[valid, :FEATURE_DIMS] - center[:FEATURE_DIMS]) / scale
    normalized[~valid] = 0.0
    return normalized


def sequence_to_tensor(
    frames: list[np.ndarray],
    sequence_length: int,
    feature_dims: int = FEATURE_DIMS,
    normalize: bool = True,
    normalize_mode: str | None = None,
    pad_mode: str = "repeat",
    resample: bool = True,
) -> np.ndarray:
    feature_count = TOTAL_POINTS * feature_dims
    if not frames:
        return np.zeros((sequence_length, feature_count), dtype=np.float32)

    frame_features: list[np.ndarray] = []
    for frame in frames:
        current = _fit_frame_to_layout(frame, feature_dims=feature_dims)
        should_normalize = normalize and str(normalize_mode or "").lower() not in {"none", "false", "0"}
        if should_normalize:
            current = normalize_frame(current, feature_dims=feature_dims)
        frame_features.append(current[:, :feature_dims].reshape(-1))

    seq = np.stack(frame_features).astype(np.float32)
    if len(seq) == sequence_length:
        return seq
    if len(seq) > sequence_length:
        if not resample:
            return seq[:sequence_length]
        indices = np.linspace(0, len(seq) - 1, sequence_length).round().astype(int)
        return seq[indices]

    if pad_mode == "zero":
        pad = np.zeros((sequence_length - len(seq), feature_count), dtype=np.float32)
    else:
        last = seq[-1:]
        pad = np.repeat(last, sequence_length - len(seq), axis=0)
    return np.concatenate([seq, pad], axis=0)


def mediapipe_landmarks_to_frame(results: Any, layout: str = "mediapipe_xyz") -> np.ndarray:
    use_confidence = layout in {"mediapipe_xyzc", "xyzc", "sentence_v2"}
    points: list[list[float]] = []
    pose_landmarks = getattr(results, "pose_landmarks", None)
    if pose_landmarks is None:
        points.extend([[0.0, 0.0, 0.0, 0.0] if use_confidence else [0.0, 0.0, 0.0] for _ in range(POSE_POINTS)])
    else:
        if use_confidence:
            pose_points = [
                [float(lm.x), float(lm.y), float(lm.z), float(getattr(lm, "visibility", 1.0) or 0.0)]
                for lm in pose_landmarks.landmark[:POSE_POINTS]
            ]
        else:
            pose_points = [
                [float(lm.x), float(lm.y), float(lm.z)]
                for lm in pose_landmarks.landmark[:POSE_POINTS]
            ]
        while len(pose_points) < POSE_POINTS:
            pose_points.append([0.0, 0.0, 0.0, 0.0] if use_confidence else [0.0, 0.0, 0.0])
        points.extend(pose_points)

    for attr in ("left_hand_landmarks", "right_hand_landmarks"):
        landmarks = getattr(results, attr, None)
        if landmarks is None:
            points.extend([[0.0, 0.0, 0.0, 0.0] if use_confidence else [0.0, 0.0, 0.0] for _ in range(HAND_POINTS)])
        else:
            if use_confidence:
                hand_points = [
                    [float(lm.x), float(lm.y), float(lm.z), 1.0]
                    for lm in landmarks.landmark[:HAND_POINTS]
                ]
            else:
                hand_points = [
                    [float(lm.x), float(lm.y), float(lm.z)]
                    for lm in landmarks.landmark[:HAND_POINTS]
                ]
            while len(hand_points) < HAND_POINTS:
                hand_points.append([0.0, 0.0, 0.0, 0.0] if use_confidence else [0.0, 0.0, 0.0])
            points.extend(hand_points)

    return np.asarray(points, dtype=np.float32)
