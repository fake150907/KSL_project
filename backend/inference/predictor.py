from __future__ import annotations

from collections import Counter, deque
from typing import Any

import inference.model_state as state
from inference.model_state import (
    SCENARIO_LABEL_ACC,
    SCENARIO_DEFAULT_SENTENCE_ACC,
    SCENARIO_DEFAULT_WORD_ACC,
    np,
    torch,
    sequence_to_tensor,
)
def softmax(x: Any) -> Any:
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / np.sum(exp)

def display_label_for(label: str | None) -> str | None:
    if not label:
        return None
    if label.startswith("SEN") and label in state.SCENARIO_LOOKUP:
        return state.SCENARIO_LOOKUP[label]
    return state.LABEL_DISPLAY_MAP.get(label, state.SCENARIO_LOOKUP.get(label, label))

def scenario_label_acc(label: str | None) -> float:
    if not label:
        return 0.0
    if label in SCENARIO_LABEL_ACC:
        return SCENARIO_LABEL_ACC[label]
    if label.startswith("WORD"):
        return SCENARIO_DEFAULT_WORD_ACC
    if label.startswith("SEN"):
        return SCENARIO_DEFAULT_SENTENCE_ACC
    return 0.5

def get_session_window(client_id: str) -> list:
    if client_id not in state.memory_windows:
        state.memory_windows[client_id] = []
    return state.memory_windows[client_id]

def get_session_misses(client_id: str) -> int:
    return int(state.memory_misses.get(client_id, 0))

def set_session_misses(client_id: str, misses: int) -> None:
    state.memory_misses[client_id] = max(0, misses)

def get_prediction_history(client_id: str) -> deque:
    if client_id not in state.memory_predictions:
        state.memory_predictions[client_id] = deque(maxlen=5)
    return state.memory_predictions[client_id]

def summarize_prediction_history(history: deque, min_count: int = 1) -> tuple[str | None, float]:
    if not history:
        return None, 0.0
    counts = Counter(item["label"] for item in history)
    label, count = counts.most_common(1)[0]
    if count < min_count:
        latest = history[-1]
        return latest["label"], float(latest["confidence"])
    confidences = [float(item["confidence"]) for item in history if item["label"] == label]
    return label, float(np.mean(confidences)) if confidences else 0.0

def normalize_model_type(model_type: str) -> str:
    aliases = {
        "word_v2":    "word_v2",
        "sentence_v2":"sentence_v2",
        "word":       "word_v2",
        "sentence":   "sentence_v2",
        "sequence":   "cnn_gru",
        "cnn-gru":    "cnn_gru",
        "cnn+gru":    "cnn_gru",
        "cnn_gru":    "cnn_gru",
        "cnn_1d":     "cnn_1d",
        "cnn-1d":     "cnn_1d",
        "1d-cnn":     "cnn_1d",
        "lstm":       "lstm",
    }
    return aliases.get(model_type.lower(), "cnn_gru")


def get_sequence_model_bundle(model_type: str) -> tuple[Any, list[str] | None]:
    if model_type in state.sequence_models:
        return state.sequence_models[model_type], state.sequence_labels.get(model_type)
    selected = normalize_model_type(model_type)
    if selected not in state.sequence_models:
        selected = "cnn_gru"
    return state.sequence_models.get(selected), state.sequence_labels.get(selected)

def top_predictions_from_probs(
    probs: Any,
    labels: list[str],
    top_k: int = 3,
) -> tuple[str | None, float, list[dict]]:
    if probs.size == 0 or not labels:
        return None, 0.0, []
    pred = int(np.argmax(probs))
    top_idx = np.argsort(probs)[::-1][:top_k]
    top = [
        {
            "label": labels[i],
            "display_label": display_label_for(labels[i]),
            "confidence": float(probs[i]),
        }
        for i in top_idx
    ]
    return labels[pred], float(probs[pred]), top

def align_tensor_features(tensor: Any, expected: int) -> Any:
    if expected <= 0 or tensor.shape[1] == expected:
        return tensor
    if tensor.shape[1] > expected:
        return tensor[:, :expected]
    aligned = np.zeros((tensor.shape[0], expected), dtype=tensor.dtype)
    aligned[:, : tensor.shape[1]] = tensor
    return aligned

def smooth_segment_frames(frames: list) -> list:
    if len(frames) < 3:
        return frames
    smoothed: list = []
    for idx, frame in enumerate(frames):
        if idx == 0 or idx == len(frames) - 1:
            smoothed.append(frame)
            continue
        smoothed.append(
            (frames[idx - 1] * 0.2 + frame * 0.6 + frames[idx + 1] * 0.2).astype(np.float32)
        )
    return smoothed

def center_trim_segment_frames(frames: list, trim_ratio: float = 0.1) -> list:
    if len(frames) < 12:
        return frames
    trim = max(1, int(round(len(frames) * trim_ratio)))
    if len(frames) - trim * 2 < 8:
        return frames
    return frames[trim:-trim]

def build_segment_tta_variants(frames: list) -> list[list]:
    variants = [frames, smooth_segment_frames(frames)]
    trimmed = center_trim_segment_frames(frames)
    if len(trimmed) != len(frames):
        variants.append(trimmed)
        variants.append(smooth_segment_frames(trimmed))
    return variants

def frames_to_model_tensor(model_type: str, frames: list, sequence_length: int) -> Any:
    tensor = sequence_to_tensor(frames, sequence_length)
    model = state.sequence_models.get(model_type) or state.sequence_models.get("cnn_gru")
    if model is not None:
        tensor = align_tensor_features(tensor, int(model.input_size))
    return tensor

def predict_sequence_frames(
    model_type: str,
    frames: list,
    sequence_length: int,
    temperature: float | None = None,
    use_tta: bool = True,
    restrict_indices: list[int] | None = None,
) -> tuple[str | None, float, list[dict], int]:
    model, labels = get_sequence_model_bundle(model_type)
    if not model or not labels:
        return None, 0.0, [], 0

    temp = max(float(temperature or 1.0), 1e-6)
    variants = build_segment_tta_variants(frames) if use_tta else [smooth_segment_frames(frames)]
    probs_list: list = []

    with torch.no_grad():
        for variant in variants:
            tensor = frames_to_model_tensor(model_type, variant, sequence_length)
            logits = model(torch.tensor(tensor[None, ...], dtype=torch.float32))[0].numpy()
            if restrict_indices:
                mask = np.full_like(logits, -np.inf)
                for idx in restrict_indices:
                    if 0 <= idx < logits.shape[0]:
                        mask[idx] = 0.0
                logits = logits + mask
            probs_list.append(softmax(logits / temp))

    if not probs_list:
        return None, 0.0, [], 0
    avg_probs = np.mean(np.stack(probs_list, axis=0), axis=0)
    label, conf, top = top_predictions_from_probs(avg_probs, labels)
    return label, conf, top, len(probs_list)


def predict_dual_scenario(
    frames: list,
    sequence_length: int,
    temperature: float | None = None,
    use_tta: bool = True,
) -> dict[str, Any]:
    out: dict[str, Any] = {}

    # WORD
    if "word_v2" in state.sequence_models and state.SCENARIO_WORD_INDICES:
        w_label, w_conf, w_top, _ = predict_sequence_frames(
            "word_v2", frames, sequence_length, temperature, use_tta,
            restrict_indices=state.SCENARIO_WORD_INDICES,
        )
        out["word"] = {
            "label": w_label, "display_label": display_label_for(w_label),
            "confidence": w_conf, "acc_prior": scenario_label_acc(w_label), "top": w_top,
        }
    else:
        out["word"] = {"label": None, "display_label": None, "confidence": 0.0, "acc_prior": 0.0, "top": []}

    # SENTENCE
    if "sentence_v2" in state.sequence_models and state.SCENARIO_SEN_INDICES:
        s_label, s_conf, s_top, _ = predict_sequence_frames(
            "sentence_v2", frames, sequence_length, temperature, use_tta,
            restrict_indices=state.SCENARIO_SEN_INDICES,
        )
        out["sentence"] = {
            "label": s_label, "display_label": display_label_for(s_label),
            "confidence": s_conf, "acc_prior": scenario_label_acc(s_label), "top": s_top,
        }
    else:
        out["sentence"] = {"label": None, "display_label": None, "confidence": 0.0, "acc_prior": 0.0, "top": []}

    # Fusion candidates
    word_cands = (out["word"].get("top") or []) or [{"label": out["word"]["label"], "confidence": out["word"]["confidence"], "acc_prior": out["word"]["acc_prior"]}]
    sen_cands  = (out["sentence"].get("top") or []) or [{"label": out["sentence"]["label"], "confidence": out["sentence"]["confidence"], "acc_prior": out["sentence"]["acc_prior"]}]
    word_cands = [i for i in word_cands if i.get("label")]
    sen_cands  = [i for i in sen_cands  if i.get("label")]

    lookup_candidates: list[dict] = []
    for wi in word_cands[:3]:
        w_lbl = str(wi.get("label") or "")
        w_conf = float(wi.get("confidence") or 0.0)
        w_acc  = scenario_label_acc(w_lbl); wi["acc_prior"] = w_acc
        for si in sen_cands[:3]:
            s_lbl = str(si.get("label") or "")
            s_conf = float(si.get("confidence") or 0.0)
            s_acc  = scenario_label_acc(s_lbl); si["acc_prior"] = s_acc
            for key in (f"{s_lbl}+{w_lbl}", f"{w_lbl}+{s_lbl}"):
                if key in state.SCENARIO_LOOKUP:
                    lookup_candidates.append({
                        "key": key, "text": state.SCENARIO_LOOKUP[key],
                        "score": round(((w_conf * w_acc) * (s_conf * s_acc)) ** 0.5, 6),
                        "source": "word_sentence_pair", "word": wi, "sentence": si,
                    })
                    break

    for source, items in (("single_sentence", sen_cands[:3]), ("single_word", word_cands[:3])):
        for item in items:
            lbl = str(item.get("label") or "")
            if lbl in state.SCENARIO_LOOKUP:
                conf = float(item.get("confidence") or 0.0)
                if source == "single_sentence" and conf < 0.55:
                    continue
                acc = scenario_label_acc(lbl); item["acc_prior"] = acc
                lookup_candidates.append({
                    "key": lbl, "text": state.SCENARIO_LOOKUP[lbl],
                    "score": round(conf * acc, 6), "source": source,
                    "word": item if source == "single_word" else None,
                    "sentence": item if source == "single_sentence" else None,
                })

    lookup_candidates.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)

    # 단일 sentence vs word 우선순위 조정
    best_sen = next((x for x in lookup_candidates if x.get("source") == "single_sentence" and float(x.get("score") or 0.0) >= 0.20), None)
    best_wrd = next((x for x in lookup_candidates if x.get("source") == "single_word"), None)
    if best_sen and best_wrd:
        raw_wlbl = str((best_wrd.get("word") or {}).get("label") or "")
        if raw_wlbl in state.SCENARIO_WORD_IDS and float(best_wrd.get("score") or 0.0) < 0.75 and float(best_sen.get("score") or 0.0) >= float(best_wrd.get("score") or 0.0) * 0.35:
            lookup_candidates = [best_sen, *[x for x in lookup_candidates if x is not best_sen]]

    _PROTECTED_SEN = {"SEN0354", "SEN0355"}

    pair_cands = [x for x in lookup_candidates if x.get("source") == "word_sentence_pair"]
    if pair_cands:
        best_single_score = max((float(x.get("score") or 0.0) for x in lookup_candidates if x.get("source") != "word_sentence_pair"), default=0.0)
        best_pair = pair_cands[0]
        pair_sen_label = str((best_pair.get("sentence") or {}).get("label") or "")
        pair_blocked = pair_sen_label in _PROTECTED_SEN
        if (
            not pair_blocked and
            float((best_pair.get("word") or {}).get("confidence") or 0.0) >= 0.12 and
            float((best_pair.get("sentence") or {}).get("confidence") or 0.0) >= 0.12 and
            float(best_pair.get("score") or 0.0) >= best_single_score * 1.05
        ):
            lookup_candidates = [best_pair, *[x for x in lookup_candidates if x is not best_pair]]

    out["fusion_candidates"] = lookup_candidates[:5]
    if lookup_candidates:
        best = lookup_candidates[0]
        out["scenario_text"] = best["text"]
        out["lookup_hit"]    = True
        out["lookup_key"]    = best["key"]
        out["lookup_source"] = best["source"]
        out["lookup_score"]  = best["score"]
    else:
        out["scenario_text"] = None
        out["lookup_hit"]    = False

    return out

def landmarks_payload_to_frame(landmarks: dict[str, Any]) -> Any:
    if np is None:
        raise RuntimeError("NumPy is not installed")
    layout = [("pose", 33), ("left_hand", 21), ("right_hand", 21)]
    points: list[list[float]] = []
    for key, expected in layout:
        raw = landmarks.get(key) if isinstance(landmarks, dict) else None
        if not isinstance(raw, list):
            raw = []
        for idx in range(expected):
            pt = raw[idx] if idx < len(raw) else None
            if isinstance(pt, list):
                xyz = [float(pt[i] or 0.0) if i < len(pt) else 0.0 for i in range(3)]
            elif isinstance(pt, dict):
                xyz = [float(pt.get(ax, 0.0) or 0.0) for ax in ("x", "y", "z")]
            else:
                xyz = [0.0, 0.0, 0.0]
            points.append(xyz)
    return np.asarray(points, dtype=np.float32)

def landmarks_have_points(points: Any) -> bool:
    if not isinstance(points, list):
        return False
    for pt in points:
        if isinstance(pt, list) and any(abs(float(v or 0.0)) > 1e-9 for v in pt[:2]):
            return True
        if isinstance(pt, dict) and (
            abs(float(pt.get("x", 0.0) or 0.0)) > 1e-9 or
            abs(float(pt.get("y", 0.0) or 0.0)) > 1e-9
        ):
            return True
    return False
