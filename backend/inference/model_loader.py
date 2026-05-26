"""inference/model_loader.py

체크포인트 로드, MediaPipe 초기화, label map 파싱을 담당합니다.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import inference.model_state as state
from inference.model_state import (
    SCENARIO_SEN_IDS,
    SCENARIO_WORD_IDS,
    build_sequence_model,
    mp,
    torch,
)

import sys

# KSL_project/backend/inference/model_loader.py 기준
BACKEND_DIR  = Path(__file__).resolve().parents[1]  # KSL_project/backend/
PROJECT_ROOT = BACKEND_DIR.parent                   # KSL_project/
ROOT_DIR     = PROJECT_ROOT  # legacy fallback 함수에서 사용

def _find_handover_dir(root: Path) -> Path:
    candidates = [
        root / "web_handover",
        root / "web_handover" / "web_handover",
    ]
    for c in candidates:
        if (c / "models").exists():
            return c
    return root / "web_handover"

HANDOVER_DIR = _find_handover_dir(PROJECT_ROOT)

# cnngru_attn.py import를 위해 code/ 경로 추가
_HANDOVER_CODE_DIR = HANDOVER_DIR / "code"
if _HANDOVER_CODE_DIR.exists() and str(_HANDOVER_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_HANDOVER_CODE_DIR))

MODEL_FILES = {
    "word_v2":     HANDOVER_DIR / "models" / "word_stage2.pt",
    "sentence_v2": HANDOVER_DIR / "models" / "sentence_stage2_reproc.pt",
}

# label map / lookup / temperature — web_handover 폴더 기준
LABEL_MAP_WORD_PATH     = HANDOVER_DIR / "labels"      / "label_map_word.json"
LABEL_MAP_SENTENCE_PATH = HANDOVER_DIR / "labels"      / "label_map_sentence.json"
LOOKUP_TABLE_PATH       = HANDOVER_DIR / "lookup"      / "lookup_table.json"
TEMPERATURE_WORD_PATH   = HANDOVER_DIR / "calibration" / "temperature_word.txt"
TEMPERATURE_SENTENCE_PATH = HANDOVER_DIR / "calibration" / "temperature_sentence.txt"


# ── label map 로드 ────────────────────────────────────────────────

def _normalize_label_map(raw: Any) -> dict[int, str]:
    """label_map JSON → {class_id(int): label(str)} 정규화.

    지원 schema:
      1) WORD:     {"WORD0001": 0, "WORD0002": 1, ...}  (label → id, 역방향)
      2) SENTENCE: {"class_id_to_sentence_id": {"0": "SEN0001", ...}, ...} (래퍼)
      3) list:     ["WORD0001", "WORD0002", ...]
      4) {str(id): label}
    """
    if isinstance(raw, dict) and "class_id_to_sentence_id" in raw:
        # SENTENCE 래퍼 schema
        return {int(k): v for k, v in raw["class_id_to_sentence_id"].items()}
    if isinstance(raw, dict):
        first_val = next(iter(raw.values()), None)
        if isinstance(first_val, int):
            # {label: id} 역방향 → {id: label}
            return {int(v): k for k, v in raw.items()}
        return {int(k): v for k, v in raw.items()}
    if isinstance(raw, list):
        return {i: lbl for i, lbl in enumerate(raw)}
    return {}


def load_word_label_map() -> list[str]:
    """WORD label map → class_id 순서대로 정렬된 label 리스트."""
    if LABEL_MAP_WORD_PATH.exists():
        with LABEL_MAP_WORD_PATH.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        id_to_label = _normalize_label_map(raw)
        return [id_to_label[i] for i in sorted(id_to_label)]
    # fallback: 기존 model_results 경로
    return _load_label_map_legacy(is_sentence=False)


def load_sentence_label_map() -> list[str]:
    """SENTENCE label map → class_id 순서대로 정렬된 label 리스트."""
    if LABEL_MAP_SENTENCE_PATH.exists():
        with LABEL_MAP_SENTENCE_PATH.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        id_to_label = _normalize_label_map(raw)
        return [id_to_label[i] for i in sorted(id_to_label)]
    return _load_label_map_legacy(is_sentence=True)


def _load_label_map_legacy(is_sentence: bool) -> list[str]:
    """기존 model_results 경로에서 label map 로드 (fallback)."""
    candidates = (
        [ROOT_DIR / "model_results" / "label_map_sen_id_2000.json"]
        if is_sentence else
        [ROOT_DIR / "model_results" / "idx_to_label_3000.json",
         ROOT_DIR / "model_results" / "label_map_word_id_3000.json"]
    )
    for path in candidates:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            id_to_label = _normalize_label_map(data)
            return [id_to_label[i] for i in sorted(id_to_label)]
    return []


def load_label_display_map() -> dict[str, str]:
    """display용 라벨 맵 (WORD ID / SEN ID → 한국어 표시명)."""
    display: dict[str, str] = {}

    def _try(path: Path, id_keys: list[str], label_keys: list[str], sen_offset: bool = False) -> None:
        if not path.exists():
            return
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            for key, value in data.items():
                if isinstance(value, dict):
                    entry_id = next((str(value.get(k) or "").strip() for k in id_keys if value.get(k)), key)
                    label    = next((str(value.get(k) or "").strip() for k in label_keys if value.get(k)), "")
                else:
                    entry_id, label = str(key).strip(), str(value).strip()
                if sen_offset and entry_id.isdigit():
                    entry_id = f"SEN{int(entry_id) + 1:04d}"
                if entry_id and label:
                    display[entry_id] = label
        except Exception as exc:
            print(f"[model_loader] display map load failed ({path}): {exc}")

    _try(ROOT_DIR / "model_results" / "idx_to_label_3000.json",        ["word_id"], ["label"])
    _try(ROOT_DIR / "model_results" / "idx_to_label_sen_2000.json",    ["sen_id", "label_id"], ["label", "text"], sen_offset=True)
    _try(ROOT_DIR / "model_results" / "idx_to_label_sen_scenario12.json", ["sen_id", "label"], ["display", "text"])
    return display


# ── lookup table 로드 ─────────────────────────────────────────────

def load_lookup_table() -> dict[str, str]:
    merged: dict[str, str] = {}

    legacy = Path(__file__).resolve().parent / "scenario_lookup.json"
    if legacy.exists():
        try:
            with legacy.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            merged.update({k: v for k, v in raw.items() if not k.startswith("_")})
        except Exception as exc:
            print(f"[model_loader] legacy lookup load failed: {exc}")

    if LOOKUP_TABLE_PATH.exists():
        try:
            with LOOKUP_TABLE_PATH.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            tbl = raw.get("lookup_table", raw)
            merged.update({k: v["label"] if isinstance(v, dict) else str(v) for k, v in tbl.items()})
        except Exception as exc:
            print(f"[model_loader] handover lookup load failed: {exc}")

    print(f"[lookup] total={len(merged)} sample={list(merged.keys())[:10]}")
    return merged

# ── temperature 로드 ──────────────────────────────────────────────

def load_temperatures() -> tuple[float, float]:
    """(word_temperature, sentence_temperature) 반환."""
    def _read(path: Path, default: float) -> float:
        try:
            return float(path.read_text(encoding="utf-8").strip())
        except Exception:
            return default
    return (
        _read(TEMPERATURE_WORD_PATH,     0.8627),
        _read(TEMPERATURE_SENTENCE_PATH, 0.9156),
    )


# ── 체크포인트 로드 ───────────────────────────────────────────────

def load_word_checkpoint(path: Path) -> tuple[Any, list[str]]:
    """WORD v2 체크포인트 로드 — CNNGRUAttn, xyz 3-channel, 225-dim."""
    if torch is None:
        raise RuntimeError("torch is not installed")

    try:
        from src.models.cnngru_attn import CNNGRUAttn
    except ImportError:
        from cnngru_attn import CNNGRUAttn  # 핸드오버 code/ 폴더 직접 사용

    bundle = torch.load(path, map_location="cpu")
    mc = bundle["model_config"]
    model = CNNGRUAttn(**mc)
    model.load_state_dict(bundle["model_state"])
    model.eval()
    model.input_size = int(mc.get("input_dim", 225))  # 라우터에서 참조
    model.is_sentence_v2 = False

    labels = load_word_label_map()
    if not labels and isinstance(bundle.get("labels"), (list, tuple)):
        labels = [str(l) for l in bundle["labels"]]

    print(f"[model_loader] WORD v2 loaded: {path.name}, classes={len(labels)}, input_dim={model.input_size}")
    return model, labels


def load_sentence_checkpoint(path: Path) -> tuple[Any, list[str]]:
    """SENTENCE v2 체크포인트 로드 — CNNGRUAttnV2, xyzc 4-channel, 300-dim, valid_lengths 필수."""
    if torch is None:
        raise RuntimeError("torch is not installed")

    try:
        from src.models.cnngru_attn import CNNGRUAttnV2
    except ImportError:
        from cnngru_attn import CNNGRUAttnV2

    bundle = torch.load(path, map_location="cpu")
    mc = bundle["model_config"]

    # ⚠ "variant" 키 제거 필수 (CNNGRUAttnV2.__init__가 받지 않음)
    mc_clean = {k: v for k, v in mc.items() if k != "variant"}
    model = CNNGRUAttnV2(**mc_clean)
    model.load_state_dict(bundle["model_state"])
    model.eval()
    model.input_size = int(mc_clean.get("input_dim", 300))
    model.is_sentence_v2 = True   # predictor에서 분기용 플래그

    labels = load_sentence_label_map()
    if not labels and isinstance(bundle.get("labels"), (list, tuple)):
        labels = [str(l) for l in bundle["labels"]]

    print(f"[model_loader] SENTENCE v2 loaded: {path.name}, classes={len(labels)}, input_dim={model.input_size}")
    return model, labels


def load_sequence_checkpoint(path: Path) -> tuple[Any, list[str]]:
    """모델 키에 따라 WORD / SENTENCE 체크포인트 로드를 자동 분기."""
    name = path.stem.lower()
    if "sentence" in name or "_sen" in name or "reproc" in name:
        return load_sentence_checkpoint(path)
    return load_word_checkpoint(path)


# ── 전체 모델 로드 (앱 시작 시 한 번) ────────────────────────────

def load_models() -> None:
    # MediaPipe 초기화
    if mp is None:
        print("[model_loader] MediaPipe not installed; /api/predict will return 503")
        return

    try:
        state.mp_holistic = mp.solutions.holistic.Holistic
        state.mp_holistic_instance = state.mp_holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        state.mp_drawing = mp.solutions.drawing_utils
        print("[model_loader] MediaPipe loaded")
    except Exception as exc:
        print(f"[model_loader] MediaPipe load failed: {exc}")

    if torch is None:
        print("[model_loader] torch not installed; landmarks only")
        return

    # 모델 파일별 로드
    for model_key, path in MODEL_FILES.items():
        if not path.exists():
            print(f"[model_loader] {model_key}: {path} not found — skipping")
            continue
        try:
            model, labels = load_sequence_checkpoint(path)
            state.sequence_models[model_key] = model
            state.sequence_labels[model_key] = labels
        except Exception as exc:
            print(f"[model_loader] {model_key} load failed: {exc}")

    # 프론트 호환: model_type="sequence"/"cnn_gru" → word_v2 로 라우팅
    if "word_v2" in state.sequence_models:
        state.sequence_models["cnn_gru"] = state.sequence_models["word_v2"]
        state.sequence_labels["cnn_gru"] = state.sequence_labels["word_v2"]

    # temperature 로드
    state.TEMPERATURE_WORD, state.TEMPERATURE_SENTENCE = load_temperatures()
    print(f"[model_loader] temperature: WORD={state.TEMPERATURE_WORD}, SENTENCE={state.TEMPERATURE_SENTENCE}")

    # 시나리오 class indices 계산
    word_labels = state.sequence_labels.get("word_v2") or state.sequence_labels.get("cnn_gru") or []
    if word_labels:
        word_idx = {label: i for i, label in enumerate(word_labels)}
        state.SCENARIO_WORD_INDICES = [word_idx[w] for w in SCENARIO_WORD_IDS if w in word_idx]
        print(f"[model_loader] SCENARIO_WORD_INDICES: {len(state.SCENARIO_WORD_INDICES)}/{len(SCENARIO_WORD_IDS)}")
    else:
        print("[model_loader] Warning: WORD label_map unavailable")

    sen_labels = state.sequence_labels.get("sentence_v2") or []
    if sen_labels:
        sen_idx = {label: i for i, label in enumerate(sen_labels)}
        state.SCENARIO_SEN_INDICES = [sen_idx[s] for s in SCENARIO_SEN_IDS if s in sen_idx]
        print(f"[model_loader] SCENARIO_SEN_INDICES: {len(state.SCENARIO_SEN_INDICES)}/{len(SCENARIO_SEN_IDS)}")
    else:
        print("[model_loader] Warning: SENTENCE label_map unavailable")

    # lookup table 로드 (핸드오버 우선)
    state.SCENARIO_LOOKUP = load_lookup_table()
    print(f"[model_loader] SCENARIO_LOOKUP: {len(state.SCENARIO_LOOKUP)} entries")

    # display label map 로드
    state.LABEL_DISPLAY_MAP = load_label_display_map()
    print(f"[model_loader] LABEL_DISPLAY_MAP: {len(state.LABEL_DISPLAY_MAP)} entries")


def ensure_models_loaded() -> None:
    if state.model_load_attempted:
        return
    with state.model_load_lock:
        if state.model_load_attempted:
            return
        load_models()
        state.model_load_attempted = True
