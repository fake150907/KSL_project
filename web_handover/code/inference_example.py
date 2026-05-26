"""WEB 담당자용 추론 예제 — WORD + SENTENCE 모델 사용 풀 흐름.

본 예제는 이미 추출된 MediaPipe Holistic keypoint NPZ를 입력으로 받아
모델 추론 → top5 추출 → 라벨 매핑까지 보여준다.

실제 운영 시:
  - 영상(mp4) → MediaPipe Holistic → keypoint frames → 본 예제 입력

요구 패키지:
  pip install torch numpy

본 파일과 같이 배포되어야 하는 자료:
  - models/word_stage2.pt, models/sentence_stage2_reproc.pt
  - code/cnngru_attn.py (본 디렉토리)
  - code/npz_dataset.py (본 디렉토리, 참고용)
  - labels/label_map_word.json, labels/label_map_sentence.json
  - lookup/lookup_table.json (선택, 시나리오 운영용)
  - lookup/gloss_vocab_sentence.json (선택, SEN → gloss 매핑)
  - calibration/temperature_word.txt (값 0.8627)
  - calibration/temperature_sentence.txt (값 0.9156)

Usage:
    python inference_example.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

# 본 폴더 구조 가정:  web_handover/
#                       ├── code/inference_example.py  (이 파일)
#                       ├── code/cnngru_attn.py
#                       ├── models/*.pt
#                       ├── labels/*.json
#                       ├── lookup/*.json
#                       └── calibration/*.txt
HANDOVER_ROOT = Path(__file__).resolve().parents[1]

from cnngru_attn import CNNGRUAttn, CNNGRUAttnV2  # noqa: E402


# ---------------------------------------------------------------
# 1. WORD 모델 inference
# ---------------------------------------------------------------

def load_word_model(device):
    """WORD Stage 2 모델 로드. v1과 동일 schema(xyz 3-channel)."""
    ckpt_path = HANDOVER_ROOT / "models" / "word_stage2.pt"
    state = torch.load(ckpt_path, map_location="cpu")
    model = CNNGRUAttn(**state["model_config"])
    model.load_state_dict(state["model_state"])
    model.to(device).eval()

    raw_label = json.loads((HANDOVER_ROOT / "labels" / "label_map_word.json").read_text(encoding="utf-8"))
    label_map = _normalize_label_map(raw_label)
    temperature = float((HANDOVER_ROOT / "calibration" / "temperature_word.txt").read_text().strip())

    print(f"[WORD] loaded {ckpt_path.name}, num_classes={len(label_map)}, T={temperature:.4f}")
    return model, label_map, temperature


@torch.no_grad()
def infer_word(model, keypoint_xyz, label_map, temperature=1.0, top_k=5):
    """WORD 단일 sample 추론.

    Args:
        model: CNNGRUAttn (v1)
        keypoint_xyz: numpy (T, 225) — T frame, 75 keypoint × xyz
        label_map: {class_id (int) → "WORDxxxx"} 또는 list
        temperature: confidence 보정 (1.0 = 미적용)
        top_k: 반환할 후보 수

    Returns:
        list of (label, prob) 위에서 top_k
    """
    device = next(model.parameters()).device
    x = torch.from_numpy(keypoint_xyz).float().unsqueeze(0).to(device)  # (1, T, 225)
    logits = model(x)
    probs = torch.softmax(logits / temperature, dim=1)
    topk = probs.topk(top_k, dim=1)
    labels = [_lookup_label(label_map, int(idx)) for idx in topk.indices[0].tolist()]
    probs_list = topk.values[0].tolist()
    return list(zip(labels, probs_list))


# ---------------------------------------------------------------
# 2. SENTENCE 모델 inference (v2 schema)
# ---------------------------------------------------------------

T_MAX_SENTENCE = 128  # SENTENCE v2 고정


def load_sentence_model(device):
    """SENTENCE Stage 2 v2 모델 로드. xyzc 4-channel + valid_lengths 필수."""
    ckpt_path = HANDOVER_ROOT / "models" / "sentence_stage2_reproc.pt"
    state = torch.load(ckpt_path, map_location="cpu")
    # v2 ckpt에는 "variant" 키가 있을 수 있음 — CNNGRUAttnV2.__init__가 받지 않으므로 제거
    mc = state["model_config"]
    mc_clean = {k: v for k, v in mc.items() if k != "variant"}
    model = CNNGRUAttnV2(**mc_clean)
    model.load_state_dict(state["model_state"])
    model.to(device).eval()

    raw_label = json.loads((HANDOVER_ROOT / "labels" / "label_map_sentence.json").read_text(encoding="utf-8"))
    label_map = _normalize_label_map(raw_label)
    temperature = float((HANDOVER_ROOT / "calibration" / "temperature_sentence.txt").read_text().strip())

    print(f"[SENTENCE] loaded {ckpt_path.name}, num_classes={len(label_map)}, T={temperature:.4f}")
    return model, label_map, temperature


def pack_sentence_input(keypoint_xyzc):
    """SENTENCE 입력 packing — T_max=128 padding + valid_length 계산.

    Args:
        keypoint_xyzc: numpy (T, 300) — T frame, 75 keypoint × xyzc 4-channel
            (xyz 좌표 + c=visibility/confidence)

    Returns:
        x_padded: numpy (128, 300) — 모자라면 0 padding, 넘으면 자름
        valid_length: int — 실제 유효 frame 수 (1 ~ 128)
    """
    T, dim = keypoint_xyzc.shape
    assert dim == 300, f"SENTENCE 입력 dim은 300이어야 함 (xyzc 4-channel × 75 keypoint), got {dim}"
    valid_length = min(T, T_MAX_SENTENCE)
    x_padded = np.zeros((T_MAX_SENTENCE, dim), dtype=np.float32)
    x_padded[:valid_length] = keypoint_xyzc[:valid_length]
    return x_padded, valid_length


@torch.no_grad()
def infer_sentence(model, keypoint_xyzc, label_map, temperature=1.0, top_k=5):
    """SENTENCE 단일 sample 추론. v2 schema 필수.

    Args:
        model: CNNGRUAttnV2 (v2)
        keypoint_xyzc: numpy (T, 300) — T frame, xyzc 4-channel
        label_map, temperature, top_k: WORD와 동일

    Returns:
        list of (label, prob)
    """
    device = next(model.parameters()).device
    x_padded, valid_length = pack_sentence_input(keypoint_xyzc)
    x = torch.from_numpy(x_padded).float().unsqueeze(0).to(device)        # (1, 128, 300)
    vl = torch.tensor([valid_length], dtype=torch.long).to(device)        # (1,)
    logits = model(x, vl)  # ⚠ v2는 forward 인자 2개
    probs = torch.softmax(logits / temperature, dim=1)
    topk = probs.topk(top_k, dim=1)
    labels = [_lookup_label(label_map, int(idx)) for idx in topk.indices[0].tolist()]
    probs_list = topk.values[0].tolist()
    return list(zip(labels, probs_list))


# ---------------------------------------------------------------
# 3. Lookup Table 적용 (운영 시나리오용, 선택)
# ---------------------------------------------------------------

def apply_lookup(sen_top5, lookup_table):
    """top5 안에 lookup key가 있으면 자연어 표현으로 매핑.

    Args:
        sen_top5: list of (sen_id, prob) — SENTENCE top5
        lookup_table: dict from lookup/lookup_table.json["lookup_table"]
            {sen_id: {"label": "자연어 표현", "weight": float}}

    Returns:
        first matched (natural_label, weight) or None
    """
    for sen_id, _ in sen_top5:
        if sen_id in lookup_table:
            entry = lookup_table[sen_id]
            return entry["label"], entry["weight"]
    return None


# ---------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------

def _normalize_label_map(raw):
    """label_map JSON을 {class_id (int) → label (str)} 형태로 정규화.

    지원 schema:
      1) WORD label_map_word.json: {"WORD0001": 0, "WORD0002": 1, ...}  (label → id)
      2) SENTENCE label_map_sentence.json: 래퍼 dict
         {"version": ..., "num_classes": ..., "class_id_to_sentence_id": {...}, "sentence_id_to_class_id": {...}}
      3) (확장) list 형태: ["WORD0001", "WORD0002", ...] 또는 dict {0: "WORD0001"}
    """
    # SENTENCE 래퍼 schema
    if isinstance(raw, dict) and "class_id_to_sentence_id" in raw:
        return {int(k): v for k, v in raw["class_id_to_sentence_id"].items()}
    # WORD schema: label → id (역방향)
    if isinstance(raw, dict):
        first_val = next(iter(raw.values()))
        if isinstance(first_val, int):
            # {label: id} 형태 — 역방향 매핑
            return {int(v): k for k, v in raw.items()}
        # {id (str/int): label} 형태
        return {int(k): v for k, v in raw.items()}
    # list 형태
    if isinstance(raw, list):
        return {i: lbl for i, lbl in enumerate(raw)}
    return {}


def _lookup_label(label_map, class_id):
    """정규화된 label_map (class_id → label)에서 조회."""
    if isinstance(label_map, dict) and 0 in label_map or isinstance(label_map, dict):
        return label_map.get(int(class_id), "?")
    return "?"


# ---------------------------------------------------------------
# 통합 데모
# ---------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}")
    print(f"[setup] handover root: {HANDOVER_ROOT}")

    # 모델 로드
    word_model, word_labels, T_word = load_word_model(device)
    sen_model, sen_labels, T_sen = load_sentence_model(device)
    lookup = json.loads((HANDOVER_ROOT / "lookup" / "lookup_table.json").read_text(encoding="utf-8"))
    lookup_table = lookup["lookup_table"]

    # 더미 입력 예제 (실제론 MediaPipe로 추출)
    print("\n=== 더미 입력으로 forward 동작 확인 ===")
    dummy_word_keypoint = np.random.randn(30, 225).astype(np.float32)  # 30 frame, xyz
    word_top5 = infer_word(word_model, dummy_word_keypoint, word_labels, T_word)
    print(f"WORD top5: {word_top5[:3]} ...")

    dummy_sen_keypoint = np.random.randn(64, 300).astype(np.float32)   # 64 frame, xyzc
    sen_top5 = infer_sentence(sen_model, dummy_sen_keypoint, sen_labels, T_sen)
    print(f"SENTENCE top5: {sen_top5[:3]} ...")

    # Lookup 적용 예제
    lookup_result = apply_lookup(sen_top5, lookup_table)
    if lookup_result:
        print(f"Lookup matched: {lookup_result}")
    else:
        print("Lookup: top5에 운영 시나리오 lookup key 없음")

    print("\n[OK] 추론 흐름 동작 확인 완료. 실제 사용 시 MediaPipe로 추출한 keypoint를 dummy 자리에 넣으세요.")


if __name__ == "__main__":
    main()
