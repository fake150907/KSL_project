# WEB 데모 모델 교체 가이드 (v2)

**대상**: WEB 담당자
**작성일**: 2026-05-23
**버전**: WORD v2 + SENTENCE v2

---

## 0. 가장 먼저 — 한 줄 요약

- **WORD 모델**: 입력 schema 동일. **파일만 갈아끼우면 됨**.
- **SENTENCE 모델**: 입력 schema 변경(xyzc 4-channel + T_max=128 + valid_lengths). **전처리·forward 코드 수정 필요**.

---

## 1. 폴더 구조

```
web_handover/
├── README.md                      ← 이 문서
├── MODEL_SPEC.md                  ← 입출력·shape 사양서
├── CHANGELOG.md                   ← v1 → v2 변경점·체크리스트
├── models/
│   ├── word_stage2.pt             (15.6 MB)
│   └── sentence_stage2_reproc.pt  (13.7 MB)
├── code/
│   ├── cnngru_attn.py             ← 모델 클래스 (CNNGRUAttn + CNNGRUAttnV2)
│   ├── npz_dataset.py             ← 입력 schema 검증 (참고)
│   └── inference_example.py       ← 추론 예제 (동작 확인됨)
├── labels/
│   ├── label_map_word.json        ← WORD 3000 class label
│   └── label_map_sentence.json    ← SENTENCE 2000 class label (양방향 매핑)
├── lookup/
│   ├── lookup_table.json          ← 시나리오 32 entry + critical 7 entry
│   └── gloss_vocab_sentence.json  ← SEN → 형태소(gloss) 매핑
└── calibration/
    ├── temperature_word.txt       ← 0.8627
    └── temperature_sentence.txt   ← 0.9156
```

---

## 2. 빠른 시작 (가장 빠른 경로)

```bash
# 1. 의존성 설치
pip install torch numpy mediapipe

# 2. 추론 예제 실행 (더미 입력으로 동작 확인)
cd web_handover/code
python inference_example.py
```

예상 출력:
```
[WORD] loaded word_stage2.pt, num_classes=3000, T=0.8627
[SENTENCE] loaded sentence_stage2_reproc.pt, num_classes=2000, T=0.9156
WORD top5: [('WORD1393', 0.0056), ('WORD2493', 0.0052), ...]
SENTENCE top5: [('SEN0064', 0.0330), ('SEN0217', 0.0157), ...]
[OK] 추론 흐름 동작 확인 완료.
```

→ 동작 확인되면 `inference_example.py`의 dummy keypoint 자리에 **실제 MediaPipe 추출 결과**를 넣으면 그대로 운영 가능.

---

## 3. 두 모델 핵심 차이 (반드시 숙지)

| 항목 | WORD v2 | SENTENCE v2 |
|---|---|---|
| 입력 채널 | xyz **3-channel** | **xyzc 4-channel** (visibility 추가) |
| Input dim/frame | 225 | **300** |
| Sequence length | 가변 | **T_max=128 fixed padding** |
| valid_lengths | 불필요 | **필수** |
| Model class | `CNNGRUAttn` | **`CNNGRUAttnV2`** |
| forward 호출 | `model(x)` | **`model(x, valid_lengths)`** |
| model_config 처리 | 그대로 init | **`"variant"` 키 제거 필수** |

상세는 [MODEL_SPEC.md](MODEL_SPEC.md), [CHANGELOG.md](CHANGELOG.md) 참고.

---

## 4. 교체 시 체크리스트

### WORD 모델 교체 (간단)
- [ ] 기존 `word_*.pt` → `models/word_stage2.pt`로 변경
- [ ] 추론 코드 그대로 작동 확인

### SENTENCE 모델 교체 (작업량 큼)
- [ ] `models/sentence_stage2_reproc.pt` 배치
- [ ] `code/cnngru_attn.py`, `code/npz_dataset.py` import 경로 설정
- [ ] MediaPipe Holistic 출력에서 **visibility(c) 추출** 추가
- [ ] keypoint frame → **xyzc 4-channel 300-dim** packing
- [ ] **T_max=128 padding + valid_lengths 계산** 추가
- [ ] forward 호출 `model(x, valid_lengths)` 형태로 변경
- [ ] model_config에서 `"variant"` 키 제거 후 init (`CNNGRUAttnV2`)
- [ ] (선택) Temperature `0.9156` 적용
- [ ] (선택) Lookup Table·gloss_vocab 통합

### 운영 검증
- [ ] 영상 1개로 영상 → MediaPipe → 모델 → top5 → 라벨 매핑 전체 흐름 동작 확인
- [ ] 시나리오 1 (복지카드 분실 11턴) 영상으로 운영 흐름 시연
- [ ] (선택) 운영 confidence threshold 재조정

---

## 5. label_map JSON 두 가지 schema

본 패키지 두 label map은 schema가 다름 — `code/inference_example.py`의 `_normalize_label_map()` 함수가 양쪽 자동 처리.

| 파일 | Schema |
|---|---|
| `labels/label_map_word.json` | `{"WORD0001": 0, "WORD0002": 1, ...}` (label → class_id) |
| `labels/label_map_sentence.json` | `{"version": "v2", "class_id_to_sentence_id": {"0": "SEN0001", ...}, "sentence_id_to_class_id": {"SEN0001": 0, ...}}` |

자체 inference 코드 쓰면 정규화 함수 참고 (`code/inference_example.py:_normalize_label_map`).

---

## 6. 운영 검증 결과 (참고)

| 모델 | AIHub val_top1 | REAL18 unseen | TEAM 청인 holdout | ECE 개선 |
|---|---:|---:|---:|---|
| WORD v2 | 96.55% | 97.53% | 60.00% (top5 75%) | 0.11 → 0.03 (3.4배) |
| SENTENCE v2 | 92.26% | 88.54% | 61.82% (top5 87.88%) | 0.07 → 0.02 (3배) |

**시나리오 E2E (REAL18 11턴 합성)**: **E2E score 0.9045 (운영 기준 0.85+ 통과)**
- Critical Word Recall 1.0000
- Lookup hit rate 1.0000

상세는 학습 측 별도 보고서 (`reports_final_sentence/`, `reports_final_word/`, `reports_final_combined/`) 참고.

---

## 7. 문제 발생 시 참고

| 문제 | 참고 위치 |
|---|---|
| 모델 사양 의문 | [MODEL_SPEC.md](MODEL_SPEC.md) |
| v1 → v2 변경점 | [CHANGELOG.md](CHANGELOG.md) |
| 예제 코드 | [code/inference_example.py](code/inference_example.py) |
| 학습·평가 상세 | (별도) `reports_final_sentence/`, `reports_final_word/`, `reports_final_combined/` |
| 입력 shape 안 맞을 때 | `code/npz_dataset.py`의 v2 schema assert 메시지 |

---

## 8. 의존성

| 패키지 | 버전 |
|---|---|
| Python | 3.10+ |
| PyTorch | 2.0+ (학습 환경: 2.11.0+cu128) |
| numpy | 1.20+ |
| mediapipe | 0.10+ |

GPU 권장 (CPU 가능, 추론 속도 1/10 수준).

---

**문의·이슈**: 학습 측 담당자에게 본 패키지의 ZIP 출처와 함께 문의.
