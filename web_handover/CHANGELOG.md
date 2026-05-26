# v1 → v2 변경사항 (WEB 담당자 체크리스트)

**날짜**: 2026-05-23
**대상**: WEB 데모 모델 교체 작업자

---

## 1. 한눈에 보는 변경 요약

| 모델 | v1 → v2 변경 강도 | WEB 코드 수정 필요 여부 |
|---|---|---|
| **WORD** | 가벼움 (정확도 96.28% → 96.55%, +0.27%p) | ❌ **모델 파일만 교체** |
| **SENTENCE** | **무거움 (입력 schema·forward 호출·전처리 모두 변경)** | ✅ **전처리·forward 코드 수정 필요** |

---

## 2. WORD v2 변경점

| 항목 | v1 | v2 | WEB 영향 |
|---|---|---|---|
| 입력 schema | xyz 3-channel, 225-dim | 동일 | 없음 |
| 모델 클래스 | `CNNGRUAttn` | 동일 | 없음 |
| forward 호출 | `model(x)` | 동일 | 없음 |
| 가중치 | val 96.28% | val 96.55% (+TEAM fine-tune) | 모델 파일만 교체 |
| 클래스 수 | 3000 | 3000 | 없음 |
| 운영 핵심 단어 인식 | (이전 측정 없음) | 복지카드/면제 100% | 없음 |

**조치**: `word_stage1.pt` (또는 이전 모델) → `models/word_stage2.pt`로 교체. 코드 수정 0.

---

## 3. SENTENCE v2 변경점 ⚠ (작업량 큰 영역)

### 3.1 입력 schema 변경
| 항목 | v1 | v2 |
|---|---|---|
| 채널 | xyz 3-channel | **xyzc 4-channel** (visibility 추가) |
| Input dim/frame | 225 | **300** |
| Sequence | 가변 | **T_max=128 fixed padding** |
| `valid_lengths` | 불필요 | **필수 인자** |

### 3.2 모델 클래스 변경
- **이전**: `CNNGRUAttn` (`from models.cnngru_attn import CNNGRUAttn`)
- **현재**: **`CNNGRUAttnV2`** (`from models.cnngru_attn import CNNGRUAttnV2`)

### 3.3 forward 호출 변경
```python
# v1
logits = model(x)  # x: (B, T, 225)

# v2
logits = model(x, valid_lengths)
# x: (B, 128, 300), valid_lengths: (B,) IntTensor
```

### 3.4 model_config 변경
v2 ckpt의 `state["model_config"]`에는 `"variant": "v2"` 키가 있음. CNNGRUAttnV2 생성 시 이 키 제거 필수:
```python
mc = state["model_config"]
mc_clean = {k: v for k, v in mc.items() if k != "variant"}
model = CNNGRUAttnV2(**mc_clean)
```

### 3.5 클래스 수
v1, v2 모두 2000 — 동일.

### 3.6 학습된 시나리오·운영 보강 추가
- `lookup/lookup_table.json` (32 entry) — 시나리오 turn 단위 정답 매핑
- `lookup/gloss_vocab_sentence.json` — SEN → 형태소(gloss) 매핑
- 모델 confidence 보정용 Temperature: T=0.9156

---

## 4. 운영 흐름 변경 (시나리오 score function)

### v1 (이전)
- SENTENCE 단일 모델 inference → top1 노출

### v2 (현재 권장)
1. SENTENCE 모델 inference → top5 추출
2. WORD 합성 필요 시(turn 2: 복지카드, turn 9: 면제) WORD 모델 추가 inference
3. **Lookup Table 32 entry**로 top5 안의 SEN ID 자연어 변환
4. **Critical Word Recall** 기준 (핵심어 SEN ID가 top5 안에 있는지) 운영 체크
5. (선택) Temperature scaling으로 confidence 보정

상세 score function 흐름은 `reports_final_sentence/SENTENCE_v2_최종_평가_보고서.md` §7 또는 `reports_final_combined/통합_평가_보고서.md` §3 참고.

---

## 5. WEB 교체 체크리스트

WORD만 교체:
- [ ] `word_stage1.pt` → `models/word_stage2.pt` 갈아끼움
- [ ] 추론 코드 그대로 작동 확인

SENTENCE 교체:
- [ ] `models/sentence_stage2_reproc.pt` 배치
- [ ] `code/cnngru_attn.py`, `code/npz_dataset.py` 코드 import 경로 맞추기
- [ ] MediaPipe Holistic 출력에서 **visibility(c) 추출** 추가
- [ ] keypoint frame → **xyzc 4-channel 300-dim**으로 packing
- [ ] **T_max=128 padding + valid_lengths 계산** 추가
- [ ] forward 호출 시 `model(x, valid_lengths)` 형태로 변경
- [ ] model_config에서 `"variant"` 키 제거 후 init
- [ ] (선택) Temperature `0.9156` 적용
- [ ] (선택) Lookup Table·gloss_vocab 통합

운영 검증:
- [ ] 영상 1개로 영상 → MediaPipe → 모델 → top5 → 라벨 매핑 전체 흐름 동작 확인
- [ ] 시나리오 1 (복지카드 분실 11턴) 영상 11개로 운영 흐름 시연
- [ ] 운영 confidence threshold (있다면) ECE 참고해 재조정

---

## 6. 문제 발생 시 참고

- 모델 사양 의문 → `MODEL_SPEC.md` 참고
- 예제 코드 → `code/inference_example.py` 참고
- 학습·평가 상세 → 별도 패키지 `reports_final_sentence/`, `reports_final_word/`, `reports_final_combined/`
- 입력 keypoint shape 안 맞을 때 → `code/npz_dataset.py`의 `NPZKeypointDatasetV2.__init__` 안 v2 schema assert 메시지 참고
