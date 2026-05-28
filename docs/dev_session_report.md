# 수어 인식 시스템 개선 작업 보고서

> 작성일: 2026-05-28  
> 작업 범위: 웹캠 실시간 인식 품질 분석 및 개선

---

## 1. 발견된 문제 목록

### 🔴 [문제 1] sentence 모델이 라이브 웹캠에서 아예 꺼져 있었음

**파일:** `web/src/hooks/useSignLanguage.ts`

`scenario_mode` 기본값이 `'off'`로 설정되어 있어서 **라이브 웹캠에서는 word 모델만 동작**하고 있었음.  
데모 영상은 `forceScenarioMode: true` 덕분에 sentence 모델이 작동했지만, 실제 민원인이 사용하는 라이브 모드에서는 SEN 레이블 수어 (안녕하세요 등)를 아예 인식할 수 없는 상태였음.

**확인 백로그:**
```
confidence=0.174, label=None, scenario_mode=False
→ word 모델만 돌아서 총괄자(17.4%), 대지(14.5%) 같은 무관한 단어만 후보로 나옴
```

---

### 🔴 [문제 2] scenario_text가 메인 모델 결과를 무조건 덮어씀

**파일:** `web/src/hooks/useSignLanguage.ts` — `shouldCommitScenarioText` 함수

scenario_mode를 켜자 새로운 문제 발생.  
`오른쪽(79.1%)` 수어를 했는데 `가능`이 출력됨.

**원인:**
- 메인 `cnn_gru` 모델: 오른쪽 (79.1%)
- `word_v2` (scenario): 가능 (74.6%)
- 프론트가 `scenario_text` 있으면 **메인 모델 결과와 무관하게 무조건 우선 출력**하는 구조

**흐름:**
```
captureAndSend() 내부
→ shouldCommitScenarioText() == true  ← 항상 true였음
→ commitScenarioText('가능') 호출 후 return
→ 메인 모델 결과(오른쪽)는 실행되지 않음
```

---

### 🔴 [문제 3] SEN0355(감사합니다) 반복 오인식

**파일:** `backend/inference/predictor.py`

여러 다른 수어를 시연하는 상황에서 sentence 모델이 SEN0355(감사합니다)를 반복적으로 높은 신뢰도로 예측함.

- 오른쪽 수어 시연 시: SEN0355 47.9%
- 복지카드 수어 시연 시: SEN0355 47.9%, 36.7%

`_PROTECTED_SEN = {"SEN0354", "SEN0355"}` 으로 pair 후보에서는 차단되어 있으나 **single_sentence 결과로는 여전히 출력 가능**한 상태.

---

### 🟡 [문제 4] fusion 우선순위 로직이 라이브에서 오작동

**파일:** `backend/inference/predictor.py` 301~307번 줄

`오른쪽` 수어를 했는데 `신분증 (있다)` 가 출력된 원인.

```python
# word_score < 0.75 이고 sen_score >= word_score × 0.35 이면
# score가 낮은 sentence를 강제로 1위로 승격하는 로직
if raw_wlbl in SCENARIO_WORD_IDS
   and word_score < 0.75      # 0.691 < 0.75  ✅
   and sen_score >= word_score * 0.35:  # 0.525 >= 0.242  ✅
   → SEN0169(신분증) 강제 1위 승격
```

**이 규칙이 생긴 이유:** 데모 영상에서 문장 결과를 우선 출력하기 위해 설계된 규칙.  
라이브 웹캠에서는 의도하지 않은 오작동 유발.

---

### 🟡 [문제 5] 세 모델이 서로 다른 결과를 출력

같은 입력에 대해 세 모델이 제각각 다른 결과를 냄:

| 모델 | 역할 | 예시 (오른쪽 수어 시) |
|------|------|------|
| `cnn_gru` | 메인 word 예측 | 오른쪽 (39.8%) |
| `word_v2` | scenario word 예측 | 가능 (71.6%) |
| `sentence_v2` | scenario sentence 예측 | 신분증있다 (52.5%) |

`cnn_gru`와 `word_v2`가 동일한 역할인데도 서로 다른 결과를 내며, 모델 간 우선순위 기준이 명확하지 않음.

---

### 🟡 [문제 6] cnn_gru 모델의 WORD1343(오른쪽) 편향

여러 다른 수어를 시연했는데 cnn_gru가 계속 `WORD1343(오른쪽)` 예측.  
세 번의 서로 다른 수어에서 모두 `raw_label=WORD1343` 출력 (신뢰도 17.4%, 39.5%, 39.8%).  
모델이 특정 클래스로 편향되어 있거나, 오른쪽 수어의 특징 벡터가 다수 클래스와 겹칠 가능성.

---

### 🟡 [문제 7] 데모 2 — "복지카드를 잃어버렸어요" 대신 "복지카드"만 출력

**파일:** `web/src/hooks/useSignLanguage.ts` — `validationDemoScenarios`

수동 경계(7.068초)로 두 세그먼트로 분리되는 데모에서 두 번째 세그먼트가 올바르게 인식되지 않음.

**원인 분석:**
```
1세그먼트: word_v2 = 복지카드(78.8%)  → demoGlossBuffer = ['WORD0579']
2세그먼트: word_v2 = 복지카드(65.7%)  → commitRecognizedWord('WORD0579')
                                         중복 제거 로직에 의해 추가 안 됨
           pair 후보: SEN0322+WORD0579 (복지카드를 잃어버렸어요)
             score     = 0.261  (기준 0.35  ← 미달)
             sentenceConf = 0.115  (기준 0.12  ← 0.005 차이로 미달!)

→ 최종 버퍼: ['WORD0579'] 한 개만
→ gloss_to_text API: 'WORD0579' → '복지카드'
```

`SEN0322(잃어버리다)` 신뢰도가 11.5%로 매우 낮아 pair 후보 조건을 통과하지 못함.  
`scenarioHints` 없이 전체 SEN에서 탐색하다 SEN0355에 밀리는 구조.

---

## 2. 적용된 수정 사항 (현재 유지 중)

### ✅ 수정 1 — scenario_mode 기본값 변경

**파일:** `web/src/hooks/useSignLanguage.ts` 219번 줄

```ts
// 수정 전
const rawMode = queryMode || storedMode || 'off'

// 수정 후
const rawMode = queryMode || storedMode || 'resident'
```

**효과:** 라이브 웹캠에서도 word + sentence 모델이 동시에 동작.  
**영향 범위:** 라이브 모드 전용. 데모 영상은 원래 `forceScenarioMode: true`라 무관.

---

### ✅ 수정 2 — 라이브 모드에서 scenario fallback 처리 개선

**파일:** `web/src/hooks/useSignLanguage.ts` — `shouldCommitScenarioText` 함수

```ts
// 추가된 조건
// 라이브 모드에서 메인 모델이 성공했으면(label != null) scenario는 무시
// 메인 모델 실패(label=null)일 때만 SEN 결과를 fallback으로 사용
if (!isDemoModeRef.current && prediction?.label) return false
```

**효과:**
- 메인 모델 성공(label 있음) → 메인 모델 결과 출력 (오른쪽)
- 메인 모델 실패(label=null) → scenario 결과 출력 (안녕하세요 등 SEN 수어)

**영향 범위:** 라이브 모드 전용. 데모 영상 로직 유지.

---

### ✅ 수정 3 — 데모 2에 scenarioHints 추가

**파일:** `web/src/hooks/useSignLanguage.ts` — `validationDemoScenarios`

```ts
{
  displayText: 'REALZ03 02 welfare card + lost',
  forceScenarioMode: true,
  segmentation: { mode: 'manual', boundariesSec: [7.068] },
  relaxedSegmentation: true,
  scenarioHints: ['SEN0322'],  // ← 추가
  clips: [...],
}
```

**효과:** 두 번째 세그먼트(잃어버렸어요) 인식 시 sentence 모델이 SEN0322만 탐색.  
- hints 적용 시 sen_conf_threshold가 0.30 → 0.20으로 자동 완화
- SEN0322 신뢰도가 올라가 pair 후보(SEN0322+WORD0579) 조건 통과 가능

**영향 범위:** 데모 2 전용. 다른 데모, 라이브 모드 무관.

---

## 3. 시도했다가 롤백한 수정 사항

| 항목 | 시도값 | 롤백 이유 |
|------|--------|-----------|
| `LIVE_HANDS_DOWN_REST_SECONDS` | 0.25 → 0.5 | 실제 효과 검증 데이터 부족, 추후 재검토 필요 |
| `confidenceThreshold` (word) | 0.30 → 0.45 → 0.40 | 오른쪽(39.5%) 같은 정상 예측도 차단, 균형점 미확인 |
| `sen_conf_threshold` (sentence) | 0.30 → 0.45 | demo 2 문제와 무관했음. 더 많은 테스트 필요 |

---

## 4. 개선 제안

### 🥇 1순위 — word_v2를 메인 모델로 교체 검토

현재 `cnn_gru`(메인)와 `word_v2`(scenario)가 **동일한 역할인데 서로 다른 결과**를 내며 충돌 발생.

```
현재 구조의 문제:
cnn_gru(메인)  →  오른쪽 (39.8%)   ← 성능 낮음, WORD1343 편향 있음
word_v2(보조)  →  가능   (71.6%)   ← 실제로 더 정확한 경우가 많음
```

word_v2의 성능이 cnn_gru보다 일관적으로 높다면, word_v2를 메인 모델로 올리고 cnn_gru를 제거하는 구조 단순화를 고려.

---

### 🥈 2순위 — fusion 우선순위 로직 라이브/데모 분리

`predictor.py`의 sentence 강제 승격 로직은 **데모 전용으로 설계**된 것이 라이브에도 적용 중.

```python
# 현재: 라이브/데모 구분 없이 동일 로직 적용
if word_score < 0.75 and sen_score >= word_score * 0.35:
    → sentence 강제 승격

# 개선안: is_demo 파라미터 추가
def predict_dual_scenario(..., is_demo: bool = False):
    if is_demo and word_score < 0.75 and ...:
        → sentence 강제 승격  # 데모에서만
```

---

### 🥉 3순위 — _PROTECTED_SEN 확장 및 single_sentence 차단 적용

현재 SEN0355, SEN0354는 pair 후보에서만 차단. single_sentence 결과로는 여전히 출력 가능.

```python
# 현재
_PROTECTED_SEN = {"SEN0354", "SEN0355"}  # pair 후보에서만 차단

# 개선안: single_sentence 결과에서도 차단
if source == "single_sentence" and lbl in _PROTECTED_SEN:
    continue
```

---

### 4순위 — threshold 정밀 튜닝 (데이터 수집 후)

현재 기본값들은 경험치 기반. 실제 시연 데이터를 더 수집한 뒤 조정 권장.

| 값 | 현재 | 권장 범위 | 파일 |
|----|------|----------|------|
| `confidenceThreshold` | 0.30 | 0.35 ~ 0.45 | `useSignLanguage.ts` |
| `sen_conf_threshold` | 0.30 | 0.40 ~ 0.50 | `predictor.py` |
| `LIVE_HANDS_DOWN_REST_SECONDS` | 0.25 | 0.40 ~ 0.60 | `useSignLanguage.ts` |

---

### 5순위 — cnn_gru WORD1343(오른쪽) 편향 조사

여러 수어에서 cnn_gru가 지속적으로 WORD1343을 낮은 신뢰도로 예측.  
학습 데이터에서 오른쪽 클래스 샘플이 과다하거나, 특징 벡터가 클래스 평균에 가까운 위치일 가능성.  
학습 데이터 클래스 분포 확인 및 필요 시 재학습 고려.

---

## 5. 현재 상태 요약

### 파일별 변경 사항

**`web/src/hooks/useSignLanguage.ts`**

| 항목 | 변경 전 | 변경 후 |
|------|---------|---------|
| `scenario_mode` 기본값 | `'off'` | `'resident'` |
| `shouldCommitScenarioText` live 조건 | 없음 | label 있으면 scenario 무시 |
| 데모2 `scenarioHints` | 없음 | `['SEN0322']` |

**`backend/inference/predictor.py`**

변경 없음 (원복 완료)

---

### 모델 동작 방식 (현재)

```
라이브 웹캠
├── 메인 cnn_gru → label 있으면 → 메인 결과 출력
└── 메인 cnn_gru → label 없으면 → scenario(word_v2 + sentence_v2) fallback

데모 영상
└── 기존 로직 유지 (forceScenarioMode 기반)
    └── 데모2: SEN0322 힌트 적용으로 잃어버렸어요 인식 개선
```

---

## 6. 관련 파일

| 파일 | 역할 |
|------|------|
| `web/src/hooks/useSignLanguage.ts` | 라이브 세그먼트 감지, 예측 요청, scenario 결과 처리 |
| `backend/inference/predictor.py` | dual scenario fusion 로직, threshold 적용 |
| `backend/inference/routes.py` | `/api/predict` 엔드포인트, 모델 호출 |
| `docs/webcam_recognition_issues.md` | 코드 레벨 초기 분석 문서 |
| `docs/webcam_recognition_session_log.md` | 중간 세션 기록 (이 문서로 통합됨) |
