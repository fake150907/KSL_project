# AIHub 용량 메모

## 목적

AIHub `WORD` 데이터 다운로드 시 필요한 저장공간을 미리 가늠할 수 있도록, 압축 파일 기준 용량과 압축 해제 후 예상 용량을 정리한 메모임.

기준 정보는 아래 파일을 참고했음.

- [aihub_filekeys_103.md](C:/github/ai-project-01/scripts/aihub_filekeys_103.md)

## 1. 압축 파일 기준 용량

### Training / REAL / WORD / Labeling

구성:

- keypoint zip `16개`
- morpheme zip `1개`

keypoint 개별 용량:

```text
01  11 GB
02   9 GB
03  11 GB
04  14 GB
05  10 GB
06  11 GB
07  10 GB
08  11 GB
09  11 GB
10  11 GB
11  11 GB
12  10 GB
13  11 GB
14  12 GB
15  11 GB
16  10 GB
```

합계:

- Training keypoint 총합: `174 GB`
- Training morpheme 총합: `110 MB`
- Training 전체 총합: 약 `174.11 GB`

### Validation / REAL / WORD / Labeling

구성:

- keypoint zip `1개`
- morpheme zip `1개`

합계:

- Validation keypoint 총합: `21 GB`
- Validation morpheme 총합: `14 MB`
- Validation 전체 총합: 약 `21.01 GB`

### 전체 합계

- keypoint zip 총 `17개`
- morpheme zip 총 `2개`
- 압축 파일 기준 전체 용량: 약 `195.12 GB`

## 2. 압축 해제 후 예상 용량

정확한 공식 값은 아니며, 현재 로컬에서 실제 morpheme 압축 해제 결과와 JSON 계열 파일의 일반적인 압축률을 바탕으로 추정한 값임.

### morpheme 기준 실제 확인값

현재 로컬에서 확인된 morpheme 추출 결과:

- 압축 기준: 약 `124 MB`
- 실제 풀린 결과: 약 `0.367 GB`

즉 morpheme은 대략 `약 3배 수준`으로 커졌다고 볼 수 있음.

이를 기준으로 추정하면:

- Training morpheme `110 MB` -> 약 `0.32 ~ 0.35 GB`
- Validation morpheme `14 MB` -> 약 `0.04 GB`

### keypoint 해제 예상

keypoint는 JSON 프레임 파일 묶음이라 압축이 잘 되는 편임.

따라서 실제 압축 해제 시:

- 최소 `약 3배`
- 많으면 `약 4배 이상`

정도로 커질 가능성을 보는 것이 안전함.

이를 기준으로 추정하면:

- Training keypoint `174 GB` -> 약 `520 ~ 700 GB`
- Validation keypoint `21 GB` -> 약 `63 ~ 84 GB`

### 전체 해제 예상

- Training 전체: 약 `520 ~ 700 GB + 0.35 GB`
- Validation 전체: 약 `63 ~ 84 GB + 0.04 GB`

즉 전체적으로는 대략:

```text
약 583 ~ 784 GB
```

수준으로 예상하는 것이 안전함.

## 3. 실무 판단

한 줄 정리:

- 압축 파일 기준: 약 `195 GB`
- 압축 해제 후 예상: 대략 `600 ~ 800 GB` 수준

즉 전체 `WORD keypoint + morpheme` 를 한 번에 다 받는 것은 저장공간 부담이 매우 큼.

## 4. 왜 subset 전략을 쓰는가

현재 프로젝트가 전체 keypoint를 한 번에 다 쓰지 않고 subset 전략을 쓰는 이유는 아래와 같음.

- keypoint 전체 용량이 너무 큼
- morpheme만 먼저 받아 라벨 분포 확인 가능함
- 필요한 sample id만 골라 keypoint zip에서 부분 추출하는 방식이 훨씬 현실적임
- 발표용 MVP는 전체 데이터보다 생활형 소수 클래스 안정화가 더 중요함

## 5. 권장 다운로드 순서

```text
1. morpheme 먼저 다운로드
2. label 후보 분석
3. 최종 라벨 선택
4. selected_label_targets.csv 생성
5. 필요한 keypoint zip만 다운로드
6. target sample만 부분 추출
```

## 6. 현재 프로젝트 관점의 결론

현재 프로젝트 기준으로는:

- 전체 `WORD` 데이터 일괄 다운로드보다
- `morpheme -> 라벨 선택 -> target sample 생성 -> keypoint subset 추출`

방식이 훨씬 현실적이고 안전함.
