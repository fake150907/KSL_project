# preprocess_report_REAL10_REAL12

- 작업 일시: 2026-04-28 17:52:21
- 소요 시간: 41.8초

## 고정 기준
- layout: mediapipe_xyz
- feature: pose33 + left_hand21 + right_hand21
- point: [x, y, z]
- feature count: 225
- sequence length: 32
- normalization: shoulder-center + shoulder-width scale

## 영상 폴더
- E:\study\KSL_data_file\[원천]10_real_word_video\05-1
- E:\study\KSL_data_file\[원천]11_real_word_video\06
- E:\study\KSL_data_file\[원천]12_real_word_video\06-1

## 처리 결과
- 총 스캔: 22500개
- 매칭 성공: 29개
- 실패: 0개
- 스킵 (MVP 외 라벨 등): 22471개

## 라벨별 샘플 수
- 가다: 8개
- 감사: 2개
- 괜찮다: 2개
- 배고프다: 4개
- 병원: 2개
- 아프다: 2개
- 우유: 4개
- 자다: 5개

## 사용 스크립트
- extract_mediapipe_REAL10_REAL12.py
- 실행: python extract_mediapipe_REAL10_REAL12.py

## 특이사항
- 담당 폴더(10, 11, 12번)의 영상 파일명에 REAL05, REAL06이 포함되어 있음
- zip 번호 != REAL 번호 원칙에 따라 파일명 내 REAL 번호 그대로 기재
- morpheme 매칭은 WORD번호 + 각도 기준으로 수행 (REAL번호 무시)
- F각도(정면)만 처리 (베이스 모델 검증 목적)

## 완료 조건 체크
- [O] MVP 8개 단어만 포함
- [O] layout mediapipe_xyz 225
- [O] 정규화 shoulder-center + shoulder-width scale
- [O] NPZ 생성 완료
- [O] failed_samples 기록 완료