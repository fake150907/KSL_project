# preprocess_report_REAL10_REAL12

- 작업 일시: 2026-04-29 09:12:24
- 소요 시간: 45.3초

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

## 완료 조건 체크
- [O] MVP 8개 단어만 포함
- [O] layout mediapipe_xyz 225
- [O] 정규화 shoulder-center + shoulder-width scale
- [O] NPZ 생성 완료
- [O] failed_samples 기록 완료