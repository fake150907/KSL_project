# DATA_PREPROCESS_WORKER

데이터 전처리 담당자용 V2 소스입니다.

수정 대상 원본:

```text
REAL_SHARD_WORKER/src/data/preprocess_mediapipe_videos.py
```

V2 변경 핵심:

```python
word_id = row["word_id"]
y = label_map_word_id_3000[word_id]
```

좌표 추출 방식과 `X` shape `(N, 32, 225)`는 기존과 동일합니다.

