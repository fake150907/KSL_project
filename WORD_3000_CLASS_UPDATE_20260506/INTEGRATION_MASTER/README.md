# INTEGRATION_MASTER

취합/통합 담당자용 V2 소스입니다.

수정 대상 원본:

```text
INTEGRATION_MASTER/scripts/merge_files.py
INTEGRATION_MASTER/scripts/merge_shards.py
```

V2 변경 핵심:

```python
word_id = row["word_id"]
y = label_map_word_id_3000[word_id]
```

취합 후 `labels`는 현재 데이터에 등장한 라벨만 쓰지 않고, `WORD0001~WORD3000` 전체 순서로 고정합니다.

