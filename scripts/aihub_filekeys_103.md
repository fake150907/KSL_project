# AI Hub Dataset 103 Useful File Keys

Source: `https://api.aihub.or.kr/info/103.do` via AI Hub Shell file tree.

For the MVP we avoid raw video, `SEN`, `CROWD`, and `SYN`.

## Training / REAL / WORD / Labeling

- `01_real_word_keypoint.zip` | 11 GB | filekey `39600`
- `01_real_word_morpheme.zip` | 110 MB | filekey `39601`
- `02_real_word_keypoint.zip` | 9 GB | filekey `39602`
- `03_real_word_keypoint.zip` | 11 GB | filekey `39603`
- `04_real_word_keypoint.zip` | 14 GB | filekey `39604`
- `05_real_word_keypoint.zip` | 10 GB | filekey `39605`
- `06_real_word_keypoint.zip` | 11 GB | filekey `39606`
- `07_real_word_keypoint.zip` | 10 GB | filekey `39607`
- `08_real_word_keypoint.zip` | 11 GB | filekey `39608`
- `09_real_word_keypoint.zip` | 11 GB | filekey `39609`
- `10_real_word_keypoint.zip` | 11 GB | filekey `39610`
- `11_real_word_keypoint.zip` | 11 GB | filekey `39611`
- `12_real_word_keypoint.zip` | 10 GB | filekey `39612`
- `13_real_word_keypoint.zip` | 11 GB | filekey `39613`
- `14_real_word_keypoint.zip` | 12 GB | filekey `39614`
- `15_real_word_keypoint.zip` | 11 GB | filekey `39615`
- `16_real_word_keypoint.zip` | 10 GB | filekey `39616`

## Validation / REAL / WORD / Labeling

- `01_real_word_morpheme.zip` | 14 MB | filekey `39478`
- `09_real_word_keypoint.zip` | 21 GB | filekey `39479`

## Recommended First Download

Download morpheme only first:

```powershell
$env:AIHUB_APIKEY="YOUR_KEY"
.\scripts\download_aihub_word_morpheme.ps1
```

Then extract labels:

```powershell
python -m src.data.extract_labels --quick_test --max_classes 8
```

Do not download keypoint files until enough disk space is confirmed. The smallest relevant training keypoint file is still about 9 GB.
