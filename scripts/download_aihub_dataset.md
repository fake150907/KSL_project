# AI Hub Dataset Download Notes

Dataset: AI Hub Korean sign-language video dataset, `dataSetSn=103`.

Do not hard-code API keys in source files. Put the key in your shell only:

```powershell
$env:AIHUB_API_KEY="YOUR_KEY_HERE"
```

AI Hub notes that API download is available after dataset approval. The site also notes that download files may be split as `zip.part*`, and recommends Linux/WSL for download and merge work.

Recommended local target layout:

```text
data/raw/
  Training/REAL/WORD/real_word_keypoint/
  Training/REAL/WORD/real_word_morpheme/
  Validation/REAL/WORD/real_word_keypoint/
  Validation/REAL/WORD/real_word_morpheme/
```

After downloading split files in WSL/Linux, merge parts like this:

```bash
find "download_folder" -name "file.zip.part*" -print0 | sort -zt'.' -k2V | xargs -0 cat > "file.zip"
```

For this MVP, download only the WORD keypoint and morpheme files first. Avoid SEN, video, CROWD, and SYN until the quick pipeline succeeds.

The first safe step is morpheme-only label discovery:

```powershell
$env:AIHUB_APIKEY="YOUR_KEY"
.\scripts\download_aihub_word_morpheme.ps1
```

Known useful file keys are documented in `scripts/aihub_filekeys_103.md`.
