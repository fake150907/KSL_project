param(
    [string]$DatasetKey = "103",
    [string]$FileKeys = "39601,39478",
    [string]$OutputDir = "data/raw/aihub_downloads/word_morpheme"
)

# Role: download only small AI Hub WORD morpheme archives for label discovery.
# Input: $env:AIHUB_APIKEY or $env:AIHUB_API_KEY
# Output: downloaded tar contents under data/raw/aihub_downloads/word_morpheme
# Example:
#   $env:AIHUB_APIKEY="YOUR_KEY"
#   .\scripts\download_aihub_word_morpheme.ps1

$ErrorActionPreference = "Stop"
$apiKey = $env:AIHUB_APIKEY
if (-not $apiKey) {
    $apiKey = $env:AIHUB_API_KEY
}
if (-not $apiKey) {
    throw "Set AIHUB_APIKEY first. Example: `$env:AIHUB_APIKEY='YOUR_KEY'"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$downloadTar = Join-Path $OutputDir "download.tar"
$url = "https://api.aihub.or.kr/down/0.6/$DatasetKey.do?fileSn=$FileKeys"

Write-Host "Downloading AI Hub dataset $DatasetKey fileSn=$FileKeys"
Write-Host "Target: $OutputDir"
curl.exe -L -C - -H "apikey:$apiKey" -o $downloadTar $url

if ((Get-Item $downloadTar).Length -lt 1024) {
    Write-Warning "Downloaded response is very small. It may be an approval/API error message."
    Get-Content $downloadTar -ErrorAction SilentlyContinue
    exit 2
}

tar -xf $downloadTar -C $OutputDir
Write-Host "Extracted tar to $OutputDir"

$zipFiles = Get-ChildItem -Path $OutputDir -Recurse -Filter "*.zip" -File
foreach ($zip in $zipFiles) {
    $dest = Join-Path $zip.DirectoryName $zip.BaseName
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    Write-Host "Expanding $($zip.FullName) -> $dest"
    Expand-Archive -LiteralPath $zip.FullName -DestinationPath $dest -Force
}

$partFiles = Get-ChildItem -Path $OutputDir -Recurse -Filter "*.part*" -File
if ($partFiles.Count -gt 0) {
    Write-Warning "zip.part files were extracted. Merge them before unzip. See scripts/download_aihub_dataset.md"
}

Write-Host "Done."
