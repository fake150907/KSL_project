"""Extract keyword snippets from AI Hub PDF docs.

Run:
  python scripts/extract_pdf_notes.py
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


PDFS = [
    Path("작업지시서/02. [자연어영역] 수어 영상.pdf"),
    Path("작업지시서/004.수어영상_데이터_구축_가이드라인.pdf"),
]
KEYWORDS = [
    "json",
    "keypoint",
    "keypoints",
    "attributes",
    "metaData",
    "start",
    "end",
    "hand_left",
    "pose",
    "face",
    "형태소",
    "비수지",
    "포맷",
]


def snippets(text: str, keyword: str, radius: int = 700) -> list[str]:
    lower = text.lower()
    key = keyword.lower()
    out: list[str] = []
    start = 0
    while True:
        idx = lower.find(key, start)
        if idx < 0:
            return out
        left = max(0, idx - radius)
        right = min(len(text), idx + len(keyword) + radius)
        out.append(text[left:right].strip())
        start = idx + len(keyword)


def main() -> None:
    output = Path("outputs/reports/pdf_schema_notes.txt")
    output.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[str] = []
    for pdf in PDFS:
        reader = PdfReader(str(pdf))
        chunks.append(f"\n\n===== {pdf.name} / pages={len(reader.pages)} =====\n")
        for page_no, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if not text:
                continue
            hits = []
            for keyword in KEYWORDS:
                if keyword.lower() in text.lower():
                    hits.append(keyword)
            if not hits:
                continue
            chunks.append(f"\n--- page {page_no} hits={', '.join(hits)} ---\n")
            chunks.append(text[:3500])
    output.write_text("\n".join(chunks), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
