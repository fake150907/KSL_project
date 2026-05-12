"""Summarize Flask webcam prediction logs.

Input:
  - outputs/logs/flask_web_demo.out.log
Output:
  - console summary
  - optional CSV with parsed prediction rows
Example:
  python scripts/summarize_webcam_predictions.py
  python scripts/summarize_webcam_predictions.py --csv outputs/reports/webcam_prediction_log_summary.csv
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


LINE_RE = re.compile(
    r"Prediction: "
    r"label=(?P<label>.*?), "
    r"conf=(?P<conf>[0-9.]+), "
    r"raw=(?P<raw_label>.*?)/(?P<raw_conf>.*?), "
    r"has_hand=(?P<has_hand>True|False), "
    r"window=(?P<window_current>\d+)/(?P<window_size>\d+), "
    r"miss=(?P<miss_current>\d+)/(?P<miss_max>\d+), "
    r"window_filled=(?P<window_filled>.*?), "
    r"top=(?P<top>.*)$"
)


def _none_if_text(value: str) -> str | None:
    value = value.strip()
    return None if value in {"None", "null", ""} else value


def _float_or_none(value: str) -> float | None:
    value = value.strip()
    if value in {"None", "null", ""}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_top(value: str) -> list[dict[str, Any]]:
    value = value.strip()
    if not value or value == "[]":
        return []
    try:
        parsed = ast.literal_eval(value)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def parse_log(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            match = LINE_RE.search(line.strip())
            if not match:
                continue
            groups = match.groupdict()
            top = _parse_top(groups["top"])
            rows.append(
                {
                    "line_no": line_no,
                    "label": _none_if_text(groups["label"]),
                    "confidence": float(groups["conf"]),
                    "raw_label": _none_if_text(groups["raw_label"]),
                    "raw_confidence": _float_or_none(groups["raw_conf"]),
                    "has_hand": groups["has_hand"] == "True",
                    "window_current": int(groups["window_current"]),
                    "window_size": int(groups["window_size"]),
                    "miss_current": int(groups["miss_current"]),
                    "miss_max": int(groups["miss_max"]),
                    "window_filled": _none_if_text(groups["window_filled"]),
                    "top1": top[0].get("label") if len(top) > 0 and isinstance(top[0], dict) else None,
                    "top1_confidence": top[0].get("confidence") if len(top) > 0 and isinstance(top[0], dict) else None,
                    "top2": top[1].get("label") if len(top) > 1 and isinstance(top[1], dict) else None,
                    "top2_confidence": top[1].get("confidence") if len(top) > 1 and isinstance(top[1], dict) else None,
                    "top3": top[2].get("label") if len(top) > 2 and isinstance(top[2], dict) else None,
                    "top3_confidence": top[2].get("confidence") if len(top) > 2 and isinstance(top[2], dict) else None,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No prediction rows found.")
        return

    total = len(rows)
    hand_rows = sum(row["has_hand"] for row in rows)
    filled_rows = sum(row["window_current"] >= row["window_size"] for row in rows)
    max_window = max(row["window_current"] for row in rows)
    predicted = [row for row in rows if row["raw_label"]]
    stable = [row for row in rows if row["label"] and row["label"] not in {"불확실"}]

    print(f"rows: {total}")
    print(f"hand_detected_rows: {hand_rows} ({hand_rows / total:.1%})")
    print(f"max_window_progress: {max_window}/{rows[0]['window_size']}")
    print(f"window_filled_rows: {filled_rows}")
    print(f"raw_prediction_rows: {len(predicted)}")
    print(f"stable_label_rows: {len(stable)}")

    if predicted:
        print("\nraw label counts:")
        for label, count in Counter(row["raw_label"] for row in predicted).most_common():
            print(f"- {label}: {count}")

    top_labels = [row["top1"] for row in rows if row["top1"]]
    if top_labels:
        print("\ntop-1 counts:")
        for label, count in Counter(top_labels).most_common():
            print(f"- {label}: {count}")

    print("\ninterpretation:")
    if filled_rows == 0:
        print("- No sequence window reached full length. Improve camera/hand detection before judging model quality.")
    elif len(predicted) == 0:
        print("- Windows filled, but no raw predictions were parsed. Check backend logging format.")
    else:
        print("- Predictions are being produced. Use the CSV to compare expected labels against top-3 outputs.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="outputs/logs/flask_web_demo.out.log")
    parser.add_argument("--csv", default="outputs/reports/webcam_prediction_log_summary.csv")
    args = parser.parse_args()

    rows = parse_log(Path(args.log))
    print_summary(rows)
    if rows and args.csv:
        write_csv(Path(args.csv), rows)
        print(f"\ncsv: {args.csv}")


if __name__ == "__main__":
    main()
