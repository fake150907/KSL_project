"""Helpers for AI Hub sign-language sample ids."""

from __future__ import annotations

import re


VALID_ANGLES = {"F", "U", "D", "R", "L"}
WORD_ID_RE = re.compile(r"(WORD\d+)")


def parse_angle(sample_id: str) -> str:
    """Return final F/U/D/R/L camera-angle marker from a sample id."""
    angle = str(sample_id).rsplit("_", 1)[-1]
    return angle if angle in VALID_ANGLES else ""


def parse_word_id(sample_id: str) -> str:
    """Return WORDxxxx from an AI Hub sample id."""
    match = WORD_ID_RE.search(str(sample_id))
    return match.group(1) if match else ""


def word_angle_key(sample_id: str) -> str:
    """Return a REAL-independent matching key such as WORD0943_F."""
    word_id = parse_word_id(sample_id)
    angle = parse_angle(sample_id)
    return f"{word_id}_{angle}" if word_id and angle else ""
