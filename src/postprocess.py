"""Post-processing helpers for extracted KIE fields."""

import re
import unicodedata

from data_preparation.normalize import normalize_number_token, repair_mojibake


DATE_PATTERNS = [
    r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}",
    r"\d{1,2}\.\d{1,2}\.\d{2,4}",
    r"\d{1,2}\s*tháng\s*\d{1,2}\s*năm\s*\d{2,4}",
    r"\d{1,2}\s*thang\s*\d{1,2}\s*nam\s*\d{2,4}",
]

TIME_PATTERNS = [
    r"\d{1,2}\s*:\s*\d{2}(\s*:\s*\d{2})?(\s*(am|pm))?",
]


def normalize_text(text: str) -> str:
    """Unicode NFC + strip + collapse whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", repair_mojibake(str(text)))
    text = re.sub(r"\s+", " ", text.strip())
    return text


def normalize_money(text: str) -> str:
    """Extract and normalize the most likely money number from OCR text."""
    if not text:
        return ""
    t = normalize_text(text).lower()
    for unit in ["vnđ", "vnd", "đồng", "dong", "đ", "rm", "sgd"]:
        t = t.replace(unit, "")
    numbers = re.findall(r"\d[\d.,]*", t)
    if not numbers:
        return ""
    return normalize_number_token(max(numbers, key=len))


def normalize_timestamp(text: str) -> str:
    """Extract date + optional time from OCR text, preserving original date order."""
    if not text:
        return ""
    t = normalize_text(text)
    t_lower = t.lower()

    date_match = None
    for pattern in DATE_PATTERNS:
        match = re.search(pattern, t_lower)
        if match:
            date_match = re.sub(r"\s+", " ", match.group(0)).strip()
            break

    time_match = None
    for pattern in TIME_PATTERNS:
        match = re.search(pattern, t_lower)
        if match:
            time_match = re.sub(r"\s*:\s*", ":", match.group(0))
            time_match = re.sub(r"\s+", " ", time_match).strip()
            break

    if date_match and time_match:
        return f"{date_match} {time_match}"
    if date_match:
        return date_match
    return t
