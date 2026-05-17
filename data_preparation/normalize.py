"""Shared normalization helpers for data preparation."""

import re
import unicodedata


MOJIBAKE_MARKERS = ("Ã", "Ä", "Æ", "á»", "áº", "â")


def repair_mojibake(text: str) -> str:
    """Repair common UTF-8-as-Latin-1/Windows-1252 mojibake in raw CSVs."""
    if not text:
        return ""
    text = str(text)
    if not any(marker in text for marker in MOJIBAKE_MARKERS):
        return text
    for encoding in ("cp1252", "latin1"):
        try:
            repaired = text.encode(encoding).decode("utf-8")
        except UnicodeError:
            continue
        if repaired and repaired != text:
            return repaired
    return text


def normalize_unicode(text: str) -> str:
    """Normalize Unicode to NFC after repairing common mojibake."""
    if not text:
        return ""
    return unicodedata.normalize("NFC", repair_mojibake(str(text)))


def normalize_whitespace(text: str) -> str:
    """Strip and collapse repeated whitespace."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip())


def normalize_text(text: str) -> str:
    """Apply Unicode and whitespace normalization."""
    return normalize_whitespace(normalize_unicode(text))


def normalize_number_token(value: str) -> str:
    """Normalize one numeric token while preserving likely decimal values."""
    value = normalize_whitespace(value)
    if not value:
        return ""

    if "." in value and "," in value:
        last_dot = value.rfind(".")
        last_comma = value.rfind(",")
        decimal_sep = "." if last_dot > last_comma else ","
        thousands_sep = "," if decimal_sep == "." else "."
        integer_part, decimal_part = value.rsplit(decimal_sep, 1)
        if decimal_part.isdigit() and 1 <= len(decimal_part) <= 2:
            integer_part = re.sub(r"[^0-9]", "", integer_part.replace(thousands_sep, ""))
            return f"{integer_part}.{decimal_part}"

    decimal_match = re.match(r"^(.+)[.,](\d{1,2})$", value)
    if decimal_match:
        integer_part = re.sub(r"[^\d]", "", decimal_match.group(1))
        decimal_part = decimal_match.group(2)
        return f"{integer_part}.{decimal_part}"

    return re.sub(r"[^\d]", "", value)


def normalize_money_gt(text: str) -> str:
    """Normalize TOTAL_COST while preserving true decimal separators."""
    if not text:
        return ""
    text = normalize_unicode(text).lower()
    for unit in ["vnđ", "vnd", "đồng", "dong", "đ", "rm", "sgd"]:
        text = text.replace(unit, "")
    numbers = re.findall(r"\d[\d.,]*", text)
    if not numbers:
        return ""
    return normalize_number_token(max(numbers, key=len))


def normalize_timestamp_gt(text: str) -> str:
    """Normalize ground-truth timestamps without changing their date order."""
    if not text:
        return ""
    return normalize_text(text)
