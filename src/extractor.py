"""Rule-based field extraction for receipt KIE."""

import re
import unicodedata
from typing import Dict, List, Optional, Tuple

from src.line_processing import add_line_features
from src.postprocess import normalize_money, normalize_text, normalize_timestamp


FIELDS = ["SELLER", "SELLER_ADDRESS", "TIMESTAMP", "TOTAL_COST"]

TOTAL_KEYWORDS = [
    "tổng", "tong",
    "tổng tiền", "tong tien",
    "thanh toán", "thanh toan",
    "phải trả", "phai tra",
    "thành tiền", "thanh tien",
    "tổng cộng", "tong cong",
    "cộng", "cong",
    "total", "amount", "grand total", "to pay", "payment",
]

ADDRESS_KEYWORDS = [
    "đường", "duong",
    "phường", "phuong",
    "quận", "quan",
    "huyện", "huyen",
    "tỉnh", "tinh",
    "tp.", "tp ", "thành phố", "thanh pho",
    "số ", "so ", "ngõ", "ngo",
    "street", "road", "ave", "avenue", "blvd", "lane",
    "district", "city", "state", "jln", "jalan", "taman",
]

TIMESTAMP_KEYWORDS = [
    "ngày", "ngay", "date",
    "giờ", "gio", "time",
    "ngày bán", "ngay ban",
]

NEGATIVE_TOTAL_KEYWORDS = [
    "subtotal", "sub total", "vat", "tax", "thuế", "thue", "change", "cash",
]

DATE_PATTERNS = [
    r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}",
    r"\d{1,2}\.\d{1,2}\.\d{2,4}",
    r"\d{1,2}\s*tháng\s*\d{1,2}\s*năm\s*\d{2,4}",
    r"\d{1,2}\s*thang\s*\d{1,2}\s*nam\s*\d{2,4}",
]

TIME_PATTERNS = [
    r"\d{1,2}\s*:\s*\d{2}(\s*:\s*\d{2})?(\s*(am|pm))?",
]


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def _norm(text: str) -> str:
    text = normalize_text(text).lower()
    return _strip_accents(text)


def _contains_any(text: str, keywords: List[str]) -> bool:
    t = _norm(text)
    return any(_strip_accents(k.lower()) in t for k in keywords)


def _money_tokens(text: str) -> List[str]:
    return re.findall(r"\d[\d.,]*", normalize_text(text))


def _has_money_value(text: str) -> bool:
    return bool(normalize_money(text))


def _money_score(line: Dict) -> int:
    text = line["text"]
    score = 0
    if _contains_any(text, TOTAL_KEYWORDS):
        score += 3
    if _has_money_value(text):
        score += 4
    if re.search(r"\d{1,3}([.,]\d{3})+([.,]\d{1,2})?", text):
        score += 2
    if _contains_any(text, ["đ", "vnd", "vnđ", "dong", "rm", "sgd"]):
        score += 1
    if _contains_any(text, NEGATIVE_TOTAL_KEYWORDS):
        score -= 2
    return score


def _timestamp_score(text: str) -> int:
    t = normalize_text(text).lower()
    score = 0
    if any(re.search(pattern, t) for pattern in DATE_PATTERNS):
        score += 4
    if any(re.search(pattern, t) for pattern in TIME_PATTERNS):
        score += 2
    if _contains_any(text, TIMESTAMP_KEYWORDS):
        score += 1
    return score


def _address_score(line: Dict) -> int:
    text = line["text"]
    score = 0
    if _contains_any(text, ADDRESS_KEYWORDS):
        score += 4
    if line["relative_y"] < 0.45:
        score += 1
    if len(text.split()) >= 5:
        score += 1
    if _timestamp_score(text) > 0:
        score -= 4
    if _money_score(line) > 3:
        score -= 4
    return score


def _seller_score(line: Dict) -> int:
    text = line["text"]
    score = 0
    if line["relative_y"] < 0.35:
        score += 3
    words = text.split()
    if 1 <= len(words) <= 8:
        score += 1
    if len(text) >= 4:
        score += 1
    if _contains_any(text, ADDRESS_KEYWORDS):
        score -= 3
    if _timestamp_score(text) > 0:
        score -= 4
    if _money_score(line) > 3:
        score -= 4
    return score


def _best_line(lines: List[Dict], scores: List[int], threshold: int = 1) -> Optional[Dict]:
    if not lines:
        return None
    best_idx = max(range(len(lines)), key=lambda idx: scores[idx])
    return lines[best_idx] if scores[best_idx] >= threshold else None


def extract_total_cost_line(lines: List[Dict], mode: str = "scoring") -> Optional[Dict]:
    """Pick the line most likely to contain the payable total amount."""
    best_line = None
    best_score = -999
    for idx, line in enumerate(lines):
        score = _money_score(line)
        if mode == "scoring" and line.get("relative_y", 0) > 0.45:
            score += 1
        if not _has_money_value(line["text"]):
            next_line = lines[idx + 1] if idx + 1 < len(lines) else None
            if next_line and _contains_any(line["text"], TOTAL_KEYWORDS) and _has_money_value(next_line["text"]):
                score += 2
                candidate = next_line
            else:
                candidate = line
        else:
            candidate = line
        if _contains_any(line["text"], TOTAL_KEYWORDS) and _has_money_value(line["text"]):
            score += 2
        if score > best_score:
            best_score = score
            best_line = candidate
    if best_line is None or best_score <= 0 or not _has_money_value(best_line["text"]):
        return None
    return best_line


def extract_timestamp_line(lines: List[Dict], mode: str = "scoring") -> Optional[Dict]:
    scores = [_timestamp_score(line["text"]) for line in lines]
    return _best_line(lines, scores, threshold=1)


def extract_seller_line(lines: List[Dict], mode: str = "scoring") -> Optional[Dict]:
    if mode == "simple_rule":
        for line in lines[:5]:
            if _money_score(line) <= 0 and _timestamp_score(line["text"]) <= 0:
                return line
        return lines[0] if lines else None
    candidates = lines[: min(10, len(lines))]
    scores = [_seller_score(line) for line in candidates]
    return _best_line(candidates, scores, threshold=1)


def extract_address_line(lines: List[Dict], mode: str = "scoring") -> Optional[Dict]:
    if mode == "simple_rule":
        for line in lines[:12]:
            if _contains_any(line["text"], ADDRESS_KEYWORDS):
                return line
        return None
    candidates = lines[: min(15, len(lines))]
    scores = [_address_score(line) for line in candidates]
    return _best_line(candidates, scores, threshold=1)


def _line_text(line: Optional[Dict]) -> str:
    return line["text"] if line else ""


def _source_line_id(line: Optional[Dict]) -> Optional[int]:
    return line.get("line_id") if line else None


def extract_fields(
    lines: List[Dict],
    image_height: int,
    mode: str = "scoring",
    include_meta: bool = False,
) -> Dict[str, str]:
    """Extract the four target fields from OCR lines."""
    lines = add_line_features(lines, image_height)

    seller_line = extract_seller_line(lines, mode)
    address_line = extract_address_line(lines, mode)
    timestamp_line = extract_timestamp_line(lines, mode)
    total_line = extract_total_cost_line(lines, mode)

    fields = {
        "SELLER": normalize_text(_line_text(seller_line)),
        "SELLER_ADDRESS": normalize_text(_line_text(address_line)),
        "TIMESTAMP": normalize_timestamp(_line_text(timestamp_line)),
        "TOTAL_COST": normalize_money(_line_text(total_line)),
    }

    if include_meta:
        fields["_meta"] = {
            "source_line_ids": {
                "SELLER": _source_line_id(seller_line),
                "SELLER_ADDRESS": _source_line_id(address_line),
                "TIMESTAMP": _source_line_id(timestamp_line),
                "TOTAL_COST": _source_line_id(total_line),
            }
        }
    return fields


def extract_total_cost(lines: List[Dict], mode: str = "scoring") -> str:
    return _line_text(extract_total_cost_line(lines, mode))


def extract_timestamp(lines: List[Dict], mode: str = "scoring") -> str:
    return _line_text(extract_timestamp_line(lines, mode))


def extract_seller(lines: List[Dict], mode: str = "scoring") -> str:
    return _line_text(extract_seller_line(lines, mode))


def extract_address(lines: List[Dict], mode: str = "scoring") -> str:
    return _line_text(extract_address_line(lines, mode))


def format_mcocr_output(fields: Dict[str, str]) -> str:
    """Format output as SELLER|||ADDRESS|||TIMESTAMP|||TOTAL."""
    return "|||".join([fields.get(field, "") for field in FIELDS])
