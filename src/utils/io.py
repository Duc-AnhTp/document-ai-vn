"""Utility đọc/ghi file JSONL dùng cho pipeline preprocessing."""

import json
from pathlib import Path
from typing import Any, Dict, List


def save_jsonl(records: List[Dict[str, Any]], path: str | Path) -> None:
    """Lưu danh sách dict thành file JSONL (mỗi dòng = 1 JSON object)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        for record in records:
            line = json.dumps(record, ensure_ascii=False)
            f.write(line + '\n')


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Đọc file JSONL → danh sách dict."""
    path = Path(path)
    records: List[Dict[str, Any]] = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records
