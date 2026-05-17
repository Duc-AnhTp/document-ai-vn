"""
stats.py — Thống kê dataset sau khi prepare.
"""
import json
import os


def _load(path: str) -> list:
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _field_stats(records: list) -> dict:
    """Tính tỉ lệ field rỗng và độ dài trung bình."""
    stats = {}
    fields = ["SELLER", "SELLER_ADDRESS", "TIMESTAMP", "TOTAL_COST"]
    total = len(records)
    if total == 0:
        return {}

    for field in fields:
        values = [r["gt"].get(field, "") for r in records]
        non_empty = [v for v in values if v]
        lengths = [len(v) for v in non_empty]
        stats[field] = {
            "filled": len(non_empty),
            "empty": total - len(non_empty),
            "fill_rate": f"{len(non_empty)/total*100:.1f}%",
            "avg_len": f"{sum(lengths)/len(lengths):.1f}" if lengths else "0",
        }
    return stats


def print_dataset_stats(name: str, train_path: str, val_path: str) -> None:
    train = _load(train_path)
    val = _load(val_path)

    print(f"\n{'='*50}")
    print(f" Dataset: {name}")
    print(f"{'='*50}")
    print(f"  Train: {len(train)} records")
    print(f"  Val:   {len(val)} records")
    print(f"  Total: {len(train)+len(val)} records")

    for split_name, records in [("Train", train), ("Val", val)]:
        if not records:
            continue
        print(f"\n  [{split_name} field stats]")
        stats = _field_stats(records)
        print(f"  {'Field':<20} {'Filled':>8} {'Empty':>8} {'Fill%':>8} {'AvgLen':>8}")
        print(f"  {'-'*56}")
        for field, s in stats.items():
            print(f"  {field:<20} {s['filled']:>8} {s['empty']:>8} {s['fill_rate']:>8} {s['avg_len']:>8}")


def main():
    datasets = [
        ("MC-OCR",
         os.path.join("data", "processed", "mc-ocr", "train.json"),
         os.path.join("data", "processed", "mc-ocr", "val.json")),
        ("SROIE",
         os.path.join("data", "processed", "sroie", "train.json"),
         os.path.join("data", "processed", "sroie", "val.json")),
    ]
    for name, train_path, val_path in datasets:
        print_dataset_stats(name, train_path, val_path)
    print()


if __name__ == "__main__":
    main()
