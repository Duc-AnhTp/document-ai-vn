"""Run the full data preparation pipeline."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preparation.prepare_mcocr import main as prepare_mcocr
from data_preparation.prepare_sroie import main as prepare_sroie
from data_preparation.stats import main as print_stats


def _warn_if_needed(name: str, result: dict | None) -> None:
    if not result:
        return
    total = result.get("train", 0) + result.get("val", 0)
    if total == 0:
        print(f"WARNING: {name} produced 0 records.")
    if result.get("val", 0) == 0:
        print(f"WARNING: {name} produced no validation records.")


def main():
    print("=" * 60)
    print("Data Preparation Pipeline")
    print("=" * 60)

    print("\n[1/2] MC-OCR")
    mcocr_result = prepare_mcocr()
    _warn_if_needed("MC-OCR", mcocr_result)

    print("\n[2/2] SROIE")
    sroie_result = prepare_sroie()
    _warn_if_needed("SROIE", sroie_result)

    print("\n[Stats]")
    print_stats()

    print("=" * 60)
    print("Done. Processed data is available in data/processed/.")
    print("=" * 60)


if __name__ == "__main__":
    main()
