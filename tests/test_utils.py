"""Unit tests for core Donut utility functions."""

import os
import sys
from unittest.mock import MagicMock


if "yaml" not in sys.modules:
    sys.modules["yaml"] = MagicMock()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.utils import compute_metrics, normalize_field_value, parse_donut_output


class TestParseDonutOutput:
    def test_full_output(self):
        text = "<s_mcocr><s_store_name>ABC Shop</s_store_name><s_date>01/01/2024</s_date><s_total>100000</s_total><s_address>123 Main St</s_address></s>"
        result = parse_donut_output(text)
        assert result["store_name"] == "ABC Shop"
        assert result["date"] == "01/01/2024"
        assert result["total"] == "100000"
        assert result["address"] == "123 Main St"

    def test_missing_fields(self):
        text = "<s_mcocr><s_store_name>XYZ</s_store_name></s>"
        result = parse_donut_output(text)
        assert result["store_name"] == "XYZ"
        assert result["date"] == ""
        assert result["total"] == ""
        assert result["address"] == ""

    def test_empty_string(self):
        result = parse_donut_output("")
        assert all(value == "" for value in result.values())

    def test_vietnamese_content(self):
        text = "<s_mcocr><s_store_name>Cua hang Nam Oanh</s_store_name><s_address>Thon Phu Thuy, Xa Phu Thi</s_address></s>"
        result = parse_donut_output(text)
        assert result["store_name"] == "Cua hang Nam Oanh"
        assert result["address"] == "Thon Phu Thuy, Xa Phu Thi"


class TestComputeMetrics:
    def test_perfect_match(self):
        preds = [{"store_name": "ABC", "date": "01/01", "total": "100", "address": "123 St"}]
        golds = [{"store_name": "ABC", "date": "01/01", "total": "100", "address": "123 St"}]
        metrics = compute_metrics(preds, golds)
        assert metrics["overall"]["f1"] == 1.0

    def test_all_wrong(self):
        preds = [{"store_name": "X", "date": "X", "total": "X", "address": "X"}]
        golds = [{"store_name": "A", "date": "B", "total": "C", "address": "D"}]
        metrics = compute_metrics(preds, golds)
        assert metrics["overall"]["f1"] == 0.0

    def test_empty_gold_and_pred(self):
        preds = [{"store_name": "", "date": "", "total": "", "address": ""}]
        golds = [{"store_name": "", "date": "", "total": "", "address": ""}]
        metrics = compute_metrics(preds, golds)
        assert metrics["overall"]["f1"] == 0.0

    def test_partial_match(self):
        preds = [{"store_name": "ABC", "date": "01/01", "total": "WRONG", "address": ""}]
        golds = [{"store_name": "ABC", "date": "01/01", "total": "100", "address": "123 St"}]
        metrics = compute_metrics(preds, golds)
        assert metrics["per_field"]["store_name"]["f1"] == 1.0
        assert metrics["per_field"]["date"]["f1"] == 1.0
        assert metrics["per_field"]["total"]["f1"] == 0.0
        assert metrics["per_field"]["address"]["f1"] == 0.0


class TestNormalizeFieldValue:
    def test_date_normalize(self):
        assert normalize_field_value("date", "01-01-2024") == normalize_field_value("date", "01/01/2024")
        assert normalize_field_value("date", "01.01.2024") == normalize_field_value("date", "01/01/2024")

    def test_total_normalize(self):
        assert normalize_field_value("total", "100,000") == normalize_field_value("total", "100000")
        assert normalize_field_value("total", "100.000 VND") == normalize_field_value("total", "100000")

    def test_address_normalize(self):
        result = normalize_field_value("address", "212 Duong Tran Phu, Quan 1")
        assert result == "212 duong tran phu q 1"

    def test_store_name_passthrough(self):
        result = normalize_field_value("store_name", "  VinCommerce  ")
        assert result == "vincommerce"
