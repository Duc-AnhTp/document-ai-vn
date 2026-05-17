import unittest

from data_preparation.prepare_mcocr import _parse_anno
from data_preparation.normalize import normalize_money_gt
from src.classifier import assign_line_labels
from src.evaluate import compute_exact_match
from src.extractor import extract_fields
from src.postprocess import normalize_money


class CoreRegressionTests(unittest.TestCase):
    def test_mcocr_total_cost_prefers_amount_over_keyword(self):
        fields = _parse_anno(
            "Shop|||Tong tien:|||74,000",
            "SELLER|||TOTAL_COST|||TOTAL_COST",
        )
        self.assertEqual(normalize_money_gt(fields["TOTAL_COST"]), "74000")

    def test_mcocr_address_lines_are_merged(self):
        fields = _parse_anno(
            "Shop|||Line 1|||Line 2",
            "SELLER|||ADDRESS|||ADDRESS",
        )
        self.assertEqual(fields["SELLER_ADDRESS"], "Line 1 Line 2")

    def test_money_normalization_preserves_decimal_when_likely(self):
        self.assertEqual(normalize_money("74,000"), "74000")
        self.assertEqual(normalize_money("9.00"), "9.00")

    def test_eval_does_not_equal_decimal_and_integer_money(self):
        self.assertFalse(compute_exact_match("900", "9.00", field="TOTAL_COST"))

    def test_classifier_money_label_matches_formatted_amount(self):
        labels = assign_line_labels([{"text": "74,000"}], {"TOTAL_COST": "74000"})
        self.assertEqual(labels, ["TOTAL_COST"])

    def test_extractor_total_uses_next_amount_line_after_keyword(self):
        lines = [
            {"text": "Tong tien:", "conf": 0.9, "bbox": [0, 500, 100, 520]},
            {"text": "74,000", "conf": 0.9, "bbox": [110, 500, 190, 520]},
        ]
        fields = extract_fields(lines, image_height=1000)
        self.assertEqual(fields["TOTAL_COST"], "74000")


if __name__ == "__main__":
    unittest.main()
