import unittest

import numpy as np
import pandas as pd

from backend.services.training_validation import (
    validate_classification_training,
    validate_regression_training,
)


class TrainingValidationTests(unittest.TestCase):
    def test_rejects_non_numeric_features_for_mlp_style_models(self):
        df = pd.DataFrame({
            "feature": ["low", "mid", "high", "low", "mid", "high", "low", "mid", "high", "low"],
            "target": list(range(10)),
        })

        err = validate_regression_training(df, "target", ["feature"], batch_size=4)

        self.assertIsNotNone(err)
        self.assertIn("特征列全部为数值型", err["error"])
        self.assertIn("feature", err["error"])

    def test_rejects_invalid_regression_target(self):
        df = pd.DataFrame({
            "feature": range(10),
            "target": ["a", "b"] * 5,
        })

        err = validate_regression_training(df, "target", ["feature"], batch_size=4)

        self.assertIsNotNone(err)
        self.assertIn("回归目标列", err["error"])
        self.assertIn("必须是数值型", err["error"])

    def test_rejects_high_missing_ratio(self):
        df = pd.DataFrame({
            "feature": [1, 2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            "target": range(10),
        })

        err = validate_regression_training(df, "target", ["feature"], batch_size=2)

        self.assertIsNotNone(err)
        self.assertIn("空值比例过高", err["error"])
        self.assertIn("删除空值后仅剩", err["error"])

    def test_rejects_too_many_classes(self):
        df = pd.DataFrame({
            "feature": range(20),
            "target": [f"id_{i}" for i in range(20)],
        })

        err = validate_classification_training(df, "target", ["feature"], batch_size=4)

        self.assertIsNotNone(err)
        self.assertIn("过于接近样本数", err["error"])
        self.assertIn("每个类别至少需要 2 个样本", err["error"])

    def test_rejects_insufficient_clean_rows(self):
        df = pd.DataFrame({
            "feature": [1, 2, 3, 4, 5],
            "target": [1, 2, 3, 4, 5],
        })

        err = validate_regression_training(df, "target", ["feature"], batch_size=2)

        self.assertIsNotNone(err)
        self.assertIn("至少需要 10 行", err["error"])

    def test_rejects_batch_size_larger_than_clean_rows(self):
        df = pd.DataFrame({
            "feature": range(10),
            "target": range(10),
        })

        err = validate_regression_training(df, "target", ["feature"], batch_size=32)

        self.assertIsNotNone(err)
        self.assertIn("批次大小为 32", err["error"])


if __name__ == "__main__":
    unittest.main()
