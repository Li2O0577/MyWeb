import unittest

import pandas as pd

from backend.routes._helpers import (
    coerce_column_like,
    coerce_columns_like,
    numeric_prediction_error,
)


class RouteHelperTests(unittest.TestCase):
    def test_numeric_prediction_error_accepts_valid_batch_payload(self):
        err = numeric_prediction_error([[1, 2], [3, 4]], "后端批量预测输入", batch=True)

        self.assertIsNone(err)

    def test_numeric_prediction_error_rejects_bad_payloads(self):
        self.assertIn("非数值内容", numeric_prediction_error(["bad"], "后端预测输入"))
        self.assertIn("空值 NaN", numeric_prediction_error([float("nan")], "后端预测输入"))
        self.assertIn("无穷值 Inf", numeric_prediction_error([float("inf")], "后端预测输入"))
        self.assertIn(
            "二维数组",
            numeric_prediction_error([1, 2, 3], "后端批量预测输入", batch=True),
        )

    def test_numeric_prediction_error_rejects_too_many_batch_rows(self):
        err = numeric_prediction_error(
            [[1, 2], [3, 4]],
            "后端批量预测输入",
            batch=True,
            max_rows=1,
        )

        self.assertIn("单次最多支持 1 行", err)

    def test_column_coercion_matches_dataframe_column_type(self):
        df = pd.DataFrame({1: [10], 2: [20]})

        self.assertEqual(coerce_column_like(df, "1"), 1)
        self.assertEqual(coerce_columns_like(df, ["1", "2"]), [1, 2])


if __name__ == "__main__":
    unittest.main()
