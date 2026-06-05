import io
import unittest

from pages._mlp_common import prepare_batch_prediction
from backend.routes._helpers import numeric_prediction_error


class FrontendHelperTests(unittest.TestCase):
    def test_prepare_batch_prediction_success(self):
        csv_file = io.StringIO("x1,x2\n1,2\n3,4\n")

        df, values, err = prepare_batch_prediction(csv_file, ["x1", "x2"], "测试批量预测")

        self.assertIsNone(err)
        self.assertEqual(len(df), 2)
        self.assertEqual(values.tolist(), [[1, 2], [3, 4]])

    def test_prepare_batch_prediction_missing_feature(self):
        csv_file = io.StringIO("x1,x3\n1,2\n")

        _df, values, err = prepare_batch_prediction(csv_file, ["x1", "x2"], "测试批量预测")

        self.assertIsNone(values)
        self.assertIn("缺少模型需要的特征列", err)

    def test_prepare_batch_prediction_rejects_non_numeric(self):
        csv_file = io.StringIO("x1,x2\n1,bad\n")

        _df, values, err = prepare_batch_prediction(csv_file, ["x1", "x2"], "测试批量预测")

        self.assertIsNone(values)
        self.assertIn("非数值内容", err)

    def test_prepare_batch_prediction_rejects_too_many_rows(self):
        csv_file = io.StringIO("x1,x2\n1,2\n3,4\n")

        _df, values, err = prepare_batch_prediction(
            csv_file,
            ["x1", "x2"],
            "测试批量预测",
            max_rows=1,
        )

        self.assertIsNone(values)
        self.assertIn("单次最多支持 1 行", err)

    def test_backend_numeric_prediction_error_rejects_bad_payloads(self):
        self.assertIn("非数值内容", numeric_prediction_error(["bad"], "后端预测输入"))
        self.assertIn("空值 NaN", numeric_prediction_error([float("nan")], "后端预测输入"))
        self.assertIn("无穷值 Inf", numeric_prediction_error([float("inf")], "后端预测输入"))
        self.assertIn(
            "二维数组",
            numeric_prediction_error([1, 2, 3], "后端批量预测输入", batch=True),
        )


if __name__ == "__main__":
    unittest.main()
