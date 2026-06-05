import unittest
from unittest.mock import patch

import pandas as pd

from backend.services import llm_service


class LlmAgentGuardTests(unittest.TestCase):
    def test_prepare_messages_trims_before_validation(self):
        messages = [{"role": "user", "content": f"m{i}"} for i in range(40)]

        trimmed = llm_service._prepare_messages(messages)
        ok, err = llm_service._validate_messages(trimmed)

        self.assertTrue(ok, err)
        self.assertEqual(len(trimmed), llm_service._MAX_AGENT_MESSAGES)
        self.assertEqual(trimmed[0]["content"], "m10")

    def test_sanitize_training_epochs_is_clamped(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})

        args = llm_service._sanitize_tool_args(
            "run_regression",
            {"target_column": "y", "epochs": 9999, "learning_rate": 999},
            df,
        )

        self.assertEqual(args["epochs"], llm_service._MAX_AGENT_TRAIN_EPOCHS)
        self.assertEqual(args["learning_rate"], 0.05)

    def test_code_interpreter_rejects_too_many_columns_before_writing_csv(self):
        df = pd.DataFrame({f"c{i}": [i] for i in range(llm_service._MAX_CODE_INTERPRETER_COLS + 1)})

        result = llm_service._execute_code_in_subprocess("print(df.shape)", df)

        self.assertIn("最多支持", result["text"])
        self.assertFalse(result["images"])

    def test_private_api_base_is_blocked_by_default(self):
        with patch.dict("os.environ", {"LLM_ALLOW_LOCAL_API_BASE": "0"}, clear=False):
            ok, err = llm_service._validate_api_base("http://127.0.0.1:11434/v1")

        self.assertFalse(ok)
        self.assertIn("不允许的 API 主机", err)

    def test_private_api_base_can_be_enabled_for_local_development(self):
        with patch.dict("os.environ", {"LLM_ALLOW_LOCAL_API_BASE": "1"}, clear=False):
            ok, err = llm_service._validate_api_base("http://127.0.0.1:11434/v1")

        self.assertTrue(ok, err)

    def test_chart_title_falls_back_to_english_when_title_is_chinese(self):
        title = llm_service._chart_title("销售额趋势", "Line Chart")

        self.assertEqual(title, "Line Chart")

    def test_generate_chart_uses_english_image_title(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8]})

        result = llm_service._generate_chart("scatter", "x", "y", "", "销售额趋势", df)

        self.assertTrue(result["images"])
        self.assertTrue(result["images"][0]["title"].isascii())
        self.assertIn("Scatter Plot", result["images"][0]["title"])

    def test_chart_labels_fall_back_to_english_for_chinese_values(self):
        labels = llm_service._category_labels(["华东", "华南"], "Group")

        self.assertEqual(labels, ["Group 1", "Group 2"])

    def test_generate_chart_with_chinese_columns_still_uses_english_title(self):
        df = pd.DataFrame({"价格": [1, 2, 3, 4], "销量": [2, 4, 6, 8], "区域": ["华东", "华南", "华东", "华北"]})

        result = llm_service._generate_chart("scatter", "价格", "销量", "区域", "销售额趋势", df)

        self.assertTrue(result["images"])
        self.assertEqual(result["images"][0]["title"], "Scatter Plot: Y vs X")


if __name__ == "__main__":
    unittest.main()
