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


if __name__ == "__main__":
    unittest.main()
