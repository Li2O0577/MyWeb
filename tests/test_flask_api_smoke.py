import io
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


MODEL_TYPES = ("decision_tree", "clustering", "regression", "classification", "diy_mlp")


def _install_torch_stub():
    """Let Flask import all route modules even when torch is not installed."""
    if "torch" in sys.modules:
        return

    torch_mod = types.ModuleType("torch")
    nn_mod = types.ModuleType("torch.nn")
    optim_mod = types.ModuleType("torch.optim")
    utils_mod = types.ModuleType("torch.utils")
    data_mod = types.ModuleType("torch.utils.data")

    class Module:
        def __init__(self, *args, **kwargs):
            pass

        def to(self, *_args, **_kwargs):
            return self

        def eval(self):
            return self

        def train(self):
            return self

        def state_dict(self):
            return {}

        def load_state_dict(self, *_args, **_kwargs):
            return None

    class Tensor:
        pass

    class _Layer(Module):
        pass

    nn_mod.Module = Module
    nn_mod.Sequential = lambda *args, **kwargs: list(args)
    nn_mod.Linear = _Layer
    nn_mod.ReLU = _Layer
    nn_mod.Dropout = _Layer
    nn_mod.BatchNorm1d = _Layer
    nn_mod.LeakyReLU = _Layer
    nn_mod.GELU = _Layer
    nn_mod.Tanh = _Layer
    nn_mod.Sigmoid = _Layer
    nn_mod.ELU = _Layer
    nn_mod.SELU = _Layer
    nn_mod.Identity = _Layer
    nn_mod.MSELoss = _Layer
    nn_mod.CrossEntropyLoss = _Layer
    nn_mod.BCEWithLogitsLoss = _Layer

    optim_mod.Adam = _Layer
    optim_mod.AdamW = _Layer
    optim_mod.SGD = _Layer
    optim_mod.RMSprop = _Layer

    data_mod.TensorDataset = _Layer
    data_mod.DataLoader = _Layer

    torch_mod.nn = nn_mod
    torch_mod.optim = optim_mod
    torch_mod.utils = utils_mod
    torch_mod.device = lambda value: value
    torch_mod.Tensor = Tensor
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_mod.save = lambda *args, **kwargs: None
    torch_mod.load = lambda *args, **kwargs: {}
    torch_mod.tensor = lambda value, **kwargs: value
    torch_mod.float32 = "float32"
    torch_mod.long = "long"
    torch_mod.no_grad = lambda: types.SimpleNamespace(__enter__=lambda self: None, __exit__=lambda self, *exc: False)

    sys.modules["torch"] = torch_mod
    sys.modules["torch.nn"] = nn_mod
    sys.modules["torch.optim"] = optim_mod
    sys.modules["torch.utils"] = utils_mod
    sys.modules["torch.utils.data"] = data_mod


class FlaskApiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_torch_stub()
        backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)

        import app as backend_app
        import models.registry as registry
        import session_store

        cls.backend_app = backend_app
        cls.registry = registry
        cls.session_store = session_store

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="myweb1_api_test_")
        self.sessions_dir = os.path.join(self.tmpdir, "sessions")
        self.models_dir = os.path.join(self.tmpdir, "models")
        os.makedirs(self.sessions_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        self.old_sessions_dir = self.session_store.SESSIONS_DIR
        self.old_models_dir = self.registry.MODELS_DIR
        self.old_registry_path = self.registry.REGISTRY_PATH

        self.session_store.SESSIONS_DIR = self.sessions_dir
        self.registry.MODELS_DIR = self.models_dir
        self.registry.REGISTRY_PATH = os.path.join(self.models_dir, "registry.json")
        self.session_store._sessions.clear()

        self.client = self.backend_app.app.test_client()

    def tearDown(self):
        self.session_store._sessions.clear()
        self.session_store.SESSIONS_DIR = self.old_sessions_dir
        self.registry.MODELS_DIR = self.old_models_dir
        self.registry.REGISTRY_PATH = self.old_registry_path
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _upload_csv(self, csv_text, filename="data.csv"):
        return self.client.post(
            "/api/data/upload",
            data={"file": (io.BytesIO(csv_text.encode("utf-8")), filename)},
            content_type="multipart/form-data",
        )

    def _sse_events(self, response):
        events = []
        for block in response.get_data(as_text=True).split("\n\n"):
            if not block.startswith("data: "):
                continue
            events.append(json.loads(block[6:]))
        return events

    def test_upload_train_predict_decision_tree_regression(self):
        rows = ["x1,x2,y"]
        for i in range(20):
            rows.append(f"{i},{i * 2},{i * 3}")

        upload_resp = self._upload_csv("\n".join(rows))
        self.assertEqual(upload_resp.status_code, 200, upload_resp.get_data(as_text=True))
        sid = upload_resp.get_json()["session_id"]

        train_resp = self.client.post("/api/decision_tree/train", json={
            "session_id": sid,
            "target_col": "y",
            "feature_cols": ["x1", "x2"],
            "task_type": "regression",
            "criterion": "squared_error",
            "max_depth": 3,
        })
        self.assertEqual(train_resp.status_code, 200, train_resp.get_data(as_text=True))
        train_data = train_resp.get_json()
        self.assertIn("version_id", train_data)
        self.assertIn("r2", train_data)
        self.assertIn("tree_rules", train_data)
        self.assertIn("tree_nodes", train_data)

        status_resp = self.client.get("/api/decision_tree/status")
        self.assertEqual(status_resp.status_code, 200, status_resp.get_data(as_text=True))
        status_data = status_resp.get_json()
        self.assertTrue(status_data["has_model"])
        self.assertEqual(status_data["version_id"], train_data["version_id"])
        self.assertIn("tree_rules", status_data)
        self.assertIn("tree_nodes", status_data)

        predict_resp = self.client.post("/api/decision_tree/predict", json={
            "input_dict": {"x1": [5], "x2": [10]},
            "task_type": "regression",
            "version_id": train_data["version_id"],
        })
        self.assertEqual(predict_resp.status_code, 200, predict_resp.get_data(as_text=True))
        self.assertIn("pred_value", predict_resp.get_json())

    def test_train_failure_returns_friendly_validation_error(self):
        rows = ["x,label,y"]
        for i in range(10):
            rows.append(f"{i},class_{i},{i}")

        upload_resp = self._upload_csv("\n".join(rows), filename="bad.csv")
        self.assertEqual(upload_resp.status_code, 200, upload_resp.get_data(as_text=True))
        sid = upload_resp.get_json()["session_id"]

        train_resp = self.client.post("/api/classification/train", json={
            "session_id": sid,
            "target_col": "label",
            "feature_cols": ["x"],
            "learning_rate": 0.001,
            "epochs": 20,
            "batch_size": 4,
            "device": "cpu",
        })

        self.assertEqual(train_resp.status_code, 400)
        payload = train_resp.get_json()
        self.assertEqual(payload["error"]["code"], "INPUT_VALIDATION_FAILED")
        self.assertIn("过于接近样本数", payload["error"]["message"])

    def test_predict_without_model_returns_model_not_found(self):
        predict_resp = self.client.post("/api/regression/predict", json={
            "features": [1.0, 2.0],
            "device": "cpu",
        })

        self.assertEqual(predict_resp.status_code, 404)
        payload = predict_resp.get_json()
        self.assertEqual(payload["error"]["code"], "MODEL_NOT_FOUND")

    def test_predict_invalid_payload_returns_prediction_failed(self):
        predict_resp = self.client.post("/api/regression/predict", json={
            "features": ["bad", 2.0],
            "device": "cpu",
        })

        self.assertEqual(predict_resp.status_code, 400)
        payload = predict_resp.get_json()
        self.assertEqual(payload["error"]["code"], "PREDICTION_FAILED")
        self.assertIn("非数值内容", payload["error"]["message"])

    def test_data_profile_and_process_return_frontend_profile(self):
        upload_resp = self._upload_csv("x,label,y\n1,a,10\n2,a,20\n, b,30\n2,a,20\n", filename="profile.csv")
        self.assertEqual(upload_resp.status_code, 200, upload_resp.get_data(as_text=True))
        upload_data = upload_resp.get_json()
        sid = upload_data["session_id"]

        self.assertIn("column_profiles", upload_data)
        self.assertIn("missing_total", upload_data)
        self.assertIn("preview", upload_data)
        self.assertEqual(upload_data["n_cols"], 3)
        x_profile = next(col for col in upload_data["column_profiles"] if col["name"] == "x")
        self.assertEqual(x_profile["stats"]["mean"], 1.666667)
        self.assertEqual(x_profile["stats"]["median"], 2.0)
        self.assertEqual(x_profile["stats"]["min"], 1.0)
        self.assertEqual(x_profile["stats"]["max"], 2.0)

        profile_resp = self.client.get(f"/api/data/{sid}/profile")
        self.assertEqual(profile_resp.status_code, 200, profile_resp.get_data(as_text=True))
        profile_data = profile_resp.get_json()
        self.assertEqual(profile_data["session_id"], sid)
        self.assertIn("x", profile_data["columns"])
        self.assertTrue(any(col["name"] == "x" for col in profile_data["column_profiles"]))

        process_resp = self.client.post(f"/api/data/{sid}/process", json={
            "operations": [
                {"op": "drop_na"},
                {"op": "drop_duplicates"},
                {"op": "drop_cols", "cols": ["label"]},
            ]
        })
        self.assertEqual(process_resp.status_code, 200, process_resp.get_data(as_text=True))
        processed = process_resp.get_json()
        self.assertEqual(processed["session_id"], sid)
        self.assertNotIn("label", processed["columns"])
        self.assertIn("column_profiles", processed)
        self.assertIn("preview", processed)

    def test_data_processing_supports_full_cleaning_operations(self):
        upload_resp = self._upload_csv(
            "x,y,cat\n1,10,a\n2,,b\n3,30,a\n4,40,c\n",
            filename="legacy_processing.csv",
        )
        self.assertEqual(upload_resp.status_code, 200, upload_resp.get_data(as_text=True))
        sid = upload_resp.get_json()["session_id"]

        process_resp = self.client.post(f"/api/data/{sid}/process", json={
            "operations": [
                {"op": "rename_col", "old": "x", "new": "x_num"},
                {"op": "cast_type", "col": "x_num", "dtype": "float"},
                {"op": "fill_na", "cols": ["y"], "method": "mean"},
                {"op": "scale", "cols": ["x_num", "y"], "method": "standard"},
                {"op": "label_encode", "cols": ["cat"]},
                {"op": "custom_formula", "new_col": "total", "expr": "x_num + y"},
                {"op": "pca", "cols": ["x_num", "y"], "n_components": 1, "prefix": "PCA"},
            ]
        })
        self.assertEqual(process_resp.status_code, 200, process_resp.get_data(as_text=True))
        processed = process_resp.get_json()

        self.assertIn("x_num", processed["columns"])
        self.assertNotIn("x", processed["columns"])
        self.assertIn("total", processed["columns"])
        self.assertIn("PCA_1", processed["columns"])
        self.assertIn("operations_applied", processed)
        self.assertEqual(processed["missing_total"], 0)
        self.assertTrue(any(col["name"] == "cat" and col["kind"] == "numeric" for col in processed["column_profiles"]))

    def test_data_processing_rejects_unsafe_formula(self):
        upload_resp = self._upload_csv("x,y\n1,2\n3,4\n", filename="formula.csv")
        self.assertEqual(upload_resp.status_code, 200, upload_resp.get_data(as_text=True))
        sid = upload_resp.get_json()["session_id"]

        process_resp = self.client.post(f"/api/data/{sid}/process", json={
            "operations": [
                {"op": "custom_formula", "new_col": "bad", "expr": "__import__('os').system('whoami')"},
            ]
        })
        self.assertEqual(process_resp.status_code, 400)
        payload = process_resp.get_json()
        self.assertEqual(payload["error"]["code"], "PROCESSING_FAILED")

    def test_data_processing_supports_outlier_operations(self):
        rows = ["x,y"]
        for i in range(1, 9):
            rows.append(f"{i},{i}")
        rows.append("100,9")

        upload_resp = self._upload_csv("\n".join(rows), filename="outliers.csv")
        self.assertEqual(upload_resp.status_code, 200, upload_resp.get_data(as_text=True))
        sid = upload_resp.get_json()["session_id"]

        winsor_resp = self.client.post(f"/api/data/{sid}/process", json={
            "operations": [
                {"op": "winsorize_outliers", "cols": ["x"], "coefficient": 1.5},
            ]
        })
        self.assertEqual(winsor_resp.status_code, 200, winsor_resp.get_data(as_text=True))
        winsor_data = winsor_resp.get_json()
        x_profile = next(col for col in winsor_data["column_profiles"] if col["name"] == "x")
        self.assertLess(x_profile["stats"]["max"], 100)

        upload_resp = self._upload_csv("\n".join(rows), filename="outliers_drop.csv")
        self.assertEqual(upload_resp.status_code, 200, upload_resp.get_data(as_text=True))
        sid = upload_resp.get_json()["session_id"]

        drop_resp = self.client.post(f"/api/data/{sid}/process", json={
            "operations": [
                {"op": "drop_outliers", "cols": ["x"], "coefficient": 1.5},
            ]
        })
        self.assertEqual(drop_resp.status_code, 200, drop_resp.get_data(as_text=True))
        self.assertEqual(drop_resp.get_json()["n_rows"], 8)

    def test_data_visualization_endpoint_returns_structured_chart_data(self):
        upload_resp = self._upload_csv(
            "x,y,z,cat\n1,10,5,a\n2,20,6,a\n3,15,7,b\n4,30,8,b\n5,25,9,c\n",
            filename="visual.csv",
        )
        self.assertEqual(upload_resp.status_code, 200, upload_resp.get_data(as_text=True))
        sid = upload_resp.get_json()["session_id"]

        scatter_resp = self.client.post(f"/api/data/{sid}/visualize", json={
            "chart_type": "scatter",
            "x_col": "x",
            "y_col": "y",
            "color_col": "cat",
        })
        self.assertEqual(scatter_resp.status_code, 200, scatter_resp.get_data(as_text=True))
        scatter = scatter_resp.get_json()
        self.assertEqual(scatter["chart_type"], "scatter")
        self.assertIn("points", scatter["data"])
        self.assertEqual(len(scatter["data"]["points"]), 5)

        bar_resp = self.client.post(f"/api/data/{sid}/visualize", json={
            "chart_type": "bar",
            "x_col": "cat",
            "y_col": "y",
            "agg": "mean",
        })
        self.assertEqual(bar_resp.status_code, 200, bar_resp.get_data(as_text=True))
        bar = bar_resp.get_json()
        self.assertIn("bars", bar["data"])
        self.assertEqual(len(bar["data"]["bars"]), 3)

        corr_resp = self.client.post(f"/api/data/{sid}/visualize", json={
            "chart_type": "corr_heatmap",
            "cols": ["x", "y", "z"],
        })
        self.assertEqual(corr_resp.status_code, 200, corr_resp.get_data(as_text=True))
        corr = corr_resp.get_json()
        self.assertEqual(corr["chart_type"], "corr_heatmap")
        self.assertEqual(len(corr["data"]["cells"]), 9)

    def test_data_visualization_endpoint_rejects_invalid_chart_config(self):
        upload_resp = self._upload_csv("x,y\n1,2\n3,4\n", filename="visual_bad.csv")
        self.assertEqual(upload_resp.status_code, 200, upload_resp.get_data(as_text=True))
        sid = upload_resp.get_json()["session_id"]

        resp = self.client.post(f"/api/data/{sid}/visualize", json={
            "chart_type": "density_heatmap",
            "x_col": "x",
            "y_col": "x",
        })
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.get_json()["error"]["code"], "VISUALIZATION_FAILED")

    def test_phase4_model_version_endpoints_are_available(self):
        for model_type in MODEL_TYPES:
            with self.subTest(model_type=model_type):
                versions_resp = self.client.get(f"/api/{model_type}/versions")
                self.assertEqual(versions_resp.status_code, 200, versions_resp.get_data(as_text=True))
                payload = versions_resp.get_json()
                self.assertEqual(payload["versions"], [])
                self.assertIsNone(payload["active"])

                status_resp = self.client.get(f"/api/{model_type}/status")
                self.assertEqual(status_resp.status_code, 200, status_resp.get_data(as_text=True))
                self.assertFalse(status_resp.get_json()["has_model"])

    def test_phase4_training_routes_validate_required_payloads(self):
        for model_type in MODEL_TYPES:
            with self.subTest(model_type=model_type):
                train_resp = self.client.post(f"/api/{model_type}/train", json={})
                self.assertEqual(train_resp.status_code, 400, train_resp.get_data(as_text=True))
                payload = train_resp.get_json()
                self.assertEqual(payload["error"]["code"], "MISSING_FIELD")

    def test_regression_status_exposes_training_curves_for_react(self):
        self.registry.register_version("regression", "reg_test_v1", {
            "dataset_name": "unit.csv",
            "session_id": "sid-test",
            "features": ["x1", "x2"],
            "target": "y",
            "metrics": {"r2": 0.8, "mae": 1.2, "rmse": 1.8},
            "params": {"learning_rate": 0.001, "epochs": 40, "batch_size": 8},
            "train_losses": [3.0, 2.0, 1.0],
            "val_losses": [3.2, 2.4, 1.4],
        }, {"model": "model.pth", "scaler": "scaler.npz", "config": "config.json"})

        status_resp = self.client.get("/api/regression/status")
        self.assertEqual(status_resp.status_code, 200, status_resp.get_data(as_text=True))
        status_data = status_resp.get_json()
        self.assertTrue(status_data["has_model"])
        self.assertEqual(status_data["version_id"], "reg_test_v1")
        self.assertEqual(status_data["train_losses"], [3.0, 2.0, 1.0])
        self.assertEqual(status_data["val_losses"], [3.2, 2.4, 1.4])
        self.assertEqual(status_data["session_id"], "sid-test")

    def test_classification_status_exposes_result_panels_for_react(self):
        self.registry.register_version("classification", "cls_test_v1", {
            "dataset_name": "unit.csv",
            "session_id": "sid-cls",
            "features": ["x1", "x2"],
            "target": "label",
            "metrics": {"acc": 0.75},
            "params": {"learning_rate": 0.001, "epochs": 40, "batch_size": 8, "n_classes": 2},
            "cm": [[3, 1], [1, 3]],
            "label_names": ["no", "yes"],
            "n_classes": 2,
            "reverse_label_map": {"0": "no", "1": "yes"},
            "train_losses": [0.8, 0.6],
            "val_losses": [0.9, 0.7],
        }, {"model": "model.pth", "scaler": "scaler.npz", "config": "config.json"})

        status_resp = self.client.get("/api/classification/status")
        self.assertEqual(status_resp.status_code, 200, status_resp.get_data(as_text=True))
        status_data = status_resp.get_json()
        self.assertTrue(status_data["has_model"])
        self.assertEqual(status_data["version_id"], "cls_test_v1")
        self.assertEqual(status_data["cm"], [[3, 1], [1, 3]])
        self.assertEqual(status_data["label_names"], ["no", "yes"])
        self.assertEqual(status_data["reverse_label_map"], {"0": "no", "1": "yes"})
        self.assertEqual(status_data["train_losses"], [0.8, 0.6])

    def test_diy_mlp_status_exposes_architecture_and_curves_for_react(self):
        layers = [
            {"neurons": 16, "activation": "ReLU", "bn": True, "dropout": 0.1},
            {"neurons": 8, "activation": "GELU", "bn": False, "dropout": 0.2},
        ]
        self.registry.register_version("diy_mlp", "diy_test_v1", {
            "dataset_name": "unit.csv",
            "session_id": "sid-diy",
            "features": ["x1", "x2"],
            "target": "label",
            "metrics": {"acc": 0.7},
            "params": {
                "task": "classification",
                "layers": layers,
                "learning_rate": 0.005,
                "optimizer": "AdamW",
                "epochs": 80,
                "batch_size": 16,
                "val_split": 0.2,
                "patience": 15,
                "n_classes": 2,
            },
            "label_names": ["low", "high"],
            "n_classes": 2,
            "reverse_label_map": {"0": "low", "1": "high"},
            "train_losses": [0.9, 0.5],
            "val_losses": [1.0, 0.6],
        }, {"model": "model.pth", "scaler": "scaler.npz", "config": "config.json"})

        status_resp = self.client.get("/api/diy_mlp/status")
        self.assertEqual(status_resp.status_code, 200, status_resp.get_data(as_text=True))
        status_data = status_resp.get_json()
        self.assertTrue(status_data["has_model"])
        self.assertEqual(status_data["version_id"], "diy_test_v1")
        self.assertEqual(status_data["params"]["task"], "classification")
        self.assertEqual(status_data["params"]["layers"], layers)
        self.assertEqual(status_data["train_losses"], [0.9, 0.5])
        self.assertEqual(status_data["val_losses"], [1.0, 0.6])
        self.assertEqual(status_data["label_names"], ["low", "high"])
        self.assertEqual(status_data["reverse_label_map"], {"0": "low", "1": "high"})

    def test_upload_train_predict_clustering_kmeans(self):
        rows = ["x1,x2"]
        for i in range(12):
            rows.append(f"{i},{i + 1}")
        for i in range(12):
            rows.append(f"{100 + i},{101 + i}")

        upload_resp = self._upload_csv("\n".join(rows), filename="cluster.csv")
        self.assertEqual(upload_resp.status_code, 200, upload_resp.get_data(as_text=True))
        sid = upload_resp.get_json()["session_id"]

        train_resp = self.client.post("/api/clustering/train", json={
            "session_id": sid,
            "feature_cols": ["x1", "x2"],
            "algorithm": "kmeans",
            "params": {"n_clusters": 2},
        })
        self.assertEqual(train_resp.status_code, 200, train_resp.get_data(as_text=True))
        train_data = train_resp.get_json()
        self.assertIn("version_id", train_data)
        self.assertEqual(train_data["n_found"], 2)
        self.assertIn("pca", train_data)
        self.assertIn("cluster_counts", train_data)

        versions_resp = self.client.get("/api/clustering/versions")
        self.assertEqual(versions_resp.status_code, 200, versions_resp.get_data(as_text=True))
        versions_data = versions_resp.get_json()
        self.assertEqual(versions_data["active"], train_data["version_id"])
        self.assertEqual(len(versions_data["versions"]), 1)

        status_resp = self.client.get("/api/clustering/status")
        self.assertEqual(status_resp.status_code, 200, status_resp.get_data(as_text=True))
        status_data = status_resp.get_json()
        self.assertTrue(status_data["has_model"])
        self.assertEqual(status_data["version_id"], train_data["version_id"])
        self.assertIn("cluster_counts", status_data)
        self.assertIn("pca", status_data)
        self.assertEqual(status_data["n_found"], 2)

        predict_resp = self.client.post("/api/clustering/predict", json={
            "features": [3, 4],
            "version_id": train_data["version_id"],
        })
        self.assertEqual(predict_resp.status_code, 200, predict_resp.get_data(as_text=True))
        self.assertIn("cluster", predict_resp.get_json())

    def test_llm_direct_chat_streams_chunks_and_done(self):
        def fake_stream_chat(_api_base, _api_key, _model, _messages):
            yield "hello", None
            yield " world", None

        with patch("routes.llm_routes.stream_chat", fake_stream_chat):
            resp = self.client.post("/api/llm/chat", json={
                "api_base": "https://api.openai.com/v1",
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
            })

        self.assertEqual(resp.status_code, 200, resp.get_data(as_text=True))
        events = self._sse_events(resp)
        self.assertEqual(events[0]["chunk"], "hello")
        self.assertEqual(events[1]["full"], "hello world")
        self.assertTrue(events[-1]["done"])

    def test_llm_agent_streams_tool_image_and_done(self):
        def fake_agent_chat(_api_base, _api_key, _model, _messages, _session_id):
            yield {"status": "thinking", "message": "checking"}
            yield {"status": "tool_call", "tool": "generate_chart", "args": {"chart_type": "scatter"}}
            yield {"status": "image", "base64": "aGVsbG8=", "title": "Chart", "alt": "chart"}
            yield {"status": "tool_result", "tool": "generate_chart", "result": "created"}
            yield {"chunk": "done"}
            yield {"done": True, "tools_used": 1}

        with patch("routes.llm_routes.agent_chat", fake_agent_chat):
            resp = self.client.post("/api/llm/agent", json={
                "session_id": "sid",
                "api_base": "https://api.openai.com/v1",
                "model": "test-model",
                "messages": [{"role": "user", "content": "plot"}],
            })

        self.assertEqual(resp.status_code, 200, resp.get_data(as_text=True))
        events = self._sse_events(resp)
        self.assertEqual(events[0]["status"], "thinking")
        self.assertEqual(events[1]["tool"], "generate_chart")
        self.assertEqual(events[2]["status"], "image")
        self.assertEqual(events[3]["result"], "created")
        self.assertEqual(events[4]["chunk"], "done")
        self.assertTrue(events[-1]["done"])

    def test_llm_routes_validate_required_payloads(self):
        direct_resp = self.client.post("/api/llm/chat", json={})
        agent_resp = self.client.post("/api/llm/agent", json={})

        self.assertEqual(direct_resp.status_code, 400)
        self.assertEqual(agent_resp.status_code, 400)
        self.assertEqual(direct_resp.get_json()["error"]["code"], "MISSING_FIELD")
        self.assertEqual(agent_resp.get_json()["error"]["code"], "MISSING_FIELD")


if __name__ == "__main__":
    unittest.main()
