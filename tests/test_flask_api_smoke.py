import io
import os
import shutil
import sys
import tempfile
import types
import unittest


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


if __name__ == "__main__":
    unittest.main()
