import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

import pandas as pd


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
        import auth_store
        import database
        import models.registry as registry
        import session_store
        import task_store

        cls.backend_app = backend_app
        cls.auth_store = auth_store
        cls.database = database
        cls.registry = registry
        cls.session_store = session_store
        cls.task_store = task_store

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="myweb1_api_test_")
        self.sessions_dir = os.path.join(self.tmpdir, "sessions")
        self.models_dir = os.path.join(self.tmpdir, "models")
        self.auth_dir = os.path.join(self.tmpdir, "auth")
        self.runtime_dir = os.path.join(self.tmpdir, "runtime")
        os.makedirs(self.sessions_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.auth_dir, exist_ok=True)

        self.old_auth_dir = self.auth_store.AUTH_DIR
        self.old_users_path = self.auth_store.USERS_PATH
        self.old_db_path = self.database.DB_PATH
        self.old_sessions_dir = self.session_store.SESSIONS_DIR
        self.old_models_dir = self.registry.MODELS_DIR
        self.old_registry_path = self.registry.REGISTRY_PATH

        self.auth_store.AUTH_DIR = self.auth_dir
        self.auth_store.USERS_PATH = os.path.join(self.auth_dir, "users.json")
        self.database.DB_PATH = os.path.join(self.runtime_dir, "myweb1.db")
        self.session_store.SESSIONS_DIR = self.sessions_dir
        self.registry.MODELS_DIR = self.models_dir
        self.registry.REGISTRY_PATH = os.path.join(self.models_dir, "registry.json")
        self.auth_store.clear_auth_state()
        self.session_store._sessions.clear()
        self.task_store.clear_tasks()

        self.client = self.backend_app.app.test_client()
        user, err = self.auth_store.create_user("tester", "password123")
        self.assertIsNone(err)
        self.user = user
        token = self.auth_store.create_session(user)
        self.client.set_cookie(self.auth_store.COOKIE_NAME, token)

    def tearDown(self):
        self.auth_store.clear_auth_state()
        self.session_store._sessions.clear()
        self.task_store.clear_tasks()
        self.auth_store.AUTH_DIR = self.old_auth_dir
        self.auth_store.USERS_PATH = self.old_users_path
        self.database.DB_PATH = self.old_db_path
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

    def _wait_for_task(self, task_id, timeout=5.0):
        deadline = time.time() + timeout
        payload = None
        while time.time() < deadline:
            resp = self.client.get(f"/api/tasks/{task_id}")
            self.assertEqual(resp.status_code, 200, resp.get_data(as_text=True))
            payload = resp.get_json()
            if payload["status"] in {"succeeded", "failed", "cancelled"}:
                return payload
            time.sleep(0.05)
        self.fail(f"Task {task_id} did not finish in {timeout} seconds; last={payload}")

    def _make_authed_client(self, username, password="password123"):
        client = self.backend_app.app.test_client()
        user, err = self.auth_store.create_user(username, password)
        self.assertIsNone(err)
        token = self.auth_store.create_session(user)
        client.set_cookie(self.auth_store.COOKIE_NAME, token)
        return client, user

    def test_auth_required_and_sessions_are_user_scoped(self):
        anon = self.backend_app.app.test_client()
        health_resp = anon.get("/api/health")
        self.assertEqual(health_resp.status_code, 401)

        upload_resp = self._upload_csv("x,y\n1,2\n3,4\n", filename="private.csv")
        self.assertEqual(upload_resp.status_code, 200, upload_resp.get_data(as_text=True))
        sid = upload_resp.get_json()["session_id"]

        other_client, _other_user = self._make_authed_client("other_user")
        profile_resp = other_client.get(f"/api/data/{sid}/profile")
        self.assertEqual(profile_resp.status_code, 404)
        self.assertEqual(profile_resp.get_json()["error"]["code"], "SESSION_EXPIRED")

        own_profile_resp = self.client.get(f"/api/data/{sid}/profile")
        self.assertEqual(own_profile_resp.status_code, 200, own_profile_resp.get_data(as_text=True))
        self.assertEqual(own_profile_resp.get_json()["session_id"], sid)

        health_resp = self.client.get("/api/health")
        self.assertEqual(health_resp.status_code, 200, health_resp.get_data(as_text=True))
        recent = health_resp.get_json()["recent_sessions"][0]
        self.assertEqual(recent["n_rows"], 2)
        self.assertEqual(recent["n_cols"], 2)
        self.assertEqual(health_resp.get_json()["resource_usage"]["active_sessions"], 1)
        self.assertGreaterEqual(health_resp.get_json()["resource_limits"]["max_sessions_per_user"], 1)

        forbidden_delete = other_client.delete(f"/api/data/{sid}")
        self.assertEqual(forbidden_delete.status_code, 404)
        delete_resp = self.client.delete(f"/api/data/{sid}")
        self.assertEqual(delete_resp.status_code, 200, delete_resp.get_data(as_text=True))
        self.assertEqual(delete_resp.get_json()["deleted"], sid)
        self.assertEqual(self.client.get(f"/api/data/{sid}/profile").status_code, 404)

    def test_data_rows_endpoint_paginates_all_rows_and_columns(self):
        rows = ["x,y,label"] + [f"{index},{index * 2},row-{index}" for index in range(205)]
        upload_resp = self._upload_csv("\n".join(rows), filename="paged.csv")
        self.assertEqual(upload_resp.status_code, 200, upload_resp.get_data(as_text=True))
        sid = upload_resp.get_json()["session_id"]

        page_resp = self.client.get(f"/api/data/{sid}/rows?page=3&page_size=100")
        self.assertEqual(page_resp.status_code, 200, page_resp.get_data(as_text=True))
        payload = page_resp.get_json()
        self.assertEqual(payload["columns"], ["x", "y", "label"])
        self.assertEqual(payload["total_rows"], 205)
        self.assertEqual(payload["total_pages"], 3)
        self.assertEqual(len(payload["rows"]), 5)
        self.assertEqual(payload["rows"][0]["label"], "row-200")

        overflow_resp = self.client.get(f"/api/data/{sid}/rows?page=999&page_size=100")
        self.assertEqual(overflow_resp.status_code, 200, overflow_resp.get_data(as_text=True))
        self.assertEqual(overflow_resp.get_json()["page"], 3)
        self.assertEqual(len(overflow_resp.get_json()["rows"]), 5)

        invalid = self.client.get(f"/api/data/{sid}/rows?page=0&page_size=100")
        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(invalid.get_json()["error"]["code"], "INVALID_PAGINATION")

    def test_upload_preview_serializes_missing_and_infinite_values_as_null(self):
        response = self._upload_csv("x,y\n1,\n2,inf\n", filename="UPPER.CSV")
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertIsNone(payload["preview"][0]["y"])
        self.assertIsNone(payload["preview"][1]["y"])
        self.assertEqual(payload["column_profiles"][1]["missing_count"], 2)
        self.assertNotIn(":NaN", response.get_data(as_text=True))
        self.assertNotIn(":Infinity", response.get_data(as_text=True))

    def test_sync_uses_upload_limits_and_updates_session(self):
        upload_resp = self._upload_csv("x,y\n1,2\n", filename="initial.csv")
        self.assertEqual(upload_resp.status_code, 200, upload_resp.get_data(as_text=True))
        sid = upload_resp.get_json()["session_id"]

        sync_resp = self.client.post(
            f"/api/data/{sid}/sync",
            data={"file": (io.BytesIO(b"x,y\n3,4\n5,6\n"), "UPDATED.CSV")},
            content_type="multipart/form-data",
        )
        self.assertEqual(sync_resp.status_code, 200, sync_resp.get_data(as_text=True))
        self.assertEqual(sync_resp.get_json()["n_rows"], 2)
        profile_resp = self.client.get(f"/api/data/{sid}/profile")
        self.assertEqual(profile_resp.get_json()["session_meta"]["source_name"], "UPDATED.CSV")

        process_resp = self.client.post(f"/api/data/{sid}/process", json={
            "operations": [{"op": "drop_rows", "positions": [1]}]
        })
        self.assertEqual(process_resp.status_code, 200, process_resp.get_data(as_text=True))
        undo_resp = self.client.post(f"/api/data/{sid}/undo", json={})
        self.assertEqual(undo_resp.status_code, 200, undo_resp.get_data(as_text=True))
        self.assertEqual(undo_resp.get_json()["n_rows"], 2)
        self.assertEqual(undo_resp.get_json()["preview"][0]["x"], 3)

        with patch("routes.data_routes.MAX_DATASET_ROWS", 1):
            rejected = self.client.post(
                f"/api/data/{sid}/sync",
                data={"file": (io.BytesIO(b"x,y\n1,2\n3,4\n"), "too_many.csv")},
                content_type="multipart/form-data",
            )
        self.assertEqual(rejected.status_code, 413)
        self.assertEqual(rejected.get_json()["error"]["code"], "DATASET_ROW_LIMIT_EXCEEDED")

    def test_session_store_rejects_path_like_ids(self):
        outside_data = os.path.join(self.tmpdir, "outside.pkl")
        outside_meta = os.path.join(self.tmpdir, "outside.meta.json")
        pd.DataFrame({"secret": [1]}).to_pickle(outside_data)
        with open(outside_meta, "w", encoding="utf-8") as fh:
            json.dump({"user_id": self.user["user_id"]}, fh)

        loaded = self.session_store.get_session("../outside", user_id=self.user["user_id"])

        self.assertIsNone(loaded)
        self.assertNotIn("../outside", self.session_store._sessions)

    def test_history_pruning_removes_discarded_state_files(self):
        sid = self.session_store.create_session(
            pd.DataFrame({"x": [0]}), user_id=self.user["user_id"]
        )
        for value in range(1, 52):
            self.session_store.update_session(
                sid,
                pd.DataFrame({"x": [value]}),
                user_id=self.user["user_id"],
                history_entry={"label": f"step {value}"},
            )

        history = self.session_store.processing_history(sid, user_id=self.user["user_id"])
        self.assertEqual(len(history["history"]), 50)
        state_files = [
            name for name in os.listdir(self.sessions_dir)
            if name.startswith(f"{sid}.state.")
        ]
        self.assertEqual(len(state_files), 50)

        self.session_store.undo_session(sid, user_id=self.user["user_id"])
        discarded_state_id = history["history"][-1]["state_id"]
        self.session_store.update_session(
            sid,
            pd.DataFrame({"x": [999]}),
            user_id=self.user["user_id"],
            history_entry={"label": "branched"},
        )
        self.assertFalse(os.path.exists(
            self.session_store._state_path(sid, discarded_state_id)
        ))

    def test_account_and_cookie_session_are_persisted_in_sqlite(self):
        with self.database.connect() as connection:
            user_row = connection.execute(
                "SELECT username FROM users WHERE user_id = ?", (self.user["user_id"],)
            ).fetchone()
            session_count = connection.execute(
                "SELECT COUNT(*) FROM auth_sessions WHERE user_id = ?", (self.user["user_id"],)
            ).fetchone()[0]

        self.assertEqual(user_row["username"], "tester")
        self.assertEqual(session_count, 1)
        health_resp = self.client.get("/api/health")
        self.assertEqual(health_resp.status_code, 200, health_resp.get_data(as_text=True))
        self.assertEqual(health_resp.get_json()["user"]["user_id"], self.user["user_id"])
        self.assertEqual(self.user["role"], "admin")

    def test_legacy_json_accounts_are_imported_into_sqlite(self):
        secret = self.auth_store._password_hash("password123")
        with open(self.auth_store.USERS_PATH, "w", encoding="utf-8") as fh:
            json.dump({
                "users": {
                    "legacy_user": {
                        "user_id": "legacy-user-id",
                        "username": "legacy_user",
                        "password_salt": secret["salt"],
                        "password_hash": secret["hash"],
                        "created_at": 123.0,
                    }
                }
            }, fh)

        legacy = self.auth_store.verify_user("legacy_user", "password123")
        self.assertIsNotNone(legacy)
        self.assertEqual(legacy["user_id"], "legacy-user-id")

    def test_admin_can_manage_accounts_and_read_audit_logs(self):
        other_client, other_user = self._make_authed_client("managed_user")
        self.assertEqual(other_user["role"], "user")

        denied = other_client.get("/api/admin/users")
        self.assertEqual(denied.status_code, 403)
        self.assertEqual(denied.get_json()["error"]["code"], "ADMIN_REQUIRED")

        users_resp = self.client.get("/api/admin/users")
        self.assertEqual(users_resp.status_code, 200, users_resp.get_data(as_text=True))
        users = users_resp.get_json()["users"]
        self.assertEqual(len(users), 2)

        disable_resp = self.client.patch(
            f"/api/admin/users/{other_user['user_id']}", json={"disabled": True}
        )
        self.assertEqual(disable_resp.status_code, 200, disable_resp.get_data(as_text=True))
        self.assertTrue(disable_resp.get_json()["user"]["disabled"])
        self.assertEqual(other_client.get("/api/health").status_code, 401)

        logs_resp = self.client.get("/api/admin/audit-logs?limit=20")
        self.assertEqual(logs_resp.status_code, 200, logs_resp.get_data(as_text=True))
        actions = [item["action"] for item in logs_resp.get_json()["audit_logs"]]
        self.assertIn("admin.user_access_updated", actions)

    def test_admin_cannot_disable_or_demote_current_account(self):
        disable_resp = self.client.patch(
            f"/api/admin/users/{self.user['user_id']}", json={"disabled": True}
        )
        demote_resp = self.client.patch(
            f"/api/admin/users/{self.user['user_id']}", json={"role": "user"}
        )
        self.assertEqual(disable_resp.status_code, 400)
        self.assertEqual(demote_resp.status_code, 400)
        self.assertEqual(disable_resp.get_json()["error"]["code"], "SELF_ACCESS_CHANGE_DENIED")

    def test_login_failures_are_rate_limited(self):
        client = self.backend_app.app.test_client()
        with patch.object(self.auth_store, "LOGIN_MAX_FAILURES", 2), \
             patch.object(self.auth_store, "LOGIN_WINDOW_SECONDS", 60), \
             patch.object(self.auth_store, "LOGIN_LOCK_SECONDS", 30):
            first = client.post("/api/auth/login", json={"username": "tester", "password": "wrong-pass"})
            second = client.post("/api/auth/login", json={"username": "tester", "password": "wrong-pass"})
            blocked = client.post("/api/auth/login", json={"username": "tester", "password": "password123"})

        self.assertEqual(first.status_code, 401)
        self.assertEqual(second.status_code, 429)
        self.assertEqual(second.get_json()["error"]["code"], "LOGIN_RATE_LIMITED")
        self.assertEqual(second.headers.get("Retry-After"), "30")
        self.assertEqual(blocked.status_code, 429)

    def test_registration_attempts_are_rate_limited(self):
        client = self.backend_app.app.test_client()
        with patch.object(self.auth_store, "REGISTER_MAX_ATTEMPTS", 1), \
             patch.object(self.auth_store, "REGISTER_WINDOW_SECONDS", 60):
            first = client.post("/api/auth/register", json={"username": "new_user_one", "password": "password123"})
            second = client.post("/api/auth/register", json={"username": "new_user_two", "password": "password123"})

        self.assertEqual(first.status_code, 200, first.get_data(as_text=True))
        self.assertEqual(second.status_code, 429)
        self.assertEqual(second.get_json()["error"]["code"], "REGISTER_RATE_LIMITED")
        self.assertEqual(second.headers.get("Retry-After"), "60")

    def test_upload_resource_quotas_are_enforced(self):
        with patch("routes.data_routes.MAX_SESSIONS_PER_USER", 1):
            first = self._upload_csv("x,y\n1,2\n", filename="first.csv")
            second = self._upload_csv("x,y\n3,4\n", filename="second.csv")

        self.assertEqual(first.status_code, 200, first.get_data(as_text=True))
        self.assertEqual(second.status_code, 429)
        self.assertEqual(second.get_json()["error"]["code"], "SESSION_QUOTA_EXCEEDED")

        self.session_store._sessions.clear()
        with patch("routes.data_routes.MAX_DATASET_ROWS", 2):
            oversized = self._upload_csv("x,y\n1,2\n3,4\n5,6\n", filename="too_many_rows.csv")
        self.assertEqual(oversized.status_code, 413)
        self.assertEqual(oversized.get_json()["error"]["code"], "DATASET_ROW_LIMIT_EXCEEDED")

    def test_task_queue_cancel_and_task_visibility_are_user_scoped(self):
        gate = threading.Event()
        first = self.task_store.create_task(
            "training",
            "long task",
            metadata={"user_id": self.user["user_id"], "model_type": "regression"},
        )
        self.task_store.submit_task(first, lambda: (gate.wait(2.0), {"version_id": "done"})[1])

        deadline = time.time() + 2.0
        first_payload = None
        while time.time() < deadline:
            first_resp = self.client.get(f"/api/tasks/{first}")
            self.assertEqual(first_resp.status_code, 200, first_resp.get_data(as_text=True))
            first_payload = first_resp.get_json()
            if first_payload["status"] == "running":
                break
            time.sleep(0.02)
        self.assertEqual(first_payload["status"], "running", first_payload)

        second = self.task_store.create_task(
            "training",
            "queued task",
            metadata={"user_id": self.user["user_id"], "model_type": "classification"},
        )
        self.task_store.submit_task(second, lambda: {"version_id": "should-not-run"})

        second_resp = self.client.get(f"/api/tasks/{second}")
        self.assertEqual(second_resp.status_code, 200, second_resp.get_data(as_text=True))
        self.assertEqual(second_resp.get_json()["status"], "queued")

        cancel_resp = self.client.post(f"/api/tasks/{second}/cancel")
        self.assertEqual(cancel_resp.status_code, 200, cancel_resp.get_data(as_text=True))
        self.assertEqual(cancel_resp.get_json()["status"], "cancelled")

        other_client, _other_user = self._make_authed_client("task_other")
        hidden_resp = other_client.get(f"/api/tasks/{first}")
        self.assertEqual(hidden_resp.status_code, 404)

        gate.set()
        first_done = self._wait_for_task(first)
        self.assertEqual(first_done["status"], "succeeded", first_done)

    def test_task_pending_quota_and_sync_capacity_are_enforced(self):
        gate = threading.Event()
        first = self.task_store.create_task(
            "training",
            "capacity holder",
            metadata={"user_id": self.user["user_id"]},
        )
        self.task_store.submit_task(first, lambda: (gate.wait(2.0), {"version_id": "done"})[1])

        deadline = time.time() + 2.0
        while time.time() < deadline:
            if self.task_store.get_task(first)["status"] == "running":
                break
            time.sleep(0.02)

        with patch.object(self.task_store, "MAX_PENDING_TASKS_PER_USER", 1):
            second = self.task_store.create_task(
                "training",
                "quota rejected",
                metadata={"user_id": self.user["user_id"]},
            )
            self.assertIsNone(self.task_store.submit_task(second, lambda: {"version_id": "no"}))
            self.assertEqual(self.task_store.get_task(second)["status"], "failed")

        third = self.task_store.create_task(
            "training",
            "sync rejected",
            metadata={"user_id": self.user["user_id"]},
        )
        self.assertFalse(self.task_store.start_inline_task(third))
        self.assertEqual(self.task_store.get_task(third)["status"], "failed")

        gate.set()
        first_done = self._wait_for_task(first)
        self.assertEqual(first_done["status"], "succeeded", first_done)

    def test_task_history_is_restored_and_interrupted_work_is_failed(self):
        finished = self.task_store.create_task(
            "training", "finished", metadata={"user_id": self.user["user_id"]}
        )
        self.task_store.finish_task(finished, {"version_id": "persisted-version"})
        interrupted = self.task_store.create_task(
            "training", "interrupted", metadata={"user_id": self.user["user_id"]}
        )

        self.task_store._tasks.clear()
        restored_count = self.task_store.restore_tasks()

        self.assertEqual(restored_count, 2)
        self.assertEqual(self.task_store.get_task(finished)["status"], "succeeded")
        interrupted_task = self.task_store.get_task(interrupted)
        self.assertEqual(interrupted_task["status"], "failed")
        self.assertIn("服务重启", interrupted_task["error"])

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
        self.assertIn("task_id", train_data)
        self.assertIn("r2", train_data)
        self.assertIn("tree_rules", train_data)
        self.assertIn("tree_nodes", train_data)

        tasks_resp = self.client.get("/api/tasks")
        self.assertEqual(tasks_resp.status_code, 200, tasks_resp.get_data(as_text=True))
        tasks = tasks_resp.get_json()["tasks"]
        task = next(item for item in tasks if item["task_id"] == train_data["task_id"])
        self.assertEqual(task["kind"], "training")
        self.assertEqual(task["status"], "succeeded")
        self.assertEqual(task["metadata"]["model_type"], "decision_tree")
        self.assertEqual(task["result"]["version_id"], train_data["version_id"])

        task_resp = self.client.get(f"/api/tasks/{train_data['task_id']}")
        self.assertEqual(task_resp.status_code, 200, task_resp.get_data(as_text=True))
        self.assertEqual(task_resp.get_json()["task_id"], train_data["task_id"])

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

        batch_resp = self.client.post("/api/decision_tree/batch_predict", json={
            "rows": [{"x1": 5, "x2": 10}, {"x1": 6, "x2": 12}],
            "task_type": "regression",
            "version_id": train_data["version_id"],
        })
        self.assertEqual(batch_resp.status_code, 200, batch_resp.get_data(as_text=True))
        batch_data = batch_resp.get_json()
        self.assertEqual(len(batch_data["predictions"]), 2)
        self.assertIn("pred_value", batch_data["predictions"][0])

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

    def test_decision_tree_classification_returns_report_for_react(self):
        rows = ["x1,x2,label"]
        for i in range(20):
            rows.append(f"{i},{i + 1},low")
        for i in range(20):
            rows.append(f"{100 + i},{101 + i},high")

        upload_resp = self._upload_csv("\n".join(rows), filename="tree_cls.csv")
        self.assertEqual(upload_resp.status_code, 200, upload_resp.get_data(as_text=True))
        sid = upload_resp.get_json()["session_id"]

        train_resp = self.client.post("/api/decision_tree/train", json={
            "session_id": sid,
            "target_col": "label",
            "feature_cols": ["x1", "x2"],
            "task_type": "classification",
            "criterion": "gini",
            "max_depth": 3,
        })
        self.assertEqual(train_resp.status_code, 200, train_resp.get_data(as_text=True))
        train_data = train_resp.get_json()
        self.assertIn("classification_report", train_data)
        self.assertIn("low", train_data["classification_report"])
        self.assertIn("f1-score", train_data["classification_report"]["low"])

        status_resp = self.client.get("/api/decision_tree/status")
        self.assertEqual(status_resp.status_code, 200, status_resp.get_data(as_text=True))
        status_data = status_resp.get_json()
        self.assertIn("classification_report", status_data)
        self.assertIn("high", status_data["classification_report"])

    def test_decision_tree_train_supports_async_task_result(self):
        rows = ["x1,x2,y"]
        for i in range(24):
            rows.append(f"{i},{i * 2},{i * 3}")

        upload_resp = self._upload_csv("\n".join(rows), filename="tree_async.csv")
        self.assertEqual(upload_resp.status_code, 200, upload_resp.get_data(as_text=True))
        sid = upload_resp.get_json()["session_id"]

        train_resp = self.client.post("/api/decision_tree/train?async=1", json={
            "session_id": sid,
            "target_col": "y",
            "feature_cols": ["x1", "x2"],
            "task_type": "regression",
            "criterion": "squared_error",
            "max_depth": 3,
        })
        self.assertEqual(train_resp.status_code, 202, train_resp.get_data(as_text=True))
        train_data = train_resp.get_json()
        self.assertTrue(train_data["async"])
        task_id = train_data["task_id"]

        task = self._wait_for_task(task_id)
        self.assertEqual(task["status"], "succeeded", task)
        self.assertEqual(task["metadata"]["model_type"], "decision_tree")
        self.assertIn("version_id", task["result"])
        self.assertIn("result_payload", task)
        self.assertEqual(task["result_payload"]["task_id"], task_id)
        self.assertIn("tree_nodes", task["result_payload"])

        status_resp = self.client.get("/api/decision_tree/status")
        self.assertEqual(status_resp.status_code, 200, status_resp.get_data(as_text=True))
        self.assertEqual(status_resp.get_json()["version_id"], task["result"]["version_id"])

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

    def test_data_processing_history_undo_redo_and_pipeline(self):
        upload_resp = self._upload_csv("x,y,cat\n1,10,a\n2,20,b\n3,30,a\n", filename="history.csv")
        self.assertEqual(upload_resp.status_code, 200, upload_resp.get_data(as_text=True))
        sid = upload_resp.get_json()["session_id"]

        process_resp = self.client.post(f"/api/data/{sid}/process", json={
            "operations": [
                {"op": "drop_cols", "cols": ["cat"]},
            ]
        })
        self.assertEqual(process_resp.status_code, 200, process_resp.get_data(as_text=True))
        processed = process_resp.get_json()
        self.assertNotIn("cat", processed["columns"])
        self.assertTrue(processed["processing_history"]["can_undo"])
        self.assertFalse(processed["processing_history"]["can_redo"])
        self.assertEqual(len(processed["processing_history"]["history"]), 2)

        undo_resp = self.client.post(f"/api/data/{sid}/undo", json={})
        self.assertEqual(undo_resp.status_code, 200, undo_resp.get_data(as_text=True))
        undone = undo_resp.get_json()
        self.assertIn("cat", undone["columns"])
        self.assertFalse(undone["processing_history"]["can_undo"])
        self.assertTrue(undone["processing_history"]["can_redo"])

        redo_resp = self.client.post(f"/api/data/{sid}/redo", json={})
        self.assertEqual(redo_resp.status_code, 200, redo_resp.get_data(as_text=True))
        redone = redo_resp.get_json()
        self.assertNotIn("cat", redone["columns"])

        save_resp = self.client.post(f"/api/data/{sid}/pipelines", json={"name": "删除类别列"})
        self.assertEqual(save_resp.status_code, 200, save_resp.get_data(as_text=True))
        pipeline = save_resp.get_json()["pipeline"]
        self.assertEqual(pipeline["name"], "删除类别列")
        self.assertEqual(pipeline["step_count"], 1)

        upload_resp = self._upload_csv("x,y,cat\n1,10,a\n2,20,b\n", filename="pipeline_apply.csv")
        self.assertEqual(upload_resp.status_code, 200, upload_resp.get_data(as_text=True))
        sid2 = upload_resp.get_json()["session_id"]
        save_direct = self.client.post(f"/api/data/{sid2}/pipelines", json={
            "name": "direct",
            "operations": [{"op": "drop_cols", "cols": ["cat"]}],
        })
        self.assertEqual(save_direct.status_code, 200, save_direct.get_data(as_text=True))
        pipeline_id = save_direct.get_json()["pipeline"]["pipeline_id"]
        apply_resp = self.client.post(f"/api/data/{sid2}/pipelines/{pipeline_id}/apply", json={})
        self.assertEqual(apply_resp.status_code, 200, apply_resp.get_data(as_text=True))
        applied = apply_resp.get_json()
        self.assertNotIn("cat", applied["columns"])
        self.assertTrue(applied["processing_history"]["can_undo"])

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

        outlier_resp = self.client.get(f"/api/data/{sid}/outliers?coefficient=3")
        self.assertEqual(outlier_resp.status_code, 200, outlier_resp.get_data(as_text=True))
        outlier_data = outlier_resp.get_json()
        self.assertEqual(outlier_data["coefficient"], 3.0)
        self.assertIn("x", outlier_data["outliers"])

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
            "hover_cols": ["cat", "z"],
        })
        self.assertEqual(scatter_resp.status_code, 200, scatter_resp.get_data(as_text=True))
        scatter = scatter_resp.get_json()
        self.assertEqual(scatter["chart_type"], "scatter")
        self.assertIn("points", scatter["data"])
        self.assertEqual(len(scatter["data"]["points"]), 5)
        self.assertIn("hover", scatter["data"]["points"][0])

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

        hist_resp = self.client.post(f"/api/data/{sid}/visualize", json={
            "chart_type": "histogram",
            "col": "y",
            "group_col": "cat",
            "bins": 4,
        })
        self.assertEqual(hist_resp.status_code, 200, hist_resp.get_data(as_text=True))
        hist = hist_resp.get_json()
        self.assertIn("bins", hist["data"])
        self.assertTrue(any(item.get("group") == "a" for item in hist["data"]["bins"]))

        violin_resp = self.client.post(f"/api/data/{sid}/visualize", json={
            "chart_type": "violin",
            "x_col": "cat",
            "y_col": "y",
        })
        self.assertEqual(violin_resp.status_code, 200, violin_resp.get_data(as_text=True))
        violin = violin_resp.get_json()
        self.assertEqual(violin["chart_type"], "violin")
        self.assertIn("groups", violin["data"])
        self.assertTrue(all("values" in group for group in violin["data"]["groups"]))

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
            "user_id": self.user["user_id"],
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
            "user_id": self.user["user_id"],
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
            "classification_report": {
                "no": {"precision": 0.75, "recall": 0.75, "f1-score": 0.75, "support": 4},
                "yes": {"precision": 0.75, "recall": 0.75, "f1-score": 0.75, "support": 4},
            },
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
        self.assertEqual(status_data["classification_report"]["no"]["f1-score"], 0.75)
        self.assertEqual(status_data["train_losses"], [0.8, 0.6])

    def test_diy_mlp_status_exposes_architecture_and_curves_for_react(self):
        layers = [
            {"neurons": 16, "activation": "ReLU", "bn": True, "dropout": 0.1},
            {"neurons": 8, "activation": "GELU", "bn": False, "dropout": 0.2},
        ]
        self.registry.register_version("diy_mlp", "diy_test_v1", {
            "user_id": self.user["user_id"],
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

        batch_resp = self.client.post("/api/clustering/batch_predict", json={
            "rows": [[3, 4], [104, 105]],
            "version_id": train_data["version_id"],
        })
        self.assertEqual(batch_resp.status_code, 200, batch_resp.get_data(as_text=True))
        batch_data = batch_resp.get_json()
        self.assertEqual(len(batch_data["clusters"]), 2)

    def test_clustering_train_supports_async_task_result(self):
        rows = ["x1,x2"]
        for i in range(12):
            rows.append(f"{i},{i + 1}")
        for i in range(12):
            rows.append(f"{100 + i},{101 + i}")

        upload_resp = self._upload_csv("\n".join(rows), filename="cluster_async.csv")
        self.assertEqual(upload_resp.status_code, 200, upload_resp.get_data(as_text=True))
        sid = upload_resp.get_json()["session_id"]

        train_resp = self.client.post("/api/clustering/train?async=1", json={
            "session_id": sid,
            "feature_cols": ["x1", "x2"],
            "algorithm": "kmeans",
            "params": {"n_clusters": 2},
        })
        self.assertEqual(train_resp.status_code, 202, train_resp.get_data(as_text=True))
        task_id = train_resp.get_json()["task_id"]

        task = self._wait_for_task(task_id)
        self.assertEqual(task["status"], "succeeded", task)
        self.assertEqual(task["metadata"]["model_type"], "clustering")
        self.assertEqual(task["result_payload"]["task_id"], task_id)
        self.assertEqual(task["result_payload"]["n_found"], 2)
        self.assertIn("version_id", task["result"])

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
