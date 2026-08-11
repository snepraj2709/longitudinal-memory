from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import time
import unittest

from evaluation.qwen_lifecycle import (
    BudgetWatchdog,
    CommandResult,
    JarvisLifecycleError,
    JarvisManager,
    LiveBudgetController,
    REMOTE_MODEL_DIR,
    REMOTE_METADATA_DIR,
    VLLM_BUILD,
    select_approved_offer,
    stage_one_gate,
    stage_two_gate,
)
from evaluation.qwen_series import BudgetStop


INVENTORY = {
    "data": [{
        "gpu_type": "H100 80GB", "region": "IN2", "workload_type": "container",
        "num_free_devices": 1, "spot_price": 112.59,
    }, {
        "gpu_type": "RTX PRO 6000 96GB", "region": "IN1", "workload_type": "container",
        "num_free_devices": 2, "spot_price": "INR 93.96/hour",
    }]
}


class QwenLifecycleTests(unittest.TestCase):
    def test_offer_prefers_approved_h100_then_rtx_fallback(self) -> None:
        self.assertEqual(select_approved_offer(INVENTORY).gpu, "H100-80GB")
        expensive = json.loads(json.dumps(INVENTORY))
        expensive["data"][0]["spot_price"] = 140
        self.assertEqual(select_approved_offer(expensive).gpu, "RTX-PRO6000-96GB")
        expensive["data"][1]["spot_price"] = 101
        with self.assertRaisesRegex(JarvisLifecycleError, "no approved"):
            select_approved_offer(expensive)

    def test_budget_reserves_cleanup_and_concurrent_attempts(self) -> None:
        now = [0.0]
        budget = LiveBudgetController(
            stage=1, hourly_rate_inr=120, stage_cap_inr=7,
            cumulative_cap_inr=1500, prior_cost_inr=0,
            started_monotonic=0, monotonic=lambda: now[0],
            cleanup_reserve_seconds=60, initial_request_seconds=60,
        )
        budget.before_attempt(None, False)
        budget.before_attempt(None, False)
        with self.assertRaises(BudgetStop):
            budget.before_attempt(None, False)
        budget.after_attempt(None, False, 5)
        budget.after_attempt(None, False, 5)
        self.assertEqual(budget.snapshot()["active_reservations"], 0)

    def test_budget_includes_cost_from_prior_attempt_in_stage_cap(self) -> None:
        budget = LiveBudgetController(
            stage=1, hourly_rate_inr=3600, stage_cap_inr=200,
            cumulative_cap_inr=200, prior_cost_inr=3.05,
            prior_stage_cost_inr=3.05, started_monotonic=0,
            monotonic=lambda: 76.0, cleanup_reserve_seconds=120,
            initial_request_seconds=1,
        )
        with self.assertRaises(BudgetStop):
            budget.before_attempt(None, False)

    def test_cleanup_download_failure_still_destroys_and_verifies_absence(self) -> None:
        commands = []

        def runner(command):
            commands.append(tuple(command))
            if command[1] == "status":
                return CommandResult(0, '{"balance":{"balance":1000.0}}', "")
            if command[1] in {"gpus", "resources"}:
                return CommandResult(0, json.dumps(INVENTORY), "")
            if command[1] == "create":
                return CommandResult(0, '{"machine_id":321}', "")
            if command[1] == "download":
                return CommandResult(1, "", "preempted")
            if command[1] == "destroy":
                return CommandResult(0, '{"status":"destroyed"}', "")
            if command[1] == "list":
                return CommandResult(0, '[]', "")
            if command[1] == "get":
                return CommandResult(1, "", "not found")
            return CommandResult(0, "{}", "")

        with tempfile.TemporaryDirectory() as directory:
            manager = JarvisManager(
                artifact_dir=Path(directory), runner=runner,
                clock=lambda: datetime(2026, 8, 11, tzinfo=timezone.utc),
                monotonic=lambda: 10.0,
            )
            manager.create()
            receipt = manager.cleanup()
            self.assertTrue(receipt["destruction_verified"])
            self.assertTrue((Path(directory) / "lifecycle.json").is_file())
        names = [command[1] for command in commands]
        self.assertLess(names.index("download"), names.index("destroy"))
        self.assertIn("list", names)
        self.assertIn("get", names)

    def test_create_uses_the_pytorch_default_6006_endpoint(self) -> None:
        commands = []

        def runner(command):
            commands.append(tuple(command))
            if command[1] == "status":
                return CommandResult(0, '{"balance":{"balance":1000.0}}', "")
            if command[1] in {"gpus", "resources"}:
                return CommandResult(0, json.dumps(INVENTORY), "")
            if command[1] == "create":
                return CommandResult(0, '{"machine_id":321}', "")
            return CommandResult(0, "{}", "")

        manager = JarvisManager(artifact_dir=Path("unused"), runner=runner)
        manager.create()
        create = next(command for command in commands if command[1] == "create")
        self.assertEqual(create[create.index("--gpu") + 1], "H100")
        self.assertEqual(create[create.index("--region") + 1], "IN2")
        self.assertEqual(create[create.index("--template") + 1], "pytorch")
        self.assertNotIn("--http-ports", create)
        manager._restore_signal_handlers()

    def test_install_uses_exact_official_commit_wheel_and_verifies_version(self) -> None:
        commands = []

        def runner(command):
            commands.append(tuple(command))
            if "import flashinfer.comm.fd_exchange" in command[-1]:
                return CommandResult(
                    0, "(3, 12) " + str(VLLM_BUILD["version"]) + " array.array[int]\n", "",
                )
            return CommandResult(0, "Python 3.12\ntorch cuda\nRTX PRO 6000\n", "")

        manager = JarvisManager(artifact_dir=Path("unused"), runner=runner)
        manager.machine_id = 321
        manager.install_vllm(str(VLLM_BUILD["revision"]))
        conda = next(command for command in commands if "conda create" in command[-1])
        self.assertIn("--override-channels -c conda-forge", conda[-1])
        install = next(command for command in commands if "pip install" in command[-1])
        self.assertIn(str(VLLM_BUILD["wheel"]), install[-1])
        self.assertIn("/home/qwen-v2-env/bin/python", install[-1])
        self.assertNotIn("git+https", install[-1])
        verify = next(command for command in commands if "flashinfer.comm.fd_exchange" in command[-1])
        self.assertIn("array.array[int]", verify[-1])

    def test_install_failure_saves_sanitized_tail(self) -> None:
        def runner(command):
            if "pip install" in command[-1]:
                return CommandResult(1, "", "")
            if command[-1].startswith("tail -n"):
                return CommandResult(0, "token=secret-value\nresolver failed\n", "")
            return CommandResult(0, "Python 3.12\ntorch cuda\nRTX PRO 6000\n", "")

        with tempfile.TemporaryDirectory() as directory:
            manager = JarvisManager(artifact_dir=Path(directory), runner=runner)
            manager.machine_id = 321
            with self.assertRaisesRegex(JarvisLifecycleError, "diagnostic saved"):
                manager.install_vllm(str(VLLM_BUILD["revision"]))
            diagnostic = (Path(directory) / "install-failure.json").read_text()
            self.assertIn("[REDACTED]", diagnostic)
            self.assertNotIn("secret-value", diagnostic)

    def test_dead_server_saves_sanitized_diagnostic(self) -> None:
        def runner(command):
            if "kill -0" in command[-1]:
                return CommandResult(1, "", "")
            return CommandResult(0, "api_key=secret-value\nstartup failed\n", "")

        with tempfile.TemporaryDirectory() as directory:
            manager = JarvisManager(artifact_dir=Path(directory), runner=runner)
            manager.machine_id = 321
            self.assertFalse(manager.server_alive())
            diagnostic = (Path(directory) / "server-failure.json").read_text()
            self.assertIn("[REDACTED]", diagnostic)
            self.assertNotIn("secret-value", diagnostic)

    def test_model_download_is_revision_pinned_and_tokenizer_checked(self) -> None:
        commands = []

        def runner(command):
            commands.append(tuple(command))
            if command[-1].startswith("tail -n 1"):
                return CommandResult(0, 'Executing on 321\n{"file_count":20,"manifest_sha256":"abc","tokenizer_class":"TokenizersBackend","tokenizer_size":248077,"total_bytes":30}\n', "")
            return CommandResult(0, "", "")

        manager = JarvisManager(artifact_dir=Path("unused"), runner=runner)
        manager.machine_id = 321
        receipt = manager.download_model(
            model_id="Qwen/Qwen3.5-27B-FP8",
            revision="97f5941bf617e31c5e237364a8602ce3f03a551a",
        )
        self.assertEqual(receipt["tokenizer_size"], 248077)
        download = commands[0][-1]
        self.assertIn("97f5941bf617e31c5e237364a8602ce3f03a551a", download)
        self.assertIn(REMOTE_MODEL_DIR, download)
        self.assertIn('os.environ["HF_TOKEN"]', download)

    def test_metadata_preflight_excludes_weights_and_checks_tokenizer(self) -> None:
        commands = []

        def runner(command):
            commands.append(tuple(command))
            if command[-1].startswith("tail -n 1"):
                return CommandResult(
                    0, '{"file_count":12,"tokenizer_class":"Qwen2Tokenizer",'
                    '"tokenizer_size":248077,"total_bytes":1000}\n', "",
                )
            return CommandResult(0, "", "")

        manager = JarvisManager(artifact_dir=Path("unused"), runner=runner)
        manager.machine_id = 321
        receipt = manager.download_model_metadata(
            model_id="Qwen/Qwen3.5-27B-FP8",
            revision="97f5941bf617e31c5e237364a8602ce3f03a551a",
        )
        self.assertEqual(receipt["tokenizer_size"], 248077)
        download = commands[0][-1]
        self.assertIn(REMOTE_METADATA_DIR, download)
        self.assertIn("ignore_patterns", download)
        self.assertIn("*.safetensors", download)

    def test_preflight_server_uses_dummy_weights_and_can_be_stopped(self) -> None:
        commands = []

        def runner(command):
            commands.append(tuple(command))
            return CommandResult(0, "", "")

        manager = JarvisManager(artifact_dir=Path("unused"), runner=runner)
        manager.machine_id = 321
        manager.start_server(preflight=True)
        manager.stop_server()
        self.assertIn("run_qwen_vllm.sh preflight", commands[0][-1])
        self.assertIn("nohup setsid", commands[0][-1])
        self.assertIn("qwen-v2-server.pid", commands[1][-1])
        self.assertIn("kill -TERM -- -", commands[1][-1])

    def test_watchdog_invokes_cleanup_when_reserve_is_exhausted(self) -> None:
        now = [100.0]
        controller = LiveBudgetController(
            stage=1, hourly_rate_inr=3600, stage_cap_inr=105,
            cumulative_cap_inr=105, prior_cost_inr=0,
            started_monotonic=0, monotonic=lambda: now[0],
            cleanup_reserve_seconds=10, initial_request_seconds=1,
        )
        calls = []
        with BudgetWatchdog(controller, lambda: calls.append("cleanup"), interval_seconds=0.01) as watchdog:
            time.sleep(0.04)
        self.assertTrue(watchdog.triggered)
        self.assertEqual(calls, ["cleanup"])

    def test_stage_gates_enforce_validity_and_reserved_projection(self) -> None:
        passed = stage_one_gate(
            {"planned_request_count": 12, "successful_count": 12},
            projected_remaining_cost_inr=1000, exact_model=True, no_oom=True,
        )
        self.assertTrue(passed.passed)
        failed = stage_two_gate(
            {"planned_request_count": 20, "successful_count": 20},
            {"planned_request_count": 798, "successful_count": 790},
            {"planned_request_count": 112, "successful_count": 100},
            batch_valid_rates={"B0_qa": 0.89},
            projected_stage_three_cost_inr=900,
        )
        self.assertFalse(failed.passed)
        self.assertIn("task_or_baseline_batch_below_90_percent", failed.reasons)

    def test_failed_destroy_can_be_retried(self) -> None:
        destroys = [CommandResult(1, "", "temporary"), CommandResult(0, "{}", "")]

        def runner(command):
            if command[1] == "download":
                return CommandResult(1, "", "missing")
            if command[1] == "destroy":
                return destroys.pop(0)
            if command[1] == "list":
                return CommandResult(0, "[]", "")
            if command[1] == "get":
                return CommandResult(1, "", "missing")
            return CommandResult(0, "{}", "")

        with tempfile.TemporaryDirectory() as directory:
            manager = JarvisManager(artifact_dir=Path(directory), runner=runner)
            manager.machine_id = 99
            with self.assertRaisesRegex(JarvisLifecycleError, "destroy command"):
                manager.cleanup()
            self.assertTrue(manager.cleanup()["destruction_verified"])


if __name__ == "__main__":
    unittest.main()
