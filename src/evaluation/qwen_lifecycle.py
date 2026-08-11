"""JarvisLabs resource selection, budget reservation, and destructive cleanup."""

from __future__ import annotations

from collections import deque
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import signal
import stat
import subprocess
import threading
import time
from typing import Callable, Mapping, Sequence

from .qwen_series import BudgetStop
from .qwen_v2_contract import SERIES_ID


H100_MAX_RATE = 133.33
RTX_MAX_RATE = 100.0
DEFAULT_CLEANUP_RESERVE_SECONDS = 600.0
STAGE_LIMITS = {
    1: {"stage_cap_inr": 200.0, "cumulative_cap_inr": 200.0},
    2: {"stage_cap_inr": 300.0, "cumulative_cap_inr": 500.0},
    3: {"stage_cap_inr": 1000.0, "cumulative_cap_inr": 1500.0},
}
VLLM_BUILD = {
    "revision": "65b7662d3fcb773afaf751ab29ac6960a0cf011d",
    "version": "0.26.1rc1.dev602+g65b7662d3",
    "wheel": (
        "https://wheels.vllm.ai/65b7662d3fcb773afaf751ab29ac6960a0cf011d/"
        "vllm-0.26.1rc1.dev602%2Bg65b7662d3-cp38-abi3-manylinux_2_28_x86_64.whl"
    ),
}
REMOTE_MODEL_DIR = "/home/qwen35-27b-fp8-v2-model"
REMOTE_METADATA_DIR = "/home/qwen35-27b-fp8-v2-metadata"
REMOTE_PYTHON = "/home/qwen-v2-env/bin/python"


class JarvisLifecycleError(RuntimeError):
    """Raised before an unapproved resource or unverifiable cleanup can proceed."""


@dataclass(frozen=True)
class ResourceOffer:
    gpu: str
    region: str
    spot_rate_inr_per_hour: float
    free_devices: int
    workload_type: str


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class StageGateResult:
    stage: int
    passed: bool
    reasons: tuple[str, ...]


class LiveBudgetController:
    """Reserve cleanup and concurrent attempts against wall-clock GPU cost."""

    def __init__(
        self,
        *,
        stage: int,
        hourly_rate_inr: float,
        stage_cap_inr: float,
        cumulative_cap_inr: float,
        prior_cost_inr: float,
        prior_stage_cost_inr: float = 0.0,
        started_monotonic: float,
        monotonic: Callable[[], float] = time.monotonic,
        cleanup_reserve_seconds: float = DEFAULT_CLEANUP_RESERVE_SECONDS,
        initial_request_seconds: float = 120.0,
    ) -> None:
        values = (
            hourly_rate_inr, stage_cap_inr, cumulative_cap_inr,
            prior_cost_inr, prior_stage_cost_inr,
            cleanup_reserve_seconds, initial_request_seconds,
        )
        if any(value < 0 for value in values) or stage not in {1, 2, 3}:
            raise ValueError("budget controller values are invalid")
        self.stage = stage
        self.hourly_rate = hourly_rate_inr
        self.stage_cap = stage_cap_inr
        self.cumulative_cap = cumulative_cap_inr
        self.prior_cost = prior_cost_inr
        self.prior_stage_cost = prior_stage_cost_inr
        self.started = started_monotonic
        self.monotonic = monotonic
        self.cleanup_reserve = cleanup_reserve_seconds
        self.initial_request = initial_request_seconds
        self._active_reservations = 0
        self._durations: deque[float] = deque(maxlen=64)
        self._lock = threading.Lock()

    def before_attempt(self, _job: object, _retry: bool) -> None:
        with self._lock:
            elapsed = max(0.0, self.monotonic() - self.started)
            request_seconds = self._projected_request_seconds()
            projected_stage_seconds = (
                elapsed
                + self.cleanup_reserve
                + (self._active_reservations + 1) * request_seconds
            )
            projected_stage = self._cost(projected_stage_seconds)
            if self.prior_stage_cost + projected_stage > self.stage_cap:
                raise BudgetStop(f"stage {self.stage} budget cap would be exceeded")
            if self.prior_cost + projected_stage > self.cumulative_cap:
                raise BudgetStop("cumulative budget cap would be exceeded")
            self._active_reservations += 1

    def after_attempt(self, _job: object, _retry: bool, elapsed_seconds: float) -> None:
        with self._lock:
            if self._active_reservations <= 0:
                raise JarvisLifecycleError("attempt completion has no budget reservation")
            self._active_reservations -= 1
            self._durations.append(max(0.0, elapsed_seconds))

    def snapshot(self) -> Mapping[str, object]:
        with self._lock:
            elapsed = max(0.0, self.monotonic() - self.started)
            return {
                "stage": self.stage,
                "hourly_rate_inr": self.hourly_rate,
                "elapsed_billable_seconds": round(elapsed, 6),
                "measured_cost_inr": round(self._cost(elapsed), 6),
                "cleanup_reserve_seconds": self.cleanup_reserve,
                "projected_request_seconds": round(self._projected_request_seconds(), 6),
                "active_reservations": self._active_reservations,
                "stage_cap_inr": self.stage_cap,
                "cumulative_cap_inr": self.cumulative_cap,
                "prior_cost_inr": self.prior_cost,
                "prior_stage_cost_inr": self.prior_stage_cost,
                "total_stage_cost_inr": round(
                    self.prior_stage_cost + self._cost(elapsed), 6,
                ),
                "total_cumulative_cost_inr": round(
                    self.prior_cost + self._cost(elapsed), 6,
                ),
            }

    def assert_cleanup_safe(self) -> None:
        with self._lock:
            elapsed = max(0.0, self.monotonic() - self.started)
            projected_stage = self._cost(elapsed + self.cleanup_reserve)
            if self.prior_stage_cost + projected_stage > self.stage_cap:
                raise BudgetStop(f"stage {self.stage} cleanup reserve is exhausted")
            if self.prior_cost + projected_stage > self.cumulative_cap:
                raise BudgetStop("cumulative cleanup reserve is exhausted")

    def _projected_request_seconds(self) -> float:
        if not self._durations:
            return self.initial_request
        ordered = sorted(self._durations)
        p90 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))]
        return max(1.0, p90 * 1.25)

    def _cost(self, seconds: float) -> float:
        return seconds * self.hourly_rate / 3600


class BudgetWatchdog(AbstractContextManager["BudgetWatchdog"]):
    """Trigger destructive cleanup when wall-clock reserve reaches a hard cap."""

    def __init__(
        self,
        controller: LiveBudgetController,
        on_stop: Callable[[], object],
        *,
        interval_seconds: float = 15.0,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("watchdog interval must be positive")
        self.controller = controller
        self.on_stop = on_stop
        self.interval = interval_seconds
        self._stop = threading.Event()
        self._triggered = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @property
    def triggered(self) -> bool:
        return self._triggered.is_set()

    def __enter__(self) -> "BudgetWatchdog":
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval * 2))
        return False

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.controller.assert_cleanup_safe()
            except BudgetStop:
                self._triggered.set()
                self.on_stop()
                return


class JarvisManager(AbstractContextManager["JarvisManager"]):
    """Own exactly one approved Jarvis instance from creation through destruction."""

    def __init__(
        self,
        *,
        artifact_dir: Path,
        runner: Callable[[Sequence[str]], CommandResult] | None = None,
        clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        remote_artifact_dir: str = "/home/qwen35-27b-fp8-v2-artifacts",
    ) -> None:
        self.artifact_dir = artifact_dir
        self.runner = runner or _run_command
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.monotonic = monotonic
        self.remote_artifact_dir = remote_artifact_dir
        self.machine_id: int | None = None
        self.offer: ResourceOffer | None = None
        self.created_at: str | None = None
        self.destroyed_at: str | None = None
        self.account_balance_before_inr: float | None = None
        self.account_balance_after_inr: float | None = None
        self.started_monotonic: float | None = None
        self.events: list[dict[str, object]] = []
        self._cleaned = False
        self._cleanup_lock = threading.RLock()
        self._prior_handlers: dict[int, object] = {}

    def preflight(self) -> ResourceOffer:
        status = self._json(("jl", "status", "--json"), "Jarvis authentication")
        self.account_balance_before_inr = _account_balance(status)
        gpus = self._json(("jl", "gpus", "--json"), "Jarvis GPU inventory")
        resources = self._json(("jl", "resources", "--json"), "Jarvis resource inventory")
        offer = select_approved_offer(gpus)
        self.events.append({
            "event": "preflight",
            "at": _utc(self.clock()),
            "account_response_sha256": _object_sha(status),
            "gpu_inventory_sha256": _object_sha(gpus),
            "resource_inventory_sha256": _object_sha(resources),
            "selected_offer": asdict(offer),
        })
        self.offer = offer
        return offer

    def create(self) -> int:
        if self.machine_id is not None:
            raise JarvisLifecycleError("Jarvis instance is already recorded")
        offer = self.offer or self.preflight()
        gpu_flag = "H100" if offer.gpu == "H100-80GB" else "RTX-PRO6000"
        response = self._json((
            "jl", "create", "--gpu", gpu_flag,
            "--spot", "--template", "pytorch", "--storage", "100",
            "--num-gpus", "1", "--region", offer.region,
            "--name", "longitudinal-memory-qwen-v2",
            "--yes", "--json",
        ), "Jarvis instance creation")
        machine_id = _machine_id(response)
        self.machine_id = machine_id
        self.created_at = _utc(self.clock())
        self.started_monotonic = self.monotonic()
        self.events.append({
            "event": "created",
            "at": self.created_at,
            "machine_id": machine_id,
            "offer": asdict(offer),
        })
        self._install_signal_handlers()
        return machine_id

    def upload_server(self, *, secret_file: Path, server_script: Path) -> None:
        machine_id = self._require_machine()
        _validate_secret_file(secret_file)
        self._ok(("jl", "upload", str(machine_id), str(server_script), "/home/run_qwen_vllm.sh"), "server upload")
        self._ok(("jl", "upload", str(machine_id), str(secret_file), "/home/.qwen-v2.env"), "secret upload")
        self._ok(("jl", "exec", str(machine_id), "--", "chmod", "600", "/home/.qwen-v2.env"), "remote secret permissions")
        self.events.append({
            "event": "server_files_uploaded",
            "at": _utc(self.clock()),
            "server_script_sha256": sha256(server_script.read_bytes()).hexdigest(),
        })

    def install_vllm(self, revision: str) -> None:
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise JarvisLifecycleError("vLLM revision must be a full commit hash")
        if revision != VLLM_BUILD["revision"]:
            raise JarvisLifecycleError("vLLM revision has no frozen wheel contract")
        machine_id = self._require_machine()
        environment = self.runner((
            "jl", "exec", str(machine_id), "--", "sh", "-lc",
            "python3 --version; python3 -c 'import torch; print(torch.__version__, torch.version.cuda)'; "
            "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader",
        ))
        if environment.returncode != 0:
            raise JarvisLifecycleError("remote Python and CUDA preflight failed")
        self.events.append({
            "event": "install_environment_recorded",
            "at": _utc(self.clock()),
            "environment": _sanitize_diagnostic(environment.stdout),
        })
        python_env = self.runner((
            "jl", "exec", str(machine_id), "--", "sh", "-lc",
            "conda create -y --override-channels -c conda-forge "
            "-p /home/qwen-v2-env python=3.12 pip "
            ">/home/qwen-v2-python-install.log 2>&1",
        ))
        if python_env.returncode != 0:
            diagnostic = self.runner((
                "jl", "exec", str(machine_id), "--", "sh", "-lc",
                "tail -n 160 /home/qwen-v2-python-install.log",
            ))
            self._write_install_failure(revision, diagnostic)
            raise JarvisLifecycleError("isolated Python 3.12 environment creation failed")
        install = self.runner((
            "jl", "exec", str(machine_id), "--", "sh", "-lc",
            f"{REMOTE_PYTHON} -m pip install --upgrade --no-cache-dir "
            f"'{VLLM_BUILD['wheel']}' >/home/qwen-v2-vllm-install.log 2>&1",
        ))
        if install.returncode != 0:
            diagnostic = self.runner((
                "jl", "exec", str(machine_id), "--", "sh", "-lc",
                "tail -n 160 /home/qwen-v2-vllm-install.log",
            ))
            self._write_install_failure(revision, diagnostic)
            self.events.append({
                "event": "vllm_install_failed",
                "at": _utc(self.clock()),
                "revision": revision,
                "diagnostic_sha256": _object_sha(_sanitize_diagnostic(diagnostic.stdout)),
            })
            raise JarvisLifecycleError("pinned vLLM installation failed; diagnostic saved")
        verify = self.runner((
            "jl", "exec", str(machine_id), "--", REMOTE_PYTHON, "-c",
            "import array,sys,vllm; import flashinfer.comm.fd_exchange; "
            "print(sys.version_info[:2], vllm.__version__, array.array[int])",
        ))
        expected = f"(3, 12) {VLLM_BUILD['version']} array.array[int]"
        if verify.returncode != 0 or verify.stdout.strip() != expected:
            self._write_install_failure(revision, verify)
            raise JarvisLifecycleError("pinned vLLM and FlashInfer runtime preflight failed")
        self.events.append({
            "event": "vllm_installed",
            "at": _utc(self.clock()),
            "revision": revision,
            "version": VLLM_BUILD["version"],
            "python": "3.12",
            "wheel_url_sha256": sha256(VLLM_BUILD["wheel"].encode("utf-8")).hexdigest(),
        })

    def start_server(self, *, preflight: bool = False) -> None:
        machine_id = self._require_machine()
        mode = "preflight" if preflight else "production"
        command = (
            "set -a; . /home/.qwen-v2.env; set +a; "
            "chmod 700 /home/run_qwen_vllm.sh; "
            f"nohup setsid /home/run_qwen_vllm.sh {mode} >/home/qwen-v2-server.log 2>&1 "
            "</dev/null & echo $! >/home/qwen-v2-server.pid"
        )
        self._ok(("jl", "exec", str(machine_id), "--", "sh", "-lc", command), "vLLM startup")
        self.events.append({
            "event": "server_started", "at": _utc(self.clock()), "mode": mode,
        })

    def stop_server(self) -> None:
        machine_id = self._require_machine()
        command = (
            "if test -s /home/qwen-v2-server.pid; then "
            "pid=$(cat /home/qwen-v2-server.pid); kill -TERM -- -\"$pid\" 2>/dev/null || true; "
            "for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do "
            "kill -0 \"$pid\" 2>/dev/null || break; sleep 1; done; "
            "kill -KILL -- -\"$pid\" 2>/dev/null || true; fi; "
            "rm -f /home/qwen-v2-server.pid"
        )
        self._ok(("jl", "exec", str(machine_id), "--", "sh", "-lc", command), "vLLM stop")
        self.events.append({"event": "server_stopped", "at": _utc(self.clock())})

    def download_model_metadata(self, *, model_id: str, revision: str) -> Mapping[str, object]:
        """Download config/tokenizer files only for a dummy-weight engine boot."""

        if model_id != "Qwen/Qwen3.5-27B-FP8" or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise JarvisLifecycleError("model metadata identity is outside the frozen contract")
        machine_id = self._require_machine()
        python = (
            "import json,os; from pathlib import Path; "
            "from huggingface_hub import snapshot_download; from transformers import AutoTokenizer; "
            f"path=snapshot_download(repo_id=\"{model_id}\",revision=\"{revision}\","
            f"local_dir=\"{REMOTE_METADATA_DIR}\",token=os.environ[\"HF_TOKEN\"],"
            "ignore_patterns=[\"*.safetensors\",\"*.bin\",\"*.pt\"]); "
            "tok=AutoTokenizer.from_pretrained(path); "
            "files=sorted((p.relative_to(path).as_posix(),p.stat().st_size) "
            "for p in Path(path).rglob(\"*\") if p.is_file()); "
            "assert files and not any(name.endswith((\".safetensors\",\".bin\",\".pt\")) "
            "for name,_ in files); "
            "print(json.dumps({\"file_count\":len(files),\"total_bytes\":sum(x[1] for x in files),"
            "\"tokenizer_class\":type(tok).__name__,\"tokenizer_size\":len(tok)},sort_keys=True))"
        )
        command = (
            "set -a; . /home/.qwen-v2.env; set +a; "
            f"{REMOTE_PYTHON} -c '{python}' >/home/qwen-v2-metadata-download.log 2>&1"
        )
        download = self.runner((
            "jl", "exec", str(machine_id), "--", "sh", "-lc", command,
        ))
        if download.returncode != 0:
            diagnostic = self.runner((
                "jl", "exec", str(machine_id), "--", "sh", "-lc",
                "set -a; . /home/.qwen-v2.env; set +a; "
                "python3 -c 'import os; from pathlib import Path; "
                "text=Path(\"/home/qwen-v2-metadata-download.log\").read_text(errors=\"replace\"); "
                "text=text.replace(os.environ[\"HF_TOKEN\"], \"[REDACTED]\"); "
                "print(text.replace(os.environ[\"VLLM_API_KEY\"], \"[REDACTED]\")[-30000:])'",
            ))
            self._write_remote_failure("model-metadata-failure.json", diagnostic)
            raise JarvisLifecycleError("pinned model metadata or tokenizer preflight failed")
        result = self.runner((
            "jl", "exec", str(machine_id), "--", "sh", "-lc",
            "tail -n 1 /home/qwen-v2-metadata-download.log",
        ))
        try:
            receipt = json.loads(result.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as error:
            raise JarvisLifecycleError("model metadata receipt is invalid") from error
        if not isinstance(receipt, Mapping) or receipt.get("tokenizer_size") != 248077:
            raise JarvisLifecycleError("metadata tokenizer does not match the frozen model")
        self.events.append({
            "event": "model_metadata_downloaded", "at": _utc(self.clock()),
            "model_id": model_id, "revision": revision, **receipt,
        })
        return receipt

    def download_model(self, *, model_id: str, revision: str) -> Mapping[str, object]:
        if model_id != "Qwen/Qwen3.5-27B-FP8" or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise JarvisLifecycleError("model download identity is outside the frozen contract")
        machine_id = self._require_machine()
        python = (
            "import hashlib,json,os; from pathlib import Path; "
            "from huggingface_hub import snapshot_download; from transformers import AutoTokenizer; "
            f"path=snapshot_download(repo_id=\"{model_id}\",revision=\"{revision}\","
            f"local_dir=\"{REMOTE_MODEL_DIR}\",token=os.environ[\"HF_TOKEN\"]); "
            "tok=AutoTokenizer.from_pretrained(path); "
            "files=sorted((p.relative_to(path).as_posix(),p.stat().st_size) "
            "for p in Path(path).rglob(\"*\") if p.is_file()); "
            "payload={\"file_count\":len(files),\"total_bytes\":sum(x[1] for x in files),"
            "\"manifest_sha256\":hashlib.sha256(json.dumps(files,separators=(\",\",\":\")).encode()).hexdigest(),"
            "\"tokenizer_class\":type(tok).__name__,\"tokenizer_size\":len(tok)}; "
            "print(json.dumps(payload,sort_keys=True))"
        )
        command = (
            "set -a; . /home/.qwen-v2.env; set +a; "
            f"{REMOTE_PYTHON} -c '{python}' >/home/qwen-v2-model-download.log 2>&1"
        )
        download = self.runner((
            "jl", "exec", str(machine_id), "--", "sh", "-lc", command,
        ))
        if download.returncode != 0:
            diagnostic = self.runner((
                "jl", "exec", str(machine_id), "--", "sh", "-lc",
                "set -a; . /home/.qwen-v2.env; set +a; "
                "python3 -c 'import os; from pathlib import Path; "
                "text=Path(\"/home/qwen-v2-model-download.log\").read_text(errors=\"replace\"); "
                "text=text.replace(os.environ[\"HF_TOKEN\"], \"[REDACTED]\"); "
                "print(text.replace(os.environ[\"VLLM_API_KEY\"], \"[REDACTED]\")[-30000:])'",
            ))
            self._write_remote_failure("model-download-failure.json", diagnostic)
            raise JarvisLifecycleError("pinned model download or tokenizer smoke test failed")
        result = self.runner((
            "jl", "exec", str(machine_id), "--", "sh", "-lc",
            "tail -n 1 /home/qwen-v2-model-download.log",
        ))
        try:
            receipt = json.loads(result.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as error:
            raise JarvisLifecycleError("model download receipt is invalid") from error
        if not isinstance(receipt, Mapping) or receipt.get("tokenizer_size") != 248077:
            raise JarvisLifecycleError("downloaded tokenizer does not match the frozen model")
        self.events.append({
            "event": "model_downloaded",
            "at": _utc(self.clock()),
            "model_id": model_id,
            "revision": revision,
            **receipt,
        })
        return receipt

    def server_alive(self) -> bool:
        machine_id = self._require_machine()
        result = self.runner((
            "jl", "exec", str(machine_id), "--", "sh", "-lc",
            "test -s /home/qwen-v2-server.pid && kill -0 \"$(cat /home/qwen-v2-server.pid)\"",
        ))
        if result.returncode == 0:
            return True
        self._collect_server_failure()
        return False

    def endpoint(self) -> str:
        response = self._json(("jl", "get", str(self._require_machine()), "--json"), "Jarvis instance details")
        values = [value for value in _string_values(response) if value.startswith("https://")]
        endpoint = next((value.rstrip("/") for value in values if "6006" in value), None)
        if endpoint is None:
            endpoints = response.get("endpoints") if isinstance(response, Mapping) else None
            if isinstance(endpoints, list) and endpoints:
                first = endpoints[0]
                endpoint = str(first.get("url") if isinstance(first, Mapping) else first).rstrip("/")
        if not endpoint or not endpoint.startswith("https://"):
            raise JarvisLifecycleError("approved port 6006 endpoint is unavailable")
        return endpoint

    def collect_remote_receipt(self) -> Mapping[str, object]:
        machine_id = self._require_machine()
        commands = {
            "gpu": "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader",
            "cuda": f"{REMOTE_PYTHON} -c 'import torch; print(torch.version.cuda)'",
            "vllm": f"{REMOTE_PYTHON} -c 'import vllm; print(vllm.__version__)'",
            "vllm_install": f"{REMOTE_PYTHON} -m pip freeze | grep '^vllm'",
        }
        values = {}
        for name, command in commands.items():
            result = self.runner(("jl", "exec", str(machine_id), "--", "sh", "-lc", command))
            if result.returncode != 0:
                raise JarvisLifecycleError(f"remote {name} provenance check failed")
            values[name] = result.stdout.strip()[:500]
        self.events.append({
            "event": "remote_provenance_recorded",
            "at": _utc(self.clock()),
            "receipt_sha256": _object_sha(values),
        })
        return values

    def cleanup(self) -> Mapping[str, object]:
        with self._cleanup_lock:
            return self._cleanup_locked()

    def _cleanup_locked(self) -> Mapping[str, object]:
        if self._cleaned:
            self._restore_signal_handlers()
            return self.receipt()
        machine_id = self.machine_id
        if machine_id is None:
            self._cleaned = True
            self.destroyed_at = _utc(self.clock())
            self._write_receipt()
            return self.receipt()
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        try:
            download = self.runner((
                "jl", "download", str(machine_id), self.remote_artifact_dir,
                str(self.artifact_dir / "remote-checkpoints"), "-r",
            ))
        except Exception:
            download = CommandResult(1, "", "download_exception")
        self.events.append({
            "event": "checkpoint_download",
            "at": _utc(self.clock()),
            "succeeded": download.returncode == 0,
            "failure_code": None if download.returncode == 0 else "download_failed",
        })
        try:
            destroy = self.runner(("jl", "destroy", str(machine_id), "--yes", "--json"))
        except Exception:
            destroy = CommandResult(1, "", "destroy_exception")
        self.events.append({
            "event": "destroy_requested",
            "at": _utc(self.clock()),
            "succeeded": destroy.returncode == 0,
        })
        if destroy.returncode != 0:
            self._write_receipt()
            raise JarvisLifecycleError("Jarvis destroy command failed")
        try:
            listing = self._json(("jl", "list", "--json"), "post-destroy instance list")
        except Exception:
            self._write_receipt()
            raise
        if machine_id in _all_machine_ids(listing):
            self._write_receipt()
            raise JarvisLifecycleError("destroyed Jarvis instance remains in instance list")
        get_result = self.runner(("jl", "get", str(machine_id), "--json"))
        if get_result.returncode == 0:
            self._write_receipt()
            raise JarvisLifecycleError("destroyed Jarvis instance is still retrievable")
        status_result = self.runner(("jl", "status", "--json"))
        if status_result.returncode == 0:
            try:
                self.account_balance_after_inr = _account_balance(json.loads(status_result.stdout))
            except (json.JSONDecodeError, JarvisLifecycleError):
                self.events.append({
                    "event": "billing_reconciliation_failed",
                    "at": _utc(self.clock()),
                })
        self.destroyed_at = _utc(self.clock())
        self.events.append({"event": "destruction_verified", "at": self.destroyed_at})
        self._cleaned = True
        self._restore_signal_handlers()
        self._write_receipt()
        return self.receipt()

    def mirror_artifacts(self, local_path: Path, remote_name: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", remote_name):
            raise JarvisLifecycleError("remote artifact name is invalid")
        machine_id = self._require_machine()
        self._ok((
            "jl", "exec", str(machine_id), "--", "mkdir", "-p", self.remote_artifact_dir,
        ), "remote artifact directory creation")
        self._ok((
            "jl", "upload", str(machine_id), str(local_path),
            f"{self.remote_artifact_dir}/{remote_name}",
        ), "artifact mirror upload")
        self.events.append({
            "event": "artifacts_mirrored",
            "at": _utc(self.clock()),
            "remote_name": remote_name,
        })

    def receipt(self) -> Mapping[str, object]:
        elapsed = None
        cost = None
        if self.started_monotonic is not None and self.offer is not None:
            elapsed = max(0.0, self.monotonic() - self.started_monotonic)
            cost = elapsed * self.offer.spot_rate_inr_per_hour / 3600
        observed_spend = None
        if self.account_balance_before_inr is not None and self.account_balance_after_inr is not None:
            observed_spend = max(
                0.0, self.account_balance_before_inr - self.account_balance_after_inr,
            )
        return {
            "schema_version": "qwen_jarvis_lifecycle_v2",
            "series_id": SERIES_ID,
            "machine_id": self.machine_id,
            "offer": None if self.offer is None else asdict(self.offer),
            "created_at": self.created_at,
            "destroyed_at": self.destroyed_at,
            "destruction_verified": any(event["event"] == "destruction_verified" for event in self.events),
            "elapsed_billable_seconds": None if elapsed is None else round(elapsed, 6),
            "gpu_cost_inr": None if cost is None else round(cost, 6),
            "account_balance_before_inr": self.account_balance_before_inr,
            "account_balance_after_inr": self.account_balance_after_inr,
            "observed_account_spend_inr": (
                None if observed_spend is None else round(observed_spend, 6)
            ),
            "events": tuple(self.events),
        }

    def __enter__(self) -> "JarvisManager":
        self.create()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.cleanup()
        return False

    def _install_signal_handlers(self) -> None:
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._prior_handlers[signum] = signal.getsignal(signum)

            def handle(received, _frame, *, manager=self):
                manager.cleanup()
                raise SystemExit(128 + received)

            signal.signal(signum, handle)

    def _restore_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for signum, handler in self._prior_handlers.items():
            signal.signal(signum, handler)
        self._prior_handlers.clear()

    def _write_receipt(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        path = self.artifact_dir / "lifecycle.json"
        payload = json.dumps(self.receipt(), indent=2, sort_keys=True) + "\n"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)

    def _write_install_failure(self, revision: str, result: CommandResult) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        path = self.artifact_dir / "install-failure.json"
        payload = {
            "schema_version": "qwen_vllm_install_failure_v1",
            "series_id": SERIES_ID,
            "revision": revision,
            "returncode": result.returncode,
            "diagnostic": _sanitize_diagnostic(result.stdout or result.stderr),
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)

    def _collect_server_failure(self) -> None:
        machine_id = self._require_machine()
        command = (
            "set -a; . /home/.qwen-v2.env; set +a; "
            "python3 -c 'import os; from pathlib import Path; "
            "text=Path(\"/home/qwen-v2-server.log\").read_text(errors=\"replace\"); "
            "print(text.replace(os.environ[\"VLLM_API_KEY\"], \"[REDACTED]\")[-12000:])'"
        )
        diagnostic = self.runner((
            "jl", "exec", str(machine_id), "--", "sh", "-lc", command,
        ))
        value = self._write_remote_failure("server-failure.json", diagnostic)
        self.events.append({
            "event": "server_failure_recorded",
            "at": _utc(self.clock()),
            "diagnostic_sha256": _object_sha(value),
        })

    def _write_remote_failure(self, name: str, result: CommandResult) -> str:
        value = _sanitize_diagnostic(result.stdout or result.stderr)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        path = self.artifact_dir / name
        payload = {
            "schema_version": "qwen_remote_failure_v1",
            "series_id": SERIES_ID,
            "returncode": result.returncode,
            "diagnostic": value,
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
        return value

    def _require_machine(self) -> int:
        if self.machine_id is None:
            raise JarvisLifecycleError("Jarvis instance has not been created")
        return self.machine_id

    def _json(self, command: Sequence[str], label: str) -> Mapping[str, object] | list[object]:
        result = self.runner(command)
        if result.returncode != 0:
            raise JarvisLifecycleError(f"{label} failed")
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise JarvisLifecycleError(f"{label} returned invalid JSON") from error
        if isinstance(value, Mapping) and value.get("error"):
            raise JarvisLifecycleError(f"{label} returned an error")
        if not isinstance(value, (Mapping, list)):
            raise JarvisLifecycleError(f"{label} response shape changed")
        return value

    def _ok(self, command: Sequence[str], label: str) -> None:
        if self.runner(command).returncode != 0:
            raise JarvisLifecycleError(f"{label} failed")


def select_approved_offer(inventory: object) -> ResourceOffer:
    offers = tuple(_offers(inventory))
    h100 = [
        item for item in offers
        if item.gpu == "H100-80GB" and item.region == "IN2"
        and item.free_devices > 0 and item.spot_rate_inr_per_hour <= H100_MAX_RATE
    ]
    if h100:
        return min(h100, key=lambda item: item.spot_rate_inr_per_hour)
    rtx = [
        item for item in offers
        if item.gpu == "RTX-PRO6000-96GB" and item.region == "IN1"
        and item.free_devices > 0 and item.spot_rate_inr_per_hour <= RTX_MAX_RATE
    ]
    if rtx:
        return min(rtx, key=lambda item: item.spot_rate_inr_per_hour)
    raise JarvisLifecycleError("no approved spot GPU is available below its price ceiling")


def stage_one_gate(
    manifest: Mapping[str, object],
    *,
    projected_remaining_cost_inr: float,
    exact_model: bool,
    no_oom: bool,
) -> StageGateResult:
    reasons = []
    if manifest.get("planned_request_count") != 12 or manifest.get("successful_count") != 12:
        reasons.append("compatibility_responses_not_12_of_12")
    if not exact_model:
        reasons.append("model_alias_or_revision_mismatch")
    if not no_oom:
        reasons.append("oom_detected")
    if projected_remaining_cost_inr * 1.2 > 1300:
        reasons.append("remaining_run_exceeds_budget_with_reserve")
    return StageGateResult(1, not reasons, tuple(reasons))


def stage_two_gate(
    extraction: Mapping[str, object],
    answers: Mapping[str, object],
    judge: Mapping[str, object],
    *,
    batch_valid_rates: Mapping[str, float],
    projected_stage_three_cost_inr: float,
) -> StageGateResult:
    reasons = []
    if extraction.get("planned_request_count") != 20 or extraction.get("successful_count") != 20:
        reasons.append("development_extraction_not_20_of_20")
    total = sum(int(item.get("planned_request_count", 0)) for item in (extraction, answers, judge))
    valid = sum(int(item.get("successful_count", 0)) for item in (extraction, answers, judge))
    if total != 930 or valid < 884:
        reasons.append("development_structural_validity_gate_failed")
    if any(value < 0.9 for value in batch_valid_rates.values()):
        reasons.append("task_or_baseline_batch_below_90_percent")
    if projected_stage_three_cost_inr * 1.2 > 1000:
        reasons.append("stage_three_exceeds_budget_with_reserve")
    return StageGateResult(2, not reasons, tuple(reasons))


def load_local_secrets(path: Path) -> Mapping[str, str]:
    _validate_secret_file(path)
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            raise JarvisLifecycleError("local secret file has an invalid line")
        name, value = stripped.split("=", 1)
        values[name.strip()] = value.strip().strip("\"").strip("'")
    required = {"HF_TOKEN", "VLLM_API_KEY"}
    if not required.issubset(values) or any(not values[name] for name in required):
        raise JarvisLifecycleError("local secret file is missing required values")
    return {name: values[name] for name in required}


def _offers(value: object):
    for row in _mappings(value):
        gpu_value = _first(row, "gpu", "gpu_type", "gpu_name", "name")
        region_value = _first(row, "region", "region_code", "location")
        workload = str(_first(row, "workload_type", "type", default="container")).casefold()
        price_value = _first(row, "spot_price", "spot_price_inr", "spot_rate", "spot")
        free_value = _first(row, "num_free_devices", "free_devices", "available", default=0)
        gpu = _canonical_gpu(str(gpu_value)) if gpu_value is not None else None
        region = _canonical_region(str(region_value)) if region_value is not None else None
        price = _number(price_value)
        free = _integer(free_value)
        if gpu and region and price is not None and free is not None and workload in {"container", "none", ""}:
            yield ResourceOffer(gpu, region, price, free, "container")


def _mappings(value: object):
    if isinstance(value, Mapping):
        yield value
        for nested in value.values():
            yield from _mappings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _mappings(nested)


def _canonical_gpu(value: str) -> str | None:
    compact = re.sub(r"[^A-Z0-9]", "", value.upper())
    if "H100" in compact and "H200" not in compact:
        return "H100-80GB"
    if "RTX" in compact and "PRO6000" in compact:
        return "RTX-PRO6000-96GB"
    return None


def _canonical_region(value: str) -> str | None:
    upper = value.upper()
    if "IN2" in upper or "NOIDA" in upper:
        return "IN2"
    if "IN1" in upper or "CHENNAI" in upper:
        return "IN1"
    return None


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"\d+(?:\.\d+)?", value.replace(",", ""))
        return float(match.group()) if match else None
    if isinstance(value, Mapping):
        return _number(_first(value, "inr", "price", "hourly", "value"))
    return None


def _integer(value: object) -> int | None:
    number = _number(value)
    return int(number) if number is not None and number >= 0 else None


def _first(mapping: Mapping[str, object], *names: str, default: object = None) -> object:
    return next((mapping[name] for name in names if name in mapping), default)


def _machine_id(value: object) -> int:
    for row in _mappings(value):
        candidate = _first(row, "machine_id", "instance_id", "id")
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            return candidate
        if isinstance(candidate, str) and candidate.isdigit():
            return int(candidate)
    raise JarvisLifecycleError("Jarvis create response has no machine ID")


def _account_balance(value: object) -> float:
    if isinstance(value, Mapping):
        balance = value.get("balance")
        if isinstance(balance, Mapping):
            candidate = balance.get("balance")
            if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                return float(candidate)
    raise JarvisLifecycleError("Jarvis account response has no numeric balance")


def _all_machine_ids(value: object) -> set[int]:
    result = set()
    for row in _mappings(value):
        candidate = _first(row, "machine_id", "instance_id", "id")
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            result.add(candidate)
        elif isinstance(candidate, str) and candidate.isdigit():
            result.add(int(candidate))
    return result


def _string_values(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for nested in value.values():
            yield from _string_values(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _string_values(nested)


def _validate_secret_file(path: Path) -> None:
    if not path.is_file():
        raise JarvisLifecycleError("local Qwen secret file is missing")
    if path.name != ".qwen-secrets.env":
        raise JarvisLifecycleError("secret file must use the ignored .qwen-secrets.env name")
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise JarvisLifecycleError("local Qwen secret file permissions must be 0600")


def _run_command(command: Sequence[str]) -> CommandResult:
    try:
        completed = subprocess.run(
            command, text=True, capture_output=True, check=False, timeout=1800,
        )
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)
    except subprocess.TimeoutExpired:
        return CommandResult(124, "", "command_timeout")


def _utc(value: datetime) -> str:
    if value.utcoffset() is None:
        raise JarvisLifecycleError("lifecycle clock must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _object_sha(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _sanitize_diagnostic(value: str) -> str:
    sanitized = re.sub(r"(?i)(authorization|token|api[_-]?key|password)(\s*[:=]\s*)\S+", r"\1\2[REDACTED]", value)
    sanitized = re.sub(r"([?&](?:token|key|auth|password)=)[^&\s]+", r"\1[REDACTED]", sanitized, flags=re.IGNORECASE)
    return sanitized.strip()[-12000:]
