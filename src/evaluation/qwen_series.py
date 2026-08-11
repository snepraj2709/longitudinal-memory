"""Frozen configuration, budget gates, and checkpoints for the Qwen series."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping


SERIES_ID = "qwen35-27b-fp8-v1"
CONFIG_PATH = Path("configs/evaluation/qwen35_27b_fp8_v1.json")


class BudgetStop(RuntimeError):
    """Raised before a request could exceed a stage or cumulative budget."""


@dataclass(frozen=True)
class BudgetGate:
    stage: int
    stage_cap_inr: float
    cumulative_cap_inr: float
    spent_stage_inr: float
    spent_total_inr: float

    def assert_can_spend(self, projected_increment_inr: float) -> None:
        if projected_increment_inr < 0:
            raise ValueError("projected_increment_inr cannot be negative")
        if self.spent_stage_inr + projected_increment_inr > self.stage_cap_inr:
            raise BudgetStop(f"stage {self.stage} budget cap would be exceeded")
        if self.spent_total_inr + projected_increment_inr > self.cumulative_cap_inr:
            raise BudgetStop("cumulative budget cap would be exceeded")


def load_series_config(repo_root: Path) -> dict[str, object]:
    path = repo_root / CONFIG_PATH
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("series_id") != SERIES_ID:
        raise ValueError("Qwen series configuration is invalid")
    return value


def config_sha256(repo_root: Path) -> str:
    return sha256((repo_root / CONFIG_PATH).read_bytes()).hexdigest()


def stage_three_is_allowed(
    *,
    valid_responses: int,
    total_responses: int,
    projected_run_cost_inr: float,
    remaining_budget_inr: float,
    minimum_valid_rate: float = 0.95,
    reserve_fraction: float = 0.2,
) -> bool:
    if total_responses <= 0 or projected_run_cost_inr < 0 or remaining_budget_inr < 0:
        return False
    valid_rate = valid_responses / total_responses
    projected_with_reserve = projected_run_cost_inr * (1 + reserve_fraction)
    return valid_rate >= minimum_valid_rate and projected_with_reserve <= remaining_budget_inr


def append_checkpoint(path: Path, record: Mapping[str, object]) -> None:
    """Append one fsynced JSONL response record without rewriting prior rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(record), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        import os

        os.fsync(handle.fileno())


def write_immutable_json(path: Path, value: Mapping[str, object]) -> None:
    """Create an artifact once; refuse accidental replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise FileExistsError(f"immutable artifact already exists: {path}") from error
