"""Verify the immutable evidence from the interrupted OpenAI Step 10.3 run."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

from .frozen_run_contracts import FrozenRunError, parse_json_bytes


MANIFEST_PATH = Path("results/evaluation/openai-step10.3-interrupted-v1/manifest.json")
MANIFEST_SHA256 = "57e34b30ceb0100a75cefa49175228fd386e89a205d27588c7104fac88061f75"


def verify_interrupted_openai_history(
    *,
    repo_root: str | Path = ".",
    manifest_path: str | Path = MANIFEST_PATH,
) -> Mapping[str, object]:
    """Verify the recovery manifest and every historical artifact it binds."""

    root = Path(repo_root).resolve()
    path = Path(manifest_path)
    if not path.is_absolute():
        path = root / path
    if not path.is_file() or _sha(path) != MANIFEST_SHA256:
        raise FrozenRunError("interrupted OpenAI recovery manifest changed")
    manifest = parse_json_bytes(path.read_bytes(), location="OpenAI recovery manifest")
    if (
        manifest.get("schema_version") != "openai_step10_3_recovery_v1"
        or manifest.get("status") != "interrupted_failed_not_scored"
        or manifest.get("historical_openai_spend_usd") != "0.4399284"
        or manifest.get("new_openai_request_count_since_recovery_goal") != 0
        or manifest.get("new_openai_cost_usd_since_recovery_goal") != "0.0000000"
    ):
        raise FrozenRunError("interrupted OpenAI recovery summary changed")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 12:
        raise FrozenRunError("interrupted OpenAI artifact inventory changed")
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise FrozenRunError("interrupted OpenAI artifact entry changed")
        relative = artifact.get("path")
        digest = artifact.get("sha256")
        size = artifact.get("bytes")
        if not isinstance(relative, str) or not isinstance(digest, str) or not isinstance(size, int):
            raise FrozenRunError("interrupted OpenAI artifact binding changed")
        artifact_path = (root / relative).resolve()
        if (
            root not in artifact_path.parents
            or not artifact_path.is_file()
            or artifact_path.stat().st_size != size
            or _sha(artifact_path) != digest
        ):
            raise FrozenRunError(f"interrupted OpenAI artifact changed: {relative}")
    return manifest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
