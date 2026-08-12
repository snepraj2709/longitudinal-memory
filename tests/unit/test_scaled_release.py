from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import unittest

from evaluation.scaled_release import (
    CAPABILITIES,
    GOLD_PATHS,
    MANIFEST_PATH,
    ORACLE_PATHS,
    PREDICTION_PATHS,
    RELEASE_DATA_PATHS,
    REVIEW_PATHS,
    RUNTIME_PATHS,
    SCHEMA_PATHS,
    USER_IDS,
    ScaledReleaseError,
    _validate_evidence,
    _validate_runtime,
    load_scaled_runtime,
    validate_scaled_release,
)
from evaluation.run_config import dataset_sha256


class ScaledReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[2]

    def test_release_blocks_false_review_approval(self) -> None:
        with self.assertRaises(ScaledReleaseError) as raised:
            validate_scaled_release(self.repo_root)

        errors = "\n".join(raised.exception.errors)
        self.assertIn("gold claims.review_status must be approved/implementation_reviewed", errors)
        self.assertIn("gold qa.review_status must be approved/implementation_reviewed", errors)
        self.assertIn("review queue evidence.status must be approved/resolved", errors)
        self.assertIn("pending_human_review=730", errors)

    def test_runtime_loader_is_user_scoped(self) -> None:
        for user_id in USER_IDS:
            runtime = load_scaled_runtime(self.repo_root, user_id)
            self.assertEqual(runtime.user["user_id"], user_id)
            self.assertEqual(len(runtime.sources), 10)
            self.assertEqual(len(runtime.qa), 50)
            self.assertEqual(len(runtime.summaries), 5)
            self.assertEqual(len(runtime.interactive), 2)
            records = runtime.sources + runtime.qa + runtime.summaries + runtime.interactive
            self.assertTrue(all(item["user_id"] == user_id for item in records))

    def test_runtime_loader_never_opens_oracle_gold_review_or_manifest(self) -> None:
        opened: list[Path] = []
        original_open = Path.open

        def tracked_open(path: Path, *args: object, **kwargs: object):
            opened.append(path.resolve())
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", new=tracked_open):
            load_scaled_runtime(self.repo_root, "user_001")

        forbidden = (MANIFEST_PATH,) + ORACLE_PATHS + GOLD_PATHS + PREDICTION_PATHS + REVIEW_PATHS + SCHEMA_PATHS
        forbidden_paths = {(self.repo_root / path).resolve() for path in forbidden}
        self.assertTrue(opened)
        self.assertTrue(all(path not in forbidden_paths for path in opened))
        self.assertEqual(
            {path for path in opened},
            {(self.repo_root / path).resolve() for path in RUNTIME_PATHS},
        )

    def test_runtime_loader_rejects_unknown_user(self) -> None:
        with self.assertRaises(ScaledReleaseError):
            load_scaled_runtime(self.repo_root, "user_999")

    def test_runtime_validation_rejects_a_scorer_field(self) -> None:
        runtime_records = [
            [json.loads(line) for line in (self.repo_root / path).read_text(encoding="utf-8").splitlines()]
            for path in RUNTIME_PATHS
        ]
        users, sources, qa, summaries, interactive = deepcopy(runtime_records)
        qa[0]["reference_answer"] = "leaked scorer value"
        errors: list[str] = []

        _validate_runtime(users, sources, qa, summaries, interactive, errors)

        self.assertTrue(any("reference_answer" in error and "leaks scorer fields" in error for error in errors))

    def test_evidence_validation_rejects_cross_user_reference(self) -> None:
        sources = [
            json.loads(line)
            for line in (self.repo_root / RUNTIME_PATHS[1]).read_text(encoding="utf-8").splitlines()
        ]
        source = next(item for item in sources if item["user_id"] == "user_002")
        evidence = [{
            "source_id": source["source_id"],
            "message_id": source["messages"][0]["message_id"],
            "quote": source["messages"][0]["text"],
        }]
        errors: list[str] = []

        _validate_evidence(
            evidence,
            "test case",
            "user_001",
            None,
            {item["source_id"]: item for item in sources},
            errors,
        )

        self.assertTrue(any("crosses the user boundary" in error for error in errors))

    def test_manifest_hashes_every_release_file(self) -> None:
        manifest = json.loads((self.repo_root / MANIFEST_PATH).read_text(encoding="utf-8"))
        declared = {item["path"]: item for item in manifest["files"]}

        self.assertEqual(set(declared), {path.as_posix() for path in RELEASE_DATA_PATHS})
        for relative in RELEASE_DATA_PATHS:
            self.assertEqual(
                declared[relative.as_posix()]["sha256"],
                hashlib.sha256((self.repo_root / relative).read_bytes()).hexdigest(),
            )
        expected = dataset_sha256(
            self.repo_root,
            tuple(path.as_posix() for path in RELEASE_DATA_PATHS),
        )
        self.assertEqual(manifest["dataset_sha256"], expected)

    def test_review_queue_snapshot_preserves_preapproval_state(self) -> None:
        for path in REVIEW_PATHS:
            records = [json.loads(line) for line in (self.repo_root / path).read_text(encoding="utf-8").splitlines()]
            self.assertTrue(records)
            self.assertTrue(all(item["status"] == "pending_human_review" for item in records))


if __name__ == "__main__":
    unittest.main()
