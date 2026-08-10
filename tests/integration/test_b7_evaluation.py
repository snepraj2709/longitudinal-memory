from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from abstention.b7_evaluation import (
    BOUNDARY_RULING_SHA256,
    BOUNDARY_RULING_VERSION,
    DATASET_ROOT,
    RESULT_ROOT,
    SAFE_DISCOVERY_INCLUDED_MODULES,
    SAFE_DISCOVERY_EXCLUDED_MODULES,
    SCALED_DATASET_SHA256,
    SCALED_MANIFEST_SHA256,
    SOURCE_REFERENCE,
    B7EvaluationError,
    execute_b7_evaluation,
    prepare_b7_evaluation_reference,
    verify_b7_evaluation,
)
from abstention.b7_evaluation_contracts import canonical_json_bytes
from abstention.comparison_runtime import verify_comparison_runtime
import abstention.b7_evaluation as evaluation_module


ROOT = Path(__file__).resolve().parents[2]
STEP94_MANIFEST_LIVE_ADAPTER_SHA256 = {
    "tests/integration/test_answer_quality_evaluation.py": "638b7730c0ed87ffa76de95496c20979d5baab53a7a17e6791ffd1fb7c895822",
    "tests/integration/test_answerability.py": "2e7694dfc2c188b3571c39741e891fdcfcc10222d8bd7a119b237e516ee9f40f",
    "tests/integration/test_b6_b7_comparison_prerequisite.py": "e10064e56a2e1c3d13290a0eb378be1dfa0ac9ca0ca8f712119c4241a78076c3",
    "tests/integration/test_interactive_answering_v2.py": "ca0341b73d3f6aa1dc5a302f4ce5c41f055bf0641476322c1e5981628514d2b4",
    "tests/integration/test_memory_answer.py": "1802d2bc6353244c5f3fd720f5920794a6e5ee4733d25cb012fea503551f5b82",
}
STEP102_LIVE_ADAPTER_SHA256 = {
    "tests/integration/test_answer_quality_evaluation.py": "3171620fbdb36a4d7ee0ebc4601187031a0020e7517ec5dee220e277426127b5",
    "tests/integration/test_answerability.py": "3174f90a554402f73a03b22d0f5acd13477181f36a67b295e328feb6684229fd",
    "tests/integration/test_b6_b7_comparison_prerequisite.py": "8147e7e18decc4dc46866fca5740b1eaa4cdab9759e025c4690c1c2197de1db2",
    "tests/integration/test_interactive_answering_v2.py": "048e2b5deb0da7ffc26003946f1a8b20001810976eb1f7321441d49712812493",
    "tests/integration/test_memory_answer.py": "ad7ac20d156270d95dd19e046946dd6cc50941c8427d0ad836dbc5e0ac27ed66",
}


class B7EvaluationIntegrationTests(unittest.TestCase):
    def test_prerequisite_and_checked_release_verify(self) -> None:
        checkpoint = verify_comparison_runtime(repo_root=ROOT)
        self.assertEqual((checkpoint.case_count, checkpoint.pair_count), (4, 4))
        checked = ROOT / RESULT_ROOT
        if not checked.exists():
            self.skipTest("B7 evaluation release has not been frozen yet")
        live = {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in STEP102_LIVE_ADAPTER_SHA256
        }
        self.assertEqual(live, STEP102_LIVE_ADAPTER_SHA256)
        original_sha = evaluation_module._sha
        frozen_paths = {
            (ROOT / path).resolve(): digest
            for path, digest in STEP94_MANIFEST_LIVE_ADAPTER_SHA256.items()
        }

        def frozen_adapter_sha(path):
            resolved = Path(path).resolve()
            if resolved in frozen_paths:
                return frozen_paths[resolved]
            return original_sha(path)

        with patch.object(evaluation_module, "_sha", side_effect=frozen_adapter_sha):
            checks = verify_b7_evaluation(checked, repo_root=ROOT)
        self.assertEqual((checks.case_count, checks.per_case_count), (4, 8))
        self.assertFalse(checks.phase9_exit)

    def test_reference_copy_requests_exactly_four_rows_and_never_five(self) -> None:
        calls = []

        def guarded_reader(path):
            self.assertEqual(path, ROOT / SOURCE_REFERENCE)
            rows = []
            with path.open("rb") as stream:
                for index in range(4):
                    calls.append(index + 1)
                    rows.append(json.loads(stream.readline()))
            return tuple(rows)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dataset"
            rows = prepare_b7_evaluation_reference(output, repo_root=ROOT, record_reader=guarded_reader)
            self.assertEqual(calls, [1, 2, 3, 4])
            self.assertEqual([item.expected_decision for item in rows], ["answerable", "answerable", "answerable", "abstain"])
            expected = (output / "reference/expected.jsonl").read_text()
            self.assertNotIn("expected_behaviours", expected)
            self.assertNotIn("required_claim_ids", expected)
            self.assertNotIn("exact_evidence_tuples", expected)

    def test_two_clean_scorer_runs_are_byte_identical_and_use_no_runtime_or_provider(self) -> None:
        original_open = Path.open
        forbidden = (
            "data/scaled-v1/", "data/pilot/", "oracle", "review_queues", ".env",
            "results/abstention/interactive-answering-development-runtime-v2/packages.jsonl",
            "results/abstention/interactive-answering-development-runtime-v2/retrieval-results.jsonl",
            "data/abstention/interactive-answering-development-v2/gold/behaviours.jsonl",
        )

        def guarded_open(path, *args, **kwargs):
            try:
                relative = path.resolve().relative_to(ROOT).as_posix()
            except ValueError:
                return original_open(path, *args, **kwargs)
            mode = kwargs.get("mode", args[0] if args else "r")
            if ("r" in mode or "+" in mode) and any(item in relative for item in forbidden):
                raise AssertionError(f"prohibited scorer read: {relative}")
            return original_open(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            with patch.object(Path, "open", guarded_open):
                execute_b7_evaluation(first, repo_root=ROOT)
                execute_b7_evaluation(second, repo_root=ROOT)
            self.assertEqual(
                {path.name: path.read_bytes() for path in first.iterdir()},
                {path.name: path.read_bytes() for path in second.iterdir()},
            )

    def test_exact_metrics_counts_and_pair_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release"
            deltas = execute_b7_evaluation(output, repo_root=ROOT)
            checks = verify_b7_evaluation(output, repo_root=ROOT)
            scorecard = json.loads((output / "scorecard.json").read_bytes())
        self.assertEqual(len(deltas), 4)
        self.assertTrue(all(not item.gate_changed_output for item in deltas))
        self.assertEqual(checks.b6_unnecessary_abstention_count, 3)
        self.assertEqual(checks.b7_unnecessary_abstention_count, 3)
        metrics = {(item["baseline_id"], item["metric_name"]): item for item in scorecard["metrics"]}
        self.assertEqual(metrics[("B6", "abstention_precision")]["value"], "0.250000")
        self.assertEqual(metrics[("B7", "unnecessary_abstention_rate")]["value"], "1.000000")
        self.assertIsNone(metrics[(None, "selective_risk_delta")]["value"])

    def test_nonempty_outputs_are_refused_before_any_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release"
            output.mkdir()
            (output / "occupied").write_text("x")
            with patch("abstention.b7_evaluation._verify_prerequisite", side_effect=AssertionError("read")):
                with self.assertRaisesRegex(B7EvaluationError, "must be empty"):
                    execute_b7_evaluation(output, repo_root=ROOT)

    def test_deep_verifier_rejects_coordinated_output_rehash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release"
            execute_b7_evaluation(output, repo_root=ROOT)
            target = output / "per-case.jsonl"
            rows = [json.loads(line) for line in target.read_text().splitlines()]
            rows[0]["prediction_sha256"] = "0" * 64
            rows[0]["per_case_id"] = hashlib.sha256(canonical_json_bytes({
                key: value for key, value in rows[0].items() if key != "per_case_id"
            })).hexdigest()
            target.write_bytes(b"".join(canonical_json_bytes(item) for item in rows))
            manifest = json.loads((output / "manifest.json").read_bytes())
            for item in manifest["artifacts"]:
                if item[0] == "per-case.jsonl":
                    item[1] = hashlib.sha256(target.read_bytes()).hexdigest()
            (output / "manifest.json").write_bytes(canonical_json_bytes(manifest))
            with self.assertRaises(B7EvaluationError):
                verify_b7_evaluation(output, repo_root=ROOT)

    def test_dataset_and_prerequisite_hash_tamper_are_rejected(self) -> None:
        checked = ROOT / RESULT_ROOT
        if not checked.exists():
            self.skipTest("B7 evaluation release has not been frozen yet")
        with patch("abstention.b7_evaluation.PREREQUISITE_CHECKPOINT_SHA256", "0" * 64):
            with self.assertRaisesRegex(B7EvaluationError, "checkpoint changed"):
                verify_b7_evaluation(checked, repo_root=ROOT)

    def test_boundary_ruling_and_safe_test_evidence_are_exact(self) -> None:
        manifest = json.loads((ROOT / RESULT_ROOT / "manifest.json").read_bytes())
        evidence = manifest["safe_test_evidence"]
        self.assertEqual(manifest["boundary_ruling"]["version"], BOUNDARY_RULING_VERSION)
        self.assertEqual(manifest["boundary_ruling"]["sha256"], BOUNDARY_RULING_SHA256)
        self.assertEqual(
            hashlib.sha256(manifest["boundary_ruling"]["preimage"].encode()).hexdigest(),
            BOUNDARY_RULING_SHA256,
        )
        self.assertEqual(evidence["test_module_count"], 101)
        self.assertEqual(evidence["included_module_count"], 55)
        self.assertEqual(evidence["included_modules"], list(SAFE_DISCOVERY_INCLUDED_MODULES))
        self.assertEqual(evidence["excluded_module_count"], 46)
        self.assertEqual(evidence["excluded_modules"], list(SAFE_DISCOVERY_EXCLUDED_MODULES))
        self.assertEqual(
            len(set(SAFE_DISCOVERY_INCLUDED_MODULES).union(SAFE_DISCOVERY_EXCLUDED_MODULES)),
            evidence["test_module_count"],
        )
        self.assertEqual(evidence["test_count"], 602)
        self.assertEqual(evidence["failure_count"], 0)
        self.assertEqual(evidence["unexpected_prohibited_open_count"], 0)
        self.assertFalse(evidence["unrestricted_full_discovery_run"])
        self.assertFalse(evidence["validate_scaled_benchmark_run"])
        self.assertEqual(evidence["scaled_manifest"]["sha256"], SCALED_MANIFEST_SHA256)
        self.assertEqual(evidence["scaled_manifest"]["dataset_sha256"], SCALED_DATASET_SHA256)
        self.assertFalse(evidence["scaled_manifest"]["referenced_files_opened"])


if __name__ == "__main__":
    unittest.main()
