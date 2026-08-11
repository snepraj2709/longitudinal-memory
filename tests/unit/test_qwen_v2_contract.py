from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from evaluation.qwen_v2_contract import (
    EXPECTED_LOGICAL_COUNTS,
    EXPECTED_PROVIDER_COUNTS,
    QwenV2ContractError,
    build_series_manifest,
    load_qwen_v2_config,
    verify_series_manifest,
)


ROOT = Path(__file__).resolve().parents[2]


class QwenV2ContractTests(unittest.TestCase):
    def test_frozen_contract_has_exact_model_baselines_and_counts(self) -> None:
        config = load_qwen_v2_config(ROOT)
        self.assertEqual(config["series_id"], "qwen35-27b-fp8-v2")
        self.assertEqual(config["model"]["revision"], "97f5941bf617e31c5e237364a8602ce3f03a551a")
        self.assertEqual(config["runtime"]["client_concurrency"], 8)
        self.assertEqual(config["workload"]["provider_requests"], EXPECTED_PROVIDER_COUNTS)
        self.assertEqual(config["workload"]["logical_predictions"], EXPECTED_LOGICAL_COUNTS)
        self.assertFalse(config["baselines"][-1]["provider_call"])

    def test_series_manifest_is_zero_call_and_verifiable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "series.json"
            build_series_manifest(ROOT, path)
            manifest = verify_series_manifest(ROOT, path)
            self.assertEqual(manifest["status"], "frozen_not_run")
            self.assertEqual(manifest["provider_request_count"], 0)
            self.assertEqual(manifest["predecessor_status"], "superseded_not_run")

    def test_manifest_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "series.json"
            build_series_manifest(ROOT, path)
            value = json.loads(path.read_text(encoding="utf-8"))
            value["provider_request_count"] = 1
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(QwenV2ContractError, "manifest changed"):
                verify_series_manifest(ROOT, path)


if __name__ == "__main__":
    unittest.main()
