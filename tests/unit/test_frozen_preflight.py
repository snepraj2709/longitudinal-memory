from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from evaluation.frozen_preflight import (
    CONFIG_PATH,
    build_frozen_preflight,
)
from evaluation.frozen_preflight_contracts import (
    FrozenPreflightError,
    batch_from_mapping,
    canonical_json_bytes,
    parse_json_bytes,
    stable_sha256,
    transmission_from_mapping,
)


ROOT = Path(__file__).resolve().parents[2]


class FrozenPreflightUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temporary.name)
        build_frozen_preflight(
            repo_root=ROOT, data_root=cls.base / "data", result_root=cls.base / "result",
        )
        cls.preflight = json.loads((cls.base / "result/preflight.json").read_text())
        cls.batches = [json.loads(line) for line in (cls.base / "result/batches.jsonl").read_text().splitlines()]
        cls.transmissions = [json.loads(line) for line in (cls.base / "data/runtime/transmission-plan.jsonl").read_text().splitlines()]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_exact_batch_request_and_cost_totals(self):
        self.assertEqual(len(self.batches), 25)
        self.assertEqual(sum(item["request_count"] for item in self.batches), 4660)
        self.assertEqual(self.batches[0]["request_count"], 100)
        self.assertEqual(self.batches[-1]["batch_id"], "batch_25_B7_interactive")
        self.assertEqual(self.preflight["expected_incremental_cost_usd"], "32.8869328")
        self.assertEqual(self.preflight["maximum_incremental_cost_usd"], "91.2131728")
        self.assertEqual(self.preflight["projected_maximum_cumulative_spend_usd"], "91.4446132")

    def test_models_prices_and_zero_retry_are_frozen(self):
        config = json.loads((ROOT / CONFIG_PATH).read_text())
        self.assertEqual(config["answer_model"]["requested_model"], "gpt-4.1-2025-04-14")
        self.assertEqual(config["extraction_model"]["requested_model"], "gpt-4.1-mini-2025-04-14")
        self.assertEqual((config["answer_model"]["input_usd_per_million_tokens"], config["answer_model"]["output_usd_per_million_tokens"]), ("2.00", "8.00"))
        self.assertEqual((config["extraction_model"]["input_usd_per_million_tokens"], config["extraction_model"]["output_usd_per_million_tokens"]), ("0.40", "1.60"))
        self.assertEqual(config["maximum_retry_requests"], 0)
        self.assertEqual(config["pricing"]["checked_at"], "2026-08-10")

    def test_three_approval_facts_remain_separate(self):
        self.assertTrue(self.preflight["credential_reuse_approved"])
        self.assertFalse(self.preflight["data_transmission_approved"])
        self.assertFalse(self.preflight["paid_execution_approved"])
        self.assertFalse(self.preflight["provider_execution_authorized"])
        self.assertEqual(self.preflight["provider_request_count"], 0)

    def test_transmission_plan_is_exact_and_contains_no_content(self):
        self.assertEqual(len(self.transmissions), 4660)
        self.assertEqual(sum(item["kind"] == "extraction_source" for item in self.transmissions), 100)
        self.assertEqual(sum(item["kind"] == "answer_case" for item in self.transmissions), 4560)
        self.assertTrue(all(item["synthetic_benchmark"] and not item["restricted_content"] for item in self.transmissions))
        serialized = canonical_json_bytes(self.transmissions)
        for prohibited in (b'"content":', b'"question":', b'"initial_user_message":', b'"profile_note":'):
            self.assertNotIn(prohibited, serialized)

    def test_transmission_id_is_derived_from_public_metadata(self):
        record = transmission_from_mapping(self.transmissions[0])
        payload = asdict(record)
        identity = payload.pop("transmission_id")
        self.assertEqual(identity, stable_sha256(payload))

    def test_strict_unknown_and_unsafe_json_rejection(self):
        changed = dict(self.batches[0])
        changed["unknown"] = True
        with self.assertRaises(FrozenPreflightError):
            batch_from_mapping(changed)
        with self.assertRaises(FrozenPreflightError):
            parse_json_bytes(b'{"x":NaN}', location="unit")
        with self.assertRaises(FrozenPreflightError):
            canonical_json_bytes({"x": float("inf")})

    def test_batch_contract_rejects_self_approval_and_retries(self):
        changed = dict(self.batches[0])
        changed["provider_execution_authorized"] = True
        with self.assertRaises(FrozenPreflightError):
            batch_from_mapping(changed)
        changed = dict(self.batches[0])
        changed["maximum_retry_requests"] = 1
        with self.assertRaises(FrozenPreflightError):
            batch_from_mapping(changed)


if __name__ == "__main__":
    unittest.main()
