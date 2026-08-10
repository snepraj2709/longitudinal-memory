from __future__ import annotations

from dataclasses import asdict
import unittest

from evaluation.frozen_run_contracts import (
    ExtractionFailure,
    FrozenRunError,
    ProviderRecord,
    canonical_json_bytes,
    failure_from_mapping,
    parse_json_bytes,
    stable_sha256,
)


class FrozenRunContractTests(unittest.TestCase):
    def test_provider_usage_is_strict_and_consistent(self):
        provider = ProviderRecord(
            response_id="resp_1", request_id="req_1",
            requested_model="gpt-4.1-mini-2025-04-14",
            returned_model="gpt-4.1-mini-2025-04-14",
            input_tokens=10, output_tokens=4, total_tokens=14,
            pacing_delay_seconds="0.0", latency_ms=12,
        )
        self.assertEqual(provider.total_tokens, 14)
        changed = asdict(provider)
        changed["total_tokens"] = 15
        with self.assertRaises(FrozenRunError):
            ProviderRecord(**changed)

    def test_failure_identity_recomputes(self):
        payload = {
            "batch_id": "batch_01_extraction_all_sources", "position": 1,
            "source_id": "source_1", "user_id": "user_001",
            "stage": "provider", "code": "provider_request_failed",
            "location": "request",
        }
        failure = ExtractionFailure(failure_id=stable_sha256(payload), **payload)
        self.assertEqual(failure_from_mapping(asdict(failure)), failure)
        changed = asdict(failure)
        changed["code"] = "different"
        with self.assertRaises(FrozenRunError):
            failure_from_mapping(changed)

    def test_unknown_and_nonfinite_json_are_rejected(self):
        payload = {
            "batch_id": "batch_01_extraction_all_sources", "position": 1,
            "source_id": "source_1", "user_id": "user_001",
            "stage": "provider", "code": "provider_request_failed",
            "location": "request",
        }
        payload["failure_id"] = stable_sha256(payload)
        payload["unknown"] = True
        with self.assertRaises(FrozenRunError):
            failure_from_mapping(payload)
        with self.assertRaises(FrozenRunError):
            parse_json_bytes(b'{"x":NaN}', location="unit")
        with self.assertRaises(FrozenRunError):
            canonical_json_bytes({"x": float("inf")})


if __name__ == "__main__":
    unittest.main()
