from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import tempfile
import unittest

from evaluation.run_config import load_frozen_config, verify_frozen_content
from extraction.gold import AtomicGoldDataError, load_atomic_gold
from extraction.source import load_pilot_sources


class AtomicExtractionGoldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.gold_path = self.repo_root / "data/phase3/atomic_extraction_gold.jsonl"
        self.sources = load_pilot_sources()
        self.records = [
            json.loads(line)
            for line in self.gold_path.read_text(encoding="utf-8").splitlines()
        ]

    def _copy_record(self, index: int = 0) -> dict[str, object]:
        return json.loads(json.dumps(self.records[index]))

    def _write_records(self, records: list[object]) -> Path:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        path = Path(temporary_directory.name) / "atomic_gold.jsonl"
        path.write_text(
            "\n".join(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                for record in records
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def _load_records(self, records: list[object]):
        return load_atomic_gold(self._write_records(records), self.sources)

    def test_loads_ten_frozen_cases_in_stable_file_order(self) -> None:
        cases = load_atomic_gold(self.gold_path, self.sources)

        self.assertEqual(
            [case.case_id for case in cases],
            [
                "atomic_cal_001",
                "atomic_conv_002",
                "atomic_conv_003",
                "atomic_cal_002",
                "atomic_conv_004",
                "atomic_cal_003",
                "atomic_email_003",
                "atomic_email_005",
                "atomic_conv_010",
                "atomic_cal_006",
            ],
        )
        self.assertEqual(
            [len(case.expected_claims) for case in cases],
            [1, 10, 10, 1, 9, 1, 0, 8, 4, 1],
        )
        with self.assertRaises(FrozenInstanceError):
            cases[0].case_id = "changed"

    def test_gold_contains_required_claim_and_time_coverage(self) -> None:
        cases = load_atomic_gold(self.gold_path, self.sources)
        claims = {
            claim.claim_id: claim
            for case in cases
            for claim in case.expected_claims
        }

        self.assertTrue(any(claim.epistemic_status == "asserted" for claim in claims.values()))
        self.assertTrue(
            any(claim.epistemic_status == "reported_by_other" for claim in claims.values())
        )
        self.assertTrue(any(claim.epistemic_status == "uncertain" for claim in claims.values()))
        self.assertTrue(any(claim.epistemic_status == "hypothetical" for claim in claims.values()))
        self.assertTrue(any(claim.epistemic_status == "corrected" for claim in claims.values()))
        self.assertTrue(
            any(
                claim.epistemic_status == "denied" and claim.polarity == "negative"
                for claim in claims.values()
            )
        )
        self.assertEqual(
            claims["email_005_campaign_launch_date"].valid_from,
            "2026-08-19",
        )
        self.assertEqual(
            claims["conv_003_aryan_job_start_report"].valid_from,
            "2026-05-11",
        )
        no_claim_case = next(case for case in cases if case.source_id == "email_003")
        self.assertEqual(no_claim_case.expected_claims, ())

    def test_rejects_duplicate_case_and_source_ids(self) -> None:
        duplicate_case = self._copy_record()
        with self.assertRaisesRegex(AtomicGoldDataError, "duplicate case_id"):
            self._load_records([self.records[0], duplicate_case])

        duplicate_source = self._copy_record()
        duplicate_source["case_id"] = "another_case"
        with self.assertRaisesRegex(AtomicGoldDataError, "duplicate source_id"):
            self._load_records([self.records[0], duplicate_source])

    def test_rejects_invalid_json_with_line_number(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        path = Path(temporary_directory.name) / "invalid.jsonl"
        path.write_text(
            json.dumps(self.records[0], separators=(",", ":")) + "\n{\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(AtomicGoldDataError, r"invalid\.jsonl:2: invalid JSON"):
            load_atomic_gold(path, self.sources)

    def test_rejects_missing_and_unknown_case_fields_together(self) -> None:
        record = self._copy_record()
        del record["case_id"]
        record["notes"] = "not part of the contract"

        with self.assertRaises(AtomicGoldDataError) as raised:
            self._load_records([record])

        self.assertIn("missing required fields: case_id", str(raised.exception))
        self.assertIn("unknown fields: notes", str(raised.exception))

    def test_rejects_unknown_source(self) -> None:
        record = self._copy_record(6)
        record["source_id"] = "unknown_source"

        with self.assertRaisesRegex(AtomicGoldDataError, "unknown source_id"):
            self._load_records([record])

    def test_rejects_invalid_step_1_claim(self) -> None:
        record = self._copy_record(1)
        record["expected_claims"][0]["polarity"] = "neutral"

        with self.assertRaisesRegex(
            AtomicGoldDataError,
            r"claims\[0\]: polarity must be one of: negative, positive",
        ):
            self._load_records([record])

    def test_rejects_unknown_message_and_inexact_quote(self) -> None:
        unknown_message = self._copy_record(1)
        unknown_message["expected_claims"][0]["evidence"][0]["message_id"] = "missing"
        with self.assertRaisesRegex(AtomicGoldDataError, "message_id does not exist"):
            self._load_records([unknown_message])

        inexact_quote = self._copy_record(1)
        inexact_quote["expected_claims"][0]["evidence"][0]["quote"] = "not in the source"
        with self.assertRaisesRegex(AtomicGoldDataError, "quote is not an exact substring"):
            self._load_records([inexact_quote])

    def test_calendar_evidence_requires_null_message_id(self) -> None:
        cases = load_atomic_gold(self.gold_path, self.sources)
        calendar_cases = [case for case in cases if case.source_id.startswith("cal_")]
        self.assertTrue(calendar_cases)
        self.assertTrue(
            all(
                evidence.message_id is None
                for case in calendar_cases
                for claim in case.expected_claims
                for evidence in claim.evidence
            )
        )

        invalid = self._copy_record()
        invalid["expected_claims"][0]["evidence"][0]["message_id"] = "calendar_message"
        with self.assertRaisesRegex(
            AtomicGoldDataError, "calendar evidence must use message_id: null"
        ):
            self._load_records([invalid])

    def test_b1_frozen_content_remains_valid(self) -> None:
        config = load_frozen_config(
            self.repo_root / "configs/full_history_baseline_v1.json"
        )

        verify_frozen_content(config, self.repo_root)


if __name__ == "__main__":
    unittest.main()
