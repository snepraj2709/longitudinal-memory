from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest

from extraction.contracts import ALLOWED_PREDICATES, AtomicClaimV1, EvidenceSpanV1
from extraction.failure_analysis import (
    ANALYSIS_VERSION,
    FROZEN_INPUT_SHA256,
    PREDICATE_FAMILIES,
    PROTECTED_B1_SHA256,
    STABLE_CATEGORIES,
    AtomicFailureAnalysisError,
    classify_valid_time_mismatch,
    run_failure_analysis,
)


class AtomicExtractionFailureAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[2]

    def claim(
        self,
        *,
        valid_from: str | None,
        valid_to: str | None,
    ) -> AtomicClaimV1:
        return AtomicClaimV1(
            claim_id="claim",
            subject_id="i_am_maya",
            speaker_id="i_am_maya",
            predicate="job_start_date",
            object="2026-05-04",
            polarity="positive",
            epistemic_status="asserted",
            valid_from=valid_from,
            valid_to=valid_to,
            confidence=1.0,
            evidence=(EvidenceSpanV1("conv_002", "msg_conv_002_002", "on May 4"),),
        )

    def test_predicate_families_and_categories_are_complete_and_stable(self) -> None:
        self.assertEqual(set(PREDICATE_FAMILIES), set(ALLOWED_PREDICATES))
        self.assertEqual(len(STABLE_CATEGORIES), len(set(STABLE_CATEGORIES)))
        self.assertEqual(
            set(STABLE_CATEGORIES),
            {
                "missed_gold_claim",
                "extra_source_backed_claim",
                "unsupported_claim",
                "subject_mismatch",
                "speaker_mismatch",
                "predicate_mismatch",
                "object_mismatch",
                "polarity_mismatch",
                "epistemic_status_mismatch",
                "missing_valid_time",
                "wrong_valid_time",
                "over_broad_valid_time",
                "evidence_span_mismatch",
            },
        )

    def test_valid_time_categories_cover_missing_wrong_and_over_broad(self) -> None:
        gold = self.claim(valid_from="2026-05-04", valid_to="2026-05-04")
        self.assertEqual(
            classify_valid_time_mismatch(
                gold, replace(gold, valid_from=None, valid_to=None)
            ),
            "missing_valid_time",
        )
        self.assertEqual(
            classify_valid_time_mismatch(
                gold,
                replace(gold, valid_from="2026-05-05", valid_to="2026-05-05"),
            ),
            "wrong_valid_time",
        )
        self.assertEqual(
            classify_valid_time_mismatch(
                gold,
                replace(gold, valid_from="2026-05-03", valid_to="2026-05-04"),
            ),
            "over_broad_valid_time",
        )

    def test_frozen_analysis_reconciles_and_every_issue_has_source_support(self) -> None:
        before = self._hashes({**FROZEN_INPUT_SHA256, **PROTECTED_B1_SHA256})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "analysis"
            manifest = run_failure_analysis(
                repo_root=self.repo_root,
                output_dir=output,
                now=lambda: datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc),
            )
            records = self._jsonl(output / "case_failures.jsonl")
            summary = json.loads((output / "summary.json").read_text())

            self.assertEqual(manifest["analysis_version"], ANALYSIS_VERSION)
            self.assertEqual(manifest["release_status"], "approved")
            self.assertEqual(manifest["human_review"]["status"], "approved")
            self.assertEqual(
                manifest["human_review"]["reviewed_manifest_sha256"],
                "d44a71d6da47d4fc970473db97c10432edf68a8282df9ecac7f4d4c40ef835a5",
            )
            self.assertEqual(len(records), 10)
            self.assertEqual(summary["counts"]["gold_claims"], 48)
            self.assertEqual(summary["counts"]["predicted_claims"], 36)
            self.assertEqual(summary["counts"]["exact_matches"], 27)
            self.assertEqual(summary["counts"]["false_positives"], 9)
            self.assertEqual(summary["counts"]["false_negatives"], 21)
            self.assertEqual(summary["counts"]["unsupported_claims"], 0)
            self.assertEqual(summary["counts"]["execution_failures"], 0)
            self.assertEqual(summary["counts"]["evidence_exact_matches"], 11)
            self.assertEqual(summary["counts"]["evidence_gold_spans"], 50)
            self.assertEqual(summary["counts"]["evidence_predicted_spans"], 36)
            self.assertEqual(summary["category_counts"]["missed_gold_claim"], 12)
            self.assertEqual(
                summary["category_counts"]["extra_source_backed_claim"], 9
            )
            self.assertEqual(summary["category_counts"]["object_mismatch"], 9)
            self.assertEqual(summary["category_counts"]["missing_valid_time"], 12)
            self.assertEqual(
                summary["category_counts"]["epistemic_status_mismatch"], 3
            )
            self.assertEqual(
                summary["category_counts"]["evidence_span_mismatch"], 25
            )
            self.assertEqual(set(summary["by_case"]), {r["case_id"] for r in records})
            self.assertEqual(
                set(summary["by_source_type"]), {"calendar", "conversation", "email"}
            )
            for record in records:
                self.assertIsNone(record["execution_failure"])
                for issue in record["issues"]:
                    self.assertRegex(issue["issue_id"], r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
                    self.assertIn(issue["category"], STABLE_CATEGORIES)
                    self.assertTrue(issue["source_references"])
                    self.assertTrue(issue["explanation"])
                    for reference in issue["source_references"]:
                        self.assertIn(reference["role"], {"gold", "prediction"})
                        self.assertTrue(reference["source_id"])
                        self.assertTrue(reference["quote"])

        self.assertEqual(before, self._hashes(before))

    def test_repeat_runs_only_change_manifest_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            run_failure_analysis(
                repo_root=self.repo_root,
                output_dir=first,
                now=lambda: datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc),
            )
            run_failure_analysis(
                repo_root=self.repo_root,
                output_dir=second,
                now=lambda: datetime(2026, 8, 8, 12, 1, tzinfo=timezone.utc),
            )
            for filename in ("case_failures.jsonl", "summary.json", "findings.md"):
                self.assertEqual((first / filename).read_bytes(), (second / filename).read_bytes())
            first_manifest = json.loads((first / "manifest.json").read_text())
            second_manifest = json.loads((second / "manifest.json").read_text())
            self.assertNotEqual(
                first_manifest.pop("generated_at_utc"),
                second_manifest.pop("generated_at_utc"),
            )
            self.assertEqual(first_manifest, second_manifest)

    def test_refuses_to_overwrite_an_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "analysis"
            output.mkdir()
            (output / "keep.txt").write_text("keep\n")
            with self.assertRaisesRegex(
                AtomicFailureAnalysisError, "refusing to overwrite"
            ):
                run_failure_analysis(repo_root=self.repo_root, output_dir=output)

    def test_analysis_module_has_no_model_or_api_client_dependency(self) -> None:
        source = (
            self.repo_root / "src/extraction/failure_analysis.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("openai_client", source)
        self.assertNotIn("OpenAI", source)
        self.assertNotIn("complete_with_metadata", source)

    def _hashes(self, paths: dict[str, str]) -> dict[str, str]:
        return {
            relative_path: hashlib.sha256(
                (self.repo_root / relative_path).read_bytes()
            ).hexdigest()
            for relative_path in paths
        }

    @staticmethod
    def _jsonl(path: Path) -> list[dict[str, object]]:
        return [json.loads(line) for line in path.read_text().splitlines()]


if __name__ == "__main__":
    unittest.main()
