from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import unittest

from evaluation.benchmark_release import (
    BENCHMARK_VERSION,
    CAPABILITIES,
    GOLD_PATHS,
    MANIFEST_PATH,
    ORACLE_PATHS,
    RELEASE_DATA_PATHS,
    RUNTIME_CASE_PATHS,
    BenchmarkReleaseError,
    _validate_runtime_records,
    load_benchmark_v1_runtime,
    validate_benchmark_v1,
)
from evaluation.run_config import dataset_sha256


PILOT_V0_PROTECTED_SHA256 = {
    "data/pilot/user.jsonl": "ec951239b54197c76c3fd048564b18494c2d97a3d632a63b985e3c2c9c247140",
    "data/pilot/sources/emails.jsonl": "ba94d6c80b8d307862e2692ad5ee37db0114599a3190811145f647976e816bed",
    "data/pilot/sources/conversations.jsonl": "0f4900d8b13ee55040bcfde4d1b7c6943786b11d1723c6c7aaf2ce0eb6480c9e",
    "data/pilot/sources/calendar.jsonl": "8f5d3dfef2ab3115b2099e79b4151415925649b892c9c3443cfe89eee89752b9",
    "data/pilot/evaluation/eval_answer.jsonl": "2cf6ec828aecf832aaa497af4d80a194ad45f713af14cb104f23ddb4a57d67be",
    "data/pilot/evaluation/eval_questions.jsonl": "bed228c6ada33bf515bd0eff9cdb35942c4c72aad71960e5dc0acdd9378f2789",
    "data/pilot/oracle-event.jsonl": "921431679979037657e8737c412e8b633a772e0fb1d7a5cda5a5f64425bc1136",
    "results/pilot/b1-full-history/failures.jsonl": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "results/pilot/b1-full-history/scores.json": "3187cedbabb60fbec2d6aca1dfd517d09cd37a404f7cd00e04d9065155b91f3d",
    "results/pilot/b1-full-history/predictions.jsonl": "63e5b1061c98edbec8a2c77c2260bd364bb3362d876fe22a02e3a10a2f5187b0",
    "results/pilot/b1-full-history/failure_analysis.jsonl": "8332084a3316e326074c767d45bf7fd5824a45f62e0bcae9e3f165b9cf053bf1",
    "results/pilot/b1-full-history/manual_review.jsonl": "9e9a74331a9af4f608b780a23fce5e936e0964f06edac874424d127188fbe2c0",
    "results/pilot/b1-full-history/case_scores.jsonl": "07a0db890fef5b1fa203edcf642b1cbaeef8602d147aac22c6c660f377da4ab8",
    "results/pilot/b1-full-history/baseline_findings.md": "62c8f0d1d64972b2fe031dd5aadd7efa7d1cb9f0d17ec5de9019879dcc2151f8",
    "results/pilot/b1-full-history/run.json": "b8f7b38fac1d28b70de219a96a567e98052a86764513da01d692019b87926507",
    "results/pilot/b1-full-history/failure_summary.json": "3c309527c10c9fc4344b7acb5994795e20e1f174557087e769d76a6b2b96bd0f",
    "results/pilot/full_history/20260807T083918Z-gpt-4.1-2025-04-14/predictions.jsonl": "d18baeaaef716258ba1575109b61c1418aec57040d449c959f4a7b909b933aea",
    "results/pilot/full_history/20260807T083918Z-gpt-4.1-2025-04-14/api_metadata.jsonl": "ae4662714f368659be9b68da1d565ad61ef7125de9220e30da277fc82de951d3",
    "results/pilot/full_history/20260807T083918Z-gpt-4.1-2025-04-14/report.md": "1ff98e8c7bf09cf589768bc4f6827fed199533d445e48af57762dcb618e7e264",
    "results/pilot/full_history/20260807T083918Z-gpt-4.1-2025-04-14/manifest.json": "1253d9a9c00618979205f9fcdaa4f68301e3082b8c5803ab9b3d450160100f84",
    "results/pilot/full_history/20260807T083918Z-gpt-4.1-2025-04-14/diagnostics.jsonl": "0f29c5cb6e4d99332b2c2209cb0a39cc599efd338bce1e9b9b3156bc550531db",
    "results/pilot/full_history/20260807T082949Z-gpt-4.1-2025-04-14/predictions.jsonl": "5bc1f506879ad4e994d5d2cb92fa43256847d5bc47fb190bec2c77a5c5ffc85b",
    "results/pilot/full_history/20260807T082949Z-gpt-4.1-2025-04-14/api_metadata.jsonl": "d9e089af5072c8173425ec07c785fab5ab72d4f95146ebae171c2845cc297b81",
    "results/pilot/full_history/20260807T082949Z-gpt-4.1-2025-04-14/report.md": "50b6288c57ff60d86bae076151177e2f8e8e8d24a92adb3b1645b26d0cb4712c",
    "results/pilot/full_history/20260807T082949Z-gpt-4.1-2025-04-14/diagnostics.jsonl": "d2ffe2f39934d445b2614a873fd19840a9f78eb53c2088880f71003e69e95407",
}


class BenchmarkReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[2]

    def test_release_validates_with_required_counts_and_balance(self) -> None:
        report = validate_benchmark_v1(self.repo_root)

        self.assertEqual(report.benchmark_version, BENCHMARK_VERSION)
        self.assertEqual(report.qa_count, 50)
        self.assertEqual(report.summary_count, 5)
        self.assertEqual(report.interactive_count, 2)
        self.assertEqual(report.claim_count, 30)
        self.assertEqual(
            dict(report.capability_counts),
            {capability: 10 for capability in CAPABILITIES},
        )
        self.assertEqual(report.human_review_status, "approved")

    def test_runtime_loader_exposes_only_runtime_records(self) -> None:
        runtime = load_benchmark_v1_runtime(self.repo_root)

        self.assertEqual(len(runtime.users), 1)
        self.assertEqual(len(runtime.observations), 72)
        self.assertEqual(len(runtime.qa), 50)
        self.assertEqual(len(runtime.summaries), 5)
        self.assertEqual(len(runtime.interactive), 2)
        for case in runtime.qa + runtime.summaries + runtime.interactive:
            self.assertFalse(hasattr(case, "reference_answer"))
            self.assertFalse(hasattr(case, "evidence"))
            self.assertFalse(hasattr(case, "should_abstain"))

    def test_runtime_loader_never_opens_manifest_or_scorer_files(self) -> None:
        opened: list[Path] = []
        original_open = Path.open

        def tracked_open(path: Path, *args: object, **kwargs: object):
            opened.append(path.resolve())
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", new=tracked_open):
            load_benchmark_v1_runtime(self.repo_root)

        forbidden = [(self.repo_root / MANIFEST_PATH).resolve()]
        forbidden.extend((self.repo_root / path).resolve() for path in ORACLE_PATHS)
        forbidden.extend((self.repo_root / path).resolve() for path in GOLD_PATHS)
        self.assertTrue(opened)
        self.assertFalse(any(path in forbidden for path in opened))

    def test_manifest_hashes_every_release_file(self) -> None:
        manifest = json.loads(
            (self.repo_root / MANIFEST_PATH).read_text(encoding="utf-8")
        )
        declared = {item["path"]: item for item in manifest["files"]}

        self.assertEqual(
            set(declared),
            {path.as_posix() for path in RELEASE_DATA_PATHS},
        )
        for relative in RELEASE_DATA_PATHS:
            path = self.repo_root / relative
            self.assertEqual(
                declared[relative.as_posix()]["sha256"],
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        self.assertEqual(
            manifest["dataset_sha256"],
            dataset_sha256(
                self.repo_root,
                tuple(path.as_posix() for path in RELEASE_DATA_PATHS),
            ),
        )

    def test_pilot_v0_data_and_results_keep_their_frozen_hashes(self) -> None:
        for relative, expected in PILOT_V0_PROTECTED_SHA256.items():
            with self.subTest(path=relative):
                path = self.repo_root / relative
                self.assertTrue(path.is_file())
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)

    def test_runtime_loader_rejects_a_scorer_field(self) -> None:
        records = {
            path.stem: [
                json.loads(line)
                for line in (self.repo_root / path)
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            for path in RUNTIME_CASE_PATHS
        }
        records["qa"][0]["reference_answer"] = "leaked"
        errors: list[str] = []

        _validate_runtime_records(
            {
                "qa": records["qa"],
                "summarization": records["summaries"],
                "interactive": records["interactive"],
            },
            errors,
        )

        self.assertTrue(any("reference_answer" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
