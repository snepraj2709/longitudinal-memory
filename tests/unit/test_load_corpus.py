from __future__ import annotations

from itertools import islice
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from evaluation.benchmark_release import load_benchmark_v1_runtime
from evaluation.scaled_release import load_scaled_runtime
from load_testing.corpus import (
    DATA_ROOT,
    EVENT_COUNT,
    QUALITY_RELEASE_HASHES,
    USER_COUNT,
    LoadCorpusError,
    iter_load_events,
    reject_quality_use,
    validate_load_corpus,
)
from scripts.generate_load_corpus import generate


EXPECTED_CORPUS_SHA256 = "c271b987c53262176f9a1692140ba5caa039cc2533404caf692bd7199b8b957b"


class LoadCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[2]

    def test_full_corpus_has_exact_counts_balance_hash_and_review_state(self) -> None:
        report = validate_load_corpus(self.repo_root)

        self.assertEqual(report.corpus_version, "load_test_v1")
        self.assertEqual(report.corpus_sha256, EXPECTED_CORPUS_SHA256)
        self.assertEqual(report.users, USER_COUNT)
        self.assertEqual(report.source_events, EVENT_COUNT)
        self.assertEqual(report.sessions, 100_000)
        self.assertEqual(report.event_chunks, 100)
        self.assertEqual(
            dict(report.source_type_counts),
            {"conversation": 125_000, "email": 125_000, "calendar": 125_000, "chat": 125_000},
        )
        self.assertEqual(report.human_review_status, "approved")

    def test_streaming_loader_yields_only_load_events(self) -> None:
        records = list(islice(iter_load_events(self.repo_root), 12))

        self.assertEqual(len(records), 12)
        self.assertEqual(
            [record["source_id"] for record in records],
            [f"load_v1_source_{index:09d}" for index in range(12)],
        )
        self.assertTrue(all(record["dataset_role"] == "load_test" for record in records))
        self.assertTrue(all("question" not in record and "reference_answer" not in record for record in records))

    def test_manifest_excludes_quality_denominators_and_protects_release_hashes(self) -> None:
        manifest = json.loads((self.repo_root / DATA_ROOT / "manifest.json").read_text(encoding="utf-8"))

        self.assertIs(manifest["quality_denominator_eligible"], False)
        self.assertIs(manifest["contains_evaluation_records"], False)
        self.assertEqual(manifest["counts"]["quality_cases"], 0)
        self.assertEqual(manifest["counts"]["quality_gold_records"], 0)
        self.assertEqual(manifest["protected_quality_releases"], QUALITY_RELEASE_HASHES)
        self.assertEqual(manifest["generator"]["llm_calls"], 0)
        self.assertEqual(manifest["release_status"], "frozen")
        self.assertEqual(manifest["review"]["human_review_status"], "approved")
        self.assertEqual(manifest["review"]["reviewed_by"], "Sneha")
        self.assertEqual(manifest["review"]["reviewed_corpus_sha256"], EXPECTED_CORPUS_SHA256)
        with self.assertRaises(LoadCorpusError):
            reject_quality_use(manifest)

    def test_quality_runtime_loaders_never_open_load_corpus_files(self) -> None:
        opened: list[Path] = []
        original_open = Path.open

        def tracked_open(path: Path, *args: object, **kwargs: object):
            opened.append(path.resolve())
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", new=tracked_open):
            load_benchmark_v1_runtime(self.repo_root)
            load_scaled_runtime(self.repo_root, "user_001")

        load_root = (self.repo_root / DATA_ROOT).resolve()
        self.assertTrue(opened)
        self.assertTrue(all(load_root not in path.parents and path != load_root for path in opened))

    def test_generator_is_byte_deterministic_for_same_configuration(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            reports = []
            manifests = []
            trees = []
            for run_number in range(2):
                run_root = root / f"run_{run_number}"
                data_root = run_root / "data" / "load-test-v1"
                schema_root = run_root / "schemas" / "load-test-v1"
                reports.append(generate(data_root, schema_root, 17, 200, 20, 50, "2026-08-08"))
                manifests.append((data_root / "manifest.json").read_bytes())
                trees.append({
                    path.relative_to(run_root).as_posix(): path.read_bytes()
                    for path in sorted(run_root.rglob("*"))
                    if path.is_file()
                })

        self.assertEqual(manifests[0], manifests[1])
        self.assertEqual(trees[0], trees[1])
        self.assertEqual(reports[0]["corpus_sha256"], reports[1]["corpus_sha256"])


if __name__ == "__main__":
    unittest.main()
