from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import summaries.summary_quality_evaluation as quality_evaluation
from summaries.summary_quality_evaluation import (
    EventMapCase,
    GoldClaim,
    GoldEvidence,
    MappedGoldEvent,
    SummaryQualityEvaluationError,
    canonical_json,
    load_event_map,
    load_required_claims,
    load_scorer_inputs,
    load_summary_gold_prefix,
    match_case_events,
    score_summary_quality,
)

from summaries.summary_quality_runtime import (
    BASELINE_VERSION,
    ClaimSnapshot,
    EXPECTED_CASE_IDS,
    ExactEvidence,
    PredictedStatement,
    SummaryQualityRuntimeError,
    TemporalFailure,
    TemporalPrediction,
    VisibleSessionSummary,
    canonical_jsonl,
    load_jsonl_records,
    load_temporal_runtime,
    run_temporal_cases,
    write_runtime_checkpoint,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/summaries/summary_quality_scorer_v2.json"
DATASET_MANIFEST = ROOT / "data/summaries/summary-quality-development-v2/manifest.json"
RUNTIME_CASES = ROOT / "data/summaries/summary-quality-development-v2/runtime/cases.jsonl"
RUNTIME_MODULE = ROOT / "src/summaries/summary_quality_runtime.py"
CHECKPOINT = ROOT / "results/summaries/summary-quality-development-runtime-v2"
SUMMARIES = ROOT / "results/summaries/grounded-summary-development-v1/summaries.jsonl"
SESSIONS = ROOT / "results/summaries/sessionization-development-v1/predictions.jsonl"
CLAIMS = ROOT / "results/phase3/phase4-input-development-gpt41-fallback-v1/claims.jsonl"
DURATIVES = ROOT / "results/summaries/durative-claim-development-v1/claims.jsonl"
GOLD_CASES = ROOT / "data/summaries/summary-quality-development-v2/gold/cases.jsonl"
GOLD_CLAIMS = ROOT / "data/summaries/summary-quality-development-v2/gold/claims.jsonl"
EVENT_MAP = ROOT / "data/summaries/summary-quality-development-v2/gold/event_map.jsonl"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SummaryQualityRuntimeCaseTests(unittest.TestCase):
    def test_frozen_config_and_manifest_bind_runtime_inputs(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        manifest = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(config["scorer_version"], "summary_quality_scorer_v2")
        self.assertEqual(config["baseline_version"], BASELINE_VERSION)
        self.assertEqual(config["case_count"], 10)
        self.assertEqual(config["model_calls"], 0)
        self.assertEqual(config["retries"], 0)
        self.assertEqual(
            sha(CONFIG),
            "28b02f194f2cfafb4c2ec82e876bfae6797a9f588465f7fcc3564da6d58bddf1",
        )
        self.assertEqual(manifest["case_count"], 10)
        self.assertEqual(manifest["user_case_counts"], {"user_001": 5, "user_002": 5})
        self.assertEqual(manifest["runtime"]["cases_sha256"], sha(RUNTIME_CASES))
        self.assertEqual(manifest["runtime"]["module_sha256"], sha(RUNTIME_MODULE))
        self.assertEqual(manifest["runtime"]["scorer_config_sha256"], sha(CONFIG))
        self.assertEqual(manifest["scorer"]["gold_cases"]["sha256"], sha(GOLD_CASES))
        self.assertEqual(manifest["scorer"]["gold_claims"]["sha256"], sha(GOLD_CLAIMS))
        self.assertEqual(manifest["scorer"]["event_map"]["sha256"], sha(EVENT_MAP))
        self.assertTrue(manifest["prior_v1_development_gold_exposure"]["occurred"])
        self.assertFalse(manifest["prior_v1_development_gold_exposure"]["blind_evaluation"])
        self.assertTrue(manifest["scorer"]["checkpoint"]["created_before_gold_copy"])
        self.assertEqual(
            manifest["scorer"]["checkpoint"]["checkpoint_manifest_sha256"],
            sha(CHECKPOINT / "checkpoint_manifest.json"),
        )
        self.assertEqual(sha(GOLD_CASES), "66271b5cd113a126f3ed5c339a599e5ff35d342fa2be233bdeea39310174e8d6")
        self.assertEqual(sha(GOLD_CLAIMS), "88361358e7972753655a38107e3a8fcef2131a3e8b9e98000e771f808ef7001f")
        self.assertEqual(sha(EVENT_MAP), "83393611342afde3bd8606a78f377c2f41969bf2e41c18a4a5493b2029052c17")
        self.assertFalse((ROOT / "configs/summaries/summary_quality_scorer_v1.json").exists())
        self.assertFalse((ROOT / "data/summaries/summary-quality-development-v1").exists())
        self.assertFalse((ROOT / "results/summaries/summary-quality-development-runtime-v1").exists())
        self.assertFalse((ROOT / "results/summaries/summary-quality-development-v1").exists())
        self.assertNotEqual(
            sha(RUNTIME_MODULE),
            "7fc7b600da7bdfcfeca843cab524ce626026a0e51159f48efd9202b1b23861c4",
        )

    def test_prefix_loader_returns_exact_ten_cases_and_users(self) -> None:
        cases = load_temporal_runtime(RUNTIME_CASES)
        self.assertEqual(tuple(item.case_id for item in cases), EXPECTED_CASE_IDS)
        self.assertEqual([item.user_id for item in cases].count("user_001"), 5)
        self.assertEqual([item.user_id for item in cases].count("user_002"), 5)
        self.assertEqual(
            sha(RUNTIME_CASES),
            "3ec24d5abe216b125f088307822bd4e48a3b82f9a21f75cf91e3110640797b33",
        )

    def test_prefix_loader_stops_before_record_eleven(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_bytes(RUNTIME_CASES.read_bytes() + b"this row must not be parsed\n")
            cases = load_temporal_runtime(path)
        self.assertEqual(len(cases), 10)

    def test_prefix_loader_rejects_reorder_test_user_and_missing_row(self) -> None:
        rows = [json.loads(line) for line in RUNTIME_CASES.read_text(encoding="utf-8").splitlines()]
        variants = []
        reordered = deepcopy(rows)
        reordered[0], reordered[1] = reordered[1], reordered[0]
        variants.append(reordered)
        wrong_user = deepcopy(rows)
        wrong_user[0]["user_id"] = "user_003"
        variants.append(wrong_user)
        variants.append(rows[:-1])
        with tempfile.TemporaryDirectory() as directory:
            for index, records in enumerate(variants):
                path = Path(directory) / f"invalid_{index}.jsonl"
                path.write_text(
                    "".join(json.dumps(item, sort_keys=True) + "\n" for item in records),
                    encoding="utf-8",
                )
                with self.subTest(index=index), self.assertRaises(SummaryQualityRuntimeError):
                    load_temporal_runtime(path)

    def test_runtime_module_has_no_scorer_data_import_surface(self) -> None:
        source = RUNTIME_MODULE.read_text(encoding="utf-8").casefold()
        forbidden = tuple(("g" + "old", "or" + "acle", "re" + "view", "sco" + "rer"))
        for token in forbidden:
            self.assertNotIn(token, source)

    def test_runtime_generation_reads_only_frozen_runtime_inputs(self) -> None:
        allowed_reads = {
            path.resolve()
            for path in (RUNTIME_CASES, SUMMARIES, SESSIONS, CLAIMS, DURATIVES)
        }
        opened_reads: set[Path] = set()
        opened_writes: set[Path] = set()
        original_open = Path.open
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "checkpoint"

            def guarded_open(path, *args, **kwargs):
                mode = args[0] if args else kwargs.get("mode", "r")
                resolved = path.resolve()
                if any(flag in mode for flag in "wax+"):
                    if resolved.parent != output.resolve():
                        raise AssertionError(f"runtime wrote outside checkpoint: {resolved}")
                    opened_writes.add(resolved)
                else:
                    if resolved not in allowed_reads:
                        raise AssertionError(f"runtime opened an unapproved input: {resolved}")
                    opened_reads.add(resolved)
                return original_open(path, *args, **kwargs)

            with mock.patch.object(Path, "open", guarded_open):
                cases = load_temporal_runtime(RUNTIME_CASES)
                predictions = run_temporal_cases(
                    cases,
                    load_jsonl_records(SUMMARIES),
                    load_jsonl_records(SESSIONS),
                    load_jsonl_records(CLAIMS),
                    load_jsonl_records(DURATIVES),
                )
                write_runtime_checkpoint(output, predictions, ())

            self.assertEqual(opened_reads, allowed_reads)
            self.assertEqual(
                {path.name for path in opened_writes},
                {"predictions.jsonl", "failures.jsonl", "run.json", "manifest.json"},
            )
            for name in ("predictions.jsonl", "failures.jsonl", "run.json", "manifest.json"):
                self.assertEqual((output / name).read_bytes(), (CHECKPOINT / name).read_bytes())


class SummaryQualityRuntimePredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_temporal_runtime(RUNTIME_CASES)
        cls.summary_rows = load_jsonl_records(SUMMARIES)
        cls.session_rows = load_jsonl_records(SESSIONS)
        cls.claim_rows = load_jsonl_records(CLAIMS)
        cls.durative_rows = load_jsonl_records(DURATIVES)

    def predictions(self):
        return run_temporal_cases(
            self.cases,
            self.summary_rows,
            self.session_rows,
            self.claim_rows,
            self.durative_rows,
        )

    def test_all_cases_receive_instruction_independent_visible_bundle(self) -> None:
        predictions = self.predictions()
        self.assertEqual(len(predictions), 10)
        self.assertEqual(tuple(item.case_id for item in predictions), EXPECTED_CASE_IDS)
        self.assertEqual({len(item.session_summaries) for item in predictions[:5]}, {8})
        self.assertEqual({len(item.session_summaries) for item in predictions[5:]}, {9})
        self.assertTrue(
            all(
                predictions[0].session_summaries == item.session_summaries
                for item in predictions[1:5]
            )
        )
        self.assertTrue(
            all(
                predictions[5].session_summaries == item.session_summaries
                for item in predictions[6:]
            )
        )
        self.assertTrue(all(item.baseline_version == BASELINE_VERSION for item in predictions))
        self.assertTrue(all(item.durative_claim_ids == () for item in predictions))

    def test_statements_are_partitioned_and_have_exact_lineage(self) -> None:
        predictions = self.predictions()
        first_by_user = {item.user_id: item for item in (predictions[0], predictions[5])}
        self.assertEqual(
            sum(len(session.observed_events) for session in first_by_user["user_001"].session_summaries),
            16,
        )
        self.assertEqual(
            sum(len(session.unresolved_questions) for session in first_by_user["user_001"].session_summaries),
            1,
        )
        self.assertEqual(
            sum(len(session.observed_events) for session in first_by_user["user_002"].session_summaries),
            17,
        )
        for prediction in first_by_user.values():
            for session in prediction.session_summaries:
                for statement in (*session.observed_events, *session.unresolved_questions):
                    self.assertEqual(statement.session_id, session.session_id)
                    self.assertEqual(
                        {item.claim_id for item in statement.claims},
                        {item.claim_id for item in statement.evidence},
                    )
                    self.assertEqual(
                        {item.claim_version_id for item in statement.claims},
                        {item.claim_version_id for item in statement.evidence},
                    )
                    self.assertTrue(all(item.user_id == prediction.user_id for item in statement.claims))
                    self.assertTrue(all(item.source_id in session.source_ids for item in statement.evidence))
                    self.assertTrue(all(item.quote for item in statement.evidence))
                    self.assertTrue(all(item.span_id for item in statement.evidence))

    def test_user_and_cutoff_filter_before_bundling(self) -> None:
        summaries = list(self.summary_rows)
        summaries.append(
            {
                "summary_id": "f" * 64,
                "user_id": "user_003",
                "session_definition_id": "not-inspected",
                "statements": "not-inspected",
            }
        )
        sessions = [dict(item) for item in self.session_rows]
        first_session_id = self.summary_rows[0]["session_definition_id"]
        for row in sessions:
            if row["definition_id"] == first_session_id:
                row["transaction_as_of"] = (
                    self.cases[0].as_of + timedelta(seconds=1)
                ).isoformat()
        predictions = run_temporal_cases(
            self.cases,
            summaries,
            sessions,
            self.claim_rows,
            self.durative_rows,
        )
        self.assertEqual(len(predictions[0].session_summaries), 7)
        self.assertEqual(len(predictions[5].session_summaries), 9)

    def test_cross_user_and_missing_provenance_are_rejected(self) -> None:
        cross_user_sessions = [dict(item) for item in self.session_rows]
        target_id = self.summary_rows[0]["session_definition_id"]
        for row in cross_user_sessions:
            if row["definition_id"] == target_id:
                row["user_id"] = "user_002"
        with self.assertRaisesRegex(SummaryQualityRuntimeError, "ownership"):
            run_temporal_cases(
                self.cases,
                self.summary_rows,
                cross_user_sessions,
                self.claim_rows,
                self.durative_rows,
            )
        claims = [deepcopy(item) for item in self.claim_rows]
        target_claim_id = self.summary_rows[0]["statements"][0]["claim_ids"][0]
        for row in claims:
            if row["claim_id"] == target_claim_id:
                row["evidence"] = []
        with self.assertRaisesRegex(SummaryQualityRuntimeError, "evidence"):
            run_temporal_cases(
                self.cases,
                self.summary_rows,
                self.session_rows,
                claims,
                self.durative_rows,
            )

    def test_session_and_statement_order_is_stable(self) -> None:
        forward = self.predictions()
        reverse = run_temporal_cases(
            self.cases,
            tuple(reversed(self.summary_rows)),
            tuple(reversed(self.session_rows)),
            tuple(reversed(self.claim_rows)),
            self.durative_rows,
        )
        self.assertEqual(canonical_jsonl(forward), canonical_jsonl(reverse))
        for prediction in forward:
            order = tuple((item.start_at, item.session_id) for item in prediction.session_summaries)
            self.assertEqual(order, tuple(sorted(order)))

    def test_serialization_is_byte_stable_and_unicode_safe(self) -> None:
        predictions = self.predictions()
        first = canonical_jsonl(predictions)
        second = canonical_jsonl(self.predictions())
        self.assertEqual(first, second)
        self.assertTrue(first.endswith(b"\n"))
        self.assertNotIn(b"NaN", first)
        self.assertEqual(len(first.splitlines()), 10)

    def test_two_clean_checkpoints_are_byte_identical_and_immutable(self) -> None:
        predictions = self.predictions()
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            first_hashes = write_runtime_checkpoint(first, predictions, ())
            second_hashes = write_runtime_checkpoint(second, self.predictions(), ())
            self.assertEqual(first_hashes, second_hashes)
            for name in first_hashes:
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
            with self.assertRaisesRegex(SummaryQualityRuntimeError, "not empty"):
                write_runtime_checkpoint(first, predictions, ())

    def test_checked_checkpoint_is_complete_and_self_verifying(self) -> None:
        expected = {
            "predictions.jsonl": "4bcd6c0c821936465f927e2f3196fcaccf108b997a24c45cc24191602942813a",
            "failures.jsonl": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "run.json": "7c414d556ef4a33e2c106f631bf20037c7e58a4c394fe75a99557f2f61fc0522",
            "manifest.json": "82babccd8d100e14389480f7d34aae5c850f27f88f6f3212ca568575fc060940",
        }
        expected.update(
            {
                "checkpoint_preflight.json": "4a2b8947c10259199ab2ca122f5d86ab8ce225f3819ec4469f98f723f5c23dd7",
                "checkpoint_manifest.json": "6b0a474e42962ac792516c0ed024fa78d9d52842974cf8e8f221f0297e04500e",
            }
        )
        self.assertEqual({item.name for item in CHECKPOINT.iterdir()}, set(expected))
        for name, digest in expected.items():
            self.assertEqual(sha(CHECKPOINT / name), digest)
        manifest = json.loads((CHECKPOINT / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["artifacts"],
            {
                name: expected[name]
                for name in ("predictions.jsonl", "failures.jsonl", "run.json")
            },
        )
        checkpoint_manifest = json.loads(
            (CHECKPOINT / "checkpoint_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            checkpoint_manifest["runtime_module_sha256"],
            "5b8b991d7fe5ea6a723fe7ca18929351ade8559b6e3082aa38c43afc228ea093",
        )
        self.assertEqual(
            checkpoint_manifest["artifacts"],
            {
                name: expected[name]
                for name in (
                    "predictions.jsonl",
                    "failures.jsonl",
                    "run.json",
                    "manifest.json",
                    "checkpoint_preflight.json",
                )
            },
        )
        preflight = json.loads(
            (CHECKPOINT / "checkpoint_preflight.json").read_text(encoding="utf-8")
        )
        absent = set(preflight["absence_attestation"]["absent_before_runtime_generation"])
        self.assertTrue(
            {
                "configs/summaries/summary_quality_scorer_v1.json",
                "configs/summaries/summary_quality_scorer_v2.json",
                "data/summaries/summary-quality-development-v1",
                "data/summaries/summary-quality-development-v2/gold",
                "src/summaries/summary_quality_evaluation.py",
                "tests/integration/test_summary_quality_evaluation.py",
                "tests/unit/test_summary_quality_evaluation.py",
            }
            <= absent
        )
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / "checkpoint"
            invalid.mkdir()
            for item in CHECKPOINT.iterdir():
                (invalid / item.name).write_bytes(item.read_bytes())
            changed = deepcopy(checkpoint_manifest)
            changed["runtime_module_sha256"] = (
                "7fc7b600da7bdfcfeca843cab524ce626026a0e51159f48efd9202b1b23861c4"
            )
            (invalid / "checkpoint_manifest.json").write_text(
                json.dumps(changed, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                SummaryQualityEvaluationError, "manifest hash mismatch"
            ):
                quality_evaluation.load_runtime_checkpoint(invalid)
        run = json.loads((CHECKPOINT / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(run["case_count"], 10)
        self.assertEqual(run["prediction_count"], 10)
        self.assertEqual(run["failure_count"], 0)
        self.assertEqual(run["model_calls"], 0)
        self.assertEqual((CHECKPOINT / "failures.jsonl").read_bytes(), b"")
        prediction_lines = (CHECKPOINT / "predictions.jsonl").read_bytes().splitlines()
        self.assertEqual(len(prediction_lines), 10)
        self.assertEqual(
            tuple(json.loads(line)["case_id"] for line in prediction_lines),
            EXPECTED_CASE_IDS,
        )
        self.assertEqual(
            (CHECKPOINT / "predictions.jsonl").read_bytes(),
            canonical_jsonl(self.predictions()),
        )

    def test_sanitized_failure_contract_remains_available(self) -> None:
        failure = TemporalFailure(
            case_id=EXPECTED_CASE_IDS[0],
            user_id="user_001",
            code="runtime_failure",
            location="case_bundle",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            hashes = write_runtime_checkpoint(root, self.predictions()[1:], (failure,))
            self.assertEqual(
                set(hashes),
                {"predictions.jsonl", "failures.jsonl", "run.json", "manifest.json"},
            )
            for name, expected in hashes.items():
                self.assertEqual(sha(root / name), expected)


class SummaryQualityScorerInputTests(unittest.TestCase):
    def test_strict_scorer_files_have_exact_development_coverage(self) -> None:
        cases = load_summary_gold_prefix(GOLD_CASES)
        claims = load_required_claims(GOLD_CLAIMS)
        mapped = load_event_map(EVENT_MAP)
        self.assertEqual(tuple(item.case_id for item in cases), EXPECTED_CASE_IDS)
        self.assertEqual(len(cases), 10)
        self.assertEqual(len(claims), 24)
        self.assertEqual(len(mapped), 10)
        self.assertEqual(sum(len(item.events) for item in mapped), 22)
        self.assertEqual({item.review_status for item in mapped}, {"implementation_reviewed"})
        quality_evaluation.validate_scorer_inputs(
            *quality_evaluation.load_runtime_checkpoint(CHECKPOINT),
            cases,
            claims,
            mapped,
        )

    def test_summary_prefix_stops_at_ten_and_rejects_test_user_or_order(self) -> None:
        lines = GOLD_CASES.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            poison = root / "poison.jsonl"
            poison.write_bytes(lines + b"this eleventh row must not be parsed\n")
            self.assertEqual(len(load_summary_gold_prefix(poison)), 10)
            rows = [json.loads(line) for line in lines.splitlines()]
            rows[0]["user_id"] = "user_003"
            invalid = root / "invalid.jsonl"
            invalid.write_text(
                "".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8"
            )
            with self.assertRaises(SummaryQualityEvaluationError):
                load_summary_gold_prefix(invalid)
            rows[0]["user_id"] = "user_001"
            rows[0], rows[1] = rows[1], rows[0]
            invalid.write_text(
                "".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8"
            )
            with self.assertRaises(SummaryQualityEvaluationError):
                load_summary_gold_prefix(invalid)

    def test_scorer_files_open_only_after_checkpoint_is_complete(self) -> None:
        ready = False
        original_loader = quality_evaluation.load_runtime_checkpoint
        original_open = Path.open
        protected = {GOLD_CASES.resolve(), GOLD_CLAIMS.resolve(), EVENT_MAP.resolve()}

        def checked_loader(root):
            nonlocal ready
            outputs = original_loader(root)
            self.assertEqual(len(outputs[0]) + len(outputs[1]), 10)
            ready = True
            return outputs

        def guarded_open(path, *args, **kwargs):
            if path.resolve() in protected and not ready:
                raise AssertionError("scorer input opened before checkpoint completion")
            return original_open(path, *args, **kwargs)

        with mock.patch.object(
            quality_evaluation, "load_runtime_checkpoint", side_effect=checked_loader
        ), mock.patch.object(Path, "open", guarded_open):
            loaded = load_scorer_inputs(CHECKPOINT, GOLD_CASES, GOLD_CLAIMS, EVENT_MAP)
        self.assertTrue(ready)
        self.assertEqual([len(value) for value in loaded], [10, 0, 10, 24, 10])

    def test_malformed_event_maps_are_rejected(self) -> None:
        predictions, failures = quality_evaluation.load_runtime_checkpoint(CHECKPOINT)
        cases = load_summary_gold_prefix(GOLD_CASES)
        claims = load_required_claims(GOLD_CLAIMS)
        mapped = list(load_event_map(EVENT_MAP))
        first = mapped[0]
        mapped[0] = replace(first, events=first.events[:-1])
        with self.assertRaisesRegex(SummaryQualityEvaluationError, "event set|Claim union"):
            quality_evaluation.validate_scorer_inputs(
                predictions, failures, cases, claims, mapped
            )
        mapped = list(load_event_map(EVENT_MAP))
        bad_event = replace(
            mapped[0].events[0],
            evidence=(mapped[1].events[0].evidence[0],),
        )
        mapped[0] = replace(mapped[0], events=(bad_event, *mapped[0].events[1:]))
        with self.assertRaisesRegex(SummaryQualityEvaluationError, "evidence"):
            quality_evaluation.validate_scorer_inputs(
                predictions, failures, cases, claims, mapped
            )
        with self.assertRaisesRegex(SummaryQualityEvaluationError, "duplicate"):
            EventMapCase(
                case_id=first.case_id,
                user_id=first.user_id,
                events=(first.events[0], first.events[0]),
                review_status="implementation_reviewed",
            )


class SummaryQualityScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, _, cls.cases, cls.claims, cls.maps = load_scorer_inputs(
            CHECKPOINT, GOLD_CASES, GOLD_CLAIMS, EVENT_MAP
        )
        cls.claim_by_id = {item.claim_id: item for item in cls.claims}

    def _statement(
        self, case_id: str, event: MappedGoldEvent, session_id: str
    ) -> PredictedStatement:
        claims = []
        evidence = []
        covered: set[GoldEvidence] = set()
        versions = {}
        for claim_id in event.required_claim_ids:
            item = self.claim_by_id[claim_id]
            version = f"version:{claim_id}"
            versions[claim_id] = version
            claims.append(
                ClaimSnapshot(
                    claim_id=claim_id,
                    claim_version_id=version,
                    user_id=item.user_id,
                    subject_id=item.subject_id,
                    speaker_id=item.speaker_id,
                    predicate=item.predicate,
                    object=item.object,
                    polarity=item.polarity,
                    epistemic_status=item.epistemic_status,
                    valid_from=item.valid_from,
                    valid_to=item.valid_to,
                    time_precision=item.time_precision,
                    lifecycle_status=item.status,
                )
            )
            for exact in item.evidence:
                if exact not in event.evidence:
                    continue
                covered.add(exact)
                evidence.append(
                    ExactEvidence(
                        claim_id=claim_id,
                        claim_version_id=version,
                        source_id=exact.source_id,
                        span_id=f"span:{claim_id}:{len(evidence)}",
                        message_id=exact.message_id,
                        quote=exact.quote,
                        support_type="supports",
                    )
                )
        first_claim = event.required_claim_ids[0]
        for exact in event.evidence:
            if exact in covered:
                continue
            evidence.append(
                ExactEvidence(
                    claim_id=first_claim,
                    claim_version_id=versions[first_claim],
                    source_id=exact.source_id,
                    span_id=f"span:{first_claim}:{len(evidence)}",
                    message_id=exact.message_id,
                    quote=exact.quote,
                    support_type="supports",
                )
            )
        lifecycle = (
            "disputed"
            if any(item.lifecycle_status == "disputed" for item in claims)
            else "historical"
            if all(item.lifecycle_status in {"historical", "superseded"} for item in claims)
            else "accepted"
        )
        return PredictedStatement(
            statement_id=hashlib.sha256(f"{case_id}:{event.event_id}".encode()).hexdigest(),
            session_id=session_id,
            text=f"Mapped development event {event.event_id}.",
            lifecycle_view=lifecycle,
            claims=tuple(sorted(claims, key=lambda item: item.claim_id)),
            evidence=tuple(
                sorted(
                    evidence,
                    key=lambda item: (
                        item.claim_id,
                        item.claim_version_id,
                        item.source_id,
                        item.span_id,
                    ),
                )
            ),
        )

    def perfect_predictions(self) -> tuple[TemporalPrediction, ...]:
        predictions = []
        for case, mapped in zip(self.cases, self.maps, strict=True):
            session_id = hashlib.sha256(f"session:{case.case_id}".encode()).hexdigest()
            statements = tuple(
                self._statement(case.case_id, event, session_id) for event in mapped.events
            )
            sources = tuple(
                dict.fromkeys(
                    item.source_id
                    for statement in statements
                    for item in statement.evidence
                )
            )
            summary = VisibleSessionSummary(
                summary_id=hashlib.sha256(f"summary:{case.case_id}".encode()).hexdigest(),
                session_id=session_id,
                start_at=case.as_of - timedelta(days=1),
                end_at=case.as_of,
                source_ids=sources,
                summary_text="Deterministic test bundle.",
                observed_events=statements,
                unresolved_questions=(),
            )
            predictions.append(
                TemporalPrediction(
                    case_id=case.case_id,
                    user_id=case.user_id,
                    capability="temporal_reasoning",
                    instruction="Test the frozen scorer contract.",
                    as_of=case.as_of,
                    baseline_version=BASELINE_VERSION,
                    session_summaries=(summary,),
                    durative_claim_ids=(),
                )
            )
        return tuple(predictions)

    def test_maximum_one_to_one_matching_prevents_duplicate_credit(self) -> None:
        prediction = self.perfect_predictions()[0]
        first = self.maps[0].events[0]
        duplicate = replace(first, event_id=first.event_id + "_duplicate")
        mapped = EventMapCase(
            case_id=self.maps[0].case_id,
            user_id=self.maps[0].user_id,
            events=(first, duplicate),
            review_status="implementation_reviewed",
        )
        single_statement = replace(
            prediction.session_summaries[0],
            observed_events=(prediction.session_summaries[0].observed_events[0],),
        )
        prediction = replace(prediction, session_summaries=(single_statement,))
        matches = match_case_events(prediction, mapped, self.claim_by_id)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].event_id, first.event_id)

    def test_perfect_predictions_cover_events_evidence_state_and_structure(self) -> None:
        score = score_summary_quality(
            self.perfect_predictions(), (), self.cases, self.claims, self.maps
        )
        for metric in (
            score.gold_event_micro_precision,
            score.gold_event_micro_recall,
            score.gold_event_micro_f1,
            score.supporting_evidence_micro_precision,
            score.supporting_evidence_micro_recall,
            score.current_historical_accuracy,
            score.correction_preservation,
            score.uncertainty_preservation,
            score.case_accounting,
            score.provenance_coverage,
        ):
            self.assertEqual(metric.value, 1.0)
        self.assertEqual(score.gold_event_micro_recall.denominator, 22)
        self.assertEqual(score.supporting_evidence_micro_recall.denominator, 31)
        self.assertEqual(score.correction_preservation.denominator, 2)
        self.assertEqual(score.uncertainty_preservation.denominator, 9)
        self.assertEqual(score.cross_user_prediction_count, 0)

    def test_false_positive_false_negative_and_failure_stay_in_denominators(self) -> None:
        predictions = list(self.perfect_predictions())
        failed = predictions.pop(0)
        failure = TemporalFailure(
            case_id=failed.case_id,
            user_id=failed.user_id,
            code="runtime_failure",
            location="case_bundle",
        )
        first = predictions[0]
        session = first.session_summaries[0]
        extra = replace(
            session.observed_events[0],
            statement_id=hashlib.sha256(b"extra false positive").hexdigest(),
            claims=tuple(
                replace(item, object={"unmatched": True})
                for item in session.observed_events[0].claims
            ),
        )
        predictions[0] = replace(
            first,
            session_summaries=(
                replace(session, observed_events=(*session.observed_events, extra)),
            ),
        )
        score = score_summary_quality(
            predictions, (failure,), self.cases, self.claims, self.maps
        )
        self.assertLess(score.gold_event_micro_precision.value, 1.0)
        self.assertLess(score.gold_event_micro_recall.value, 1.0)
        self.assertEqual(score.case_accounting.value, 1.0)
        self.assertEqual(score.runtime_failure_count, 1)
        self.assertEqual(score.failure_count, 1)

    def test_dropped_correction_state_and_uncertainty_are_visible(self) -> None:
        predictions = list(self.perfect_predictions())
        correction = predictions[1]
        session = correction.session_summaries[0]
        statement = session.observed_events[0]
        keep = statement.claims[-1].claim_id
        dropped = replace(
            statement,
            claims=tuple(item for item in statement.claims if item.claim_id == keep),
            evidence=tuple(item for item in statement.evidence if item.claim_id == keep),
        )
        predictions[1] = replace(
            correction,
            session_summaries=(replace(session, observed_events=(dropped,)),),
        )
        historical = predictions[0]
        session = historical.session_summaries[0]
        statement = session.observed_events[0]
        changed = replace(
            statement,
            claims=tuple(replace(item, lifecycle_status="current") for item in statement.claims),
        )
        predictions[0] = replace(
            historical,
            session_summaries=(
                replace(
                    session,
                    observed_events=(changed, *session.observed_events[1:]),
                ),
            ),
        )
        maps = list(self.maps)
        direct = replace(maps[0].events[0], uncertainty_expected=True)
        maps[0] = replace(maps[0], events=(direct, *maps[0].events[1:]))
        score = score_summary_quality(
            predictions, (), self.cases, self.claims, maps
        )
        self.assertLess(score.correction_preservation.value, 1.0)
        self.assertLess(score.current_historical_accuracy.value, 1.0)
        self.assertLess(score.uncertainty_preservation.value, 1.0)

    def test_zero_denominators_are_null_with_stable_reasons(self) -> None:
        maps = tuple(
            replace(
                mapped,
                events=tuple(
                    replace(
                        event,
                        expected_state="none",
                        correction_role="none",
                        uncertainty_expected=False,
                    )
                    for event in mapped.events
                ),
            )
            for mapped in self.maps
        )
        score = score_summary_quality(
            self.perfect_predictions(), (), self.cases, self.claims, maps
        )
        self.assertIsNone(score.current_historical_accuracy.value)
        self.assertEqual(
            score.current_historical_accuracy.null_reason,
            "no_matched_reviewed_state_events",
        )
        self.assertIsNone(score.correction_preservation.value)
        self.assertEqual(score.correction_preservation.null_reason, "no_reviewed_correction_events")
        self.assertIsNone(score.uncertainty_preservation.value)
        self.assertEqual(score.uncertainty_preservation.null_reason, "no_reviewed_uncertainty_events")

    def test_scorecard_serialization_is_canonical_and_byte_stable(self) -> None:
        first = score_summary_quality(
            self.perfect_predictions(), (), self.cases, self.claims, self.maps
        )
        second = score_summary_quality(
            self.perfect_predictions(), (), self.cases, self.claims, self.maps
        )
        self.assertEqual(canonical_json(first), canonical_json(second))
        self.assertTrue(canonical_json(first).endswith(b"\n"))


if __name__ == "__main__":
    unittest.main()
