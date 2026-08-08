from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from extraction.contracts import (
    AtomicClaimValidationError,
    validate_atomic_claim,
)
from extraction.predicate_registry import (
    CONFLICT_COMPATIBILITIES,
    OBJECT_SHAPES,
    PREDICATE_FAMILIES,
    SUBJECT_SCOPES,
    TEMPORAL_BEHAVIORS,
    PredicateRegistryError,
    load_default_predicate_registry,
    load_predicate_registry,
    render_registry_for_prompt,
    validate_object_shape,
)
from extraction.run_atomic import AtomicPipelineError, dry_run_atomic


REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "configs/extraction/predicate_registry_v1.json"
REGISTRY_SHA256 = "64991ee0b9e52e61d5238b2c404447d22d633e3adf0fb858757dd2fa5c574b4d"


class PredicateRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))

    def write_registry(self, raw: object, directory: str) -> Path:
        path = Path(directory) / "registry.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        return path

    def claim(self, *, predicate: str, object_value: object) -> dict[str, object]:
        return {
            "claim_id": "claim_001",
            "subject_id": "i_am_maya",
            "speaker_id": "i_am_maya",
            "predicate": predicate,
            "object": object_value,
            "polarity": "positive",
            "epistemic_status": "asserted",
            "valid_from": None,
            "valid_to": None,
            "confidence": 1.0,
            "evidence": [
                {
                    "source_id": "source_001",
                    "message_id": "message_001",
                    "quote": "Source-grounded statement.",
                }
            ],
        }

    def test_loads_reviewed_registry_with_stable_hash_and_counts(self) -> None:
        registry = load_default_predicate_registry()

        self.assertEqual(registry.registry_version, "predicate_registry_v1")
        self.assertEqual(registry.review_status, "implementation_reviewed")
        self.assertEqual(registry.content_sha256, REGISTRY_SHA256)
        self.assertEqual(len(registry.definitions), 63)
        self.assertEqual(
            Counter(item.introduced_in for item in registry.definitions),
            {"phase3_atomic_v2": 36, "benchmark_v1": 27},
        )
        self.assertEqual(
            Counter(item.family for item in registry.definitions),
            {
                "assessment": 7,
                "belief": 3,
                "commitment": 6,
                "event": 2,
                "goal": 4,
                "preference": 1,
                "relationship": 9,
                "role": 9,
                "schedule": 7,
                "state": 9,
                "task": 6,
            },
        )

    def test_rejects_missing_extra_duplicate_and_non_snake_case_entries(self) -> None:
        mutations = {}
        missing = deepcopy(self.raw)
        missing["predicates"][0].pop("family")
        mutations["fields changed"] = missing
        extra = deepcopy(self.raw)
        extra["predicates"][0]["description"] = "extra"
        mutations["fields changed extra"] = extra
        duplicate = deepcopy(self.raw)
        duplicate["predicates"][1]["predicate"] = duplicate["predicates"][0]["predicate"]
        mutations["duplicate predicate"] = duplicate
        bad_name = deepcopy(self.raw)
        bad_name["predicates"][0]["predicate"] = "AcceptedOffer"
        mutations["lowercase snake_case"] = bad_name

        with tempfile.TemporaryDirectory() as directory:
            for expected, raw in mutations.items():
                with self.subTest(expected=expected):
                    path = self.write_registry(raw, directory)
                    with self.assertRaises(PredicateRegistryError) as raised:
                        load_predicate_registry(path)
                    self.assertIn(expected.split()[0], str(raised.exception))

    def test_rejects_unknown_family_and_unsupported_metadata_values(self) -> None:
        fields = {
            "family": (PREDICATE_FAMILIES, "unknown_family"),
            "subject_scope": (SUBJECT_SCOPES, "account"),
            "object_shape": (OBJECT_SHAPES, "free_form"),
            "temporal_behavior": (TEMPORAL_BEHAVIORS, "sometimes"),
            "conflict_compatibility": (
                CONFLICT_COMPATIBILITIES,
                "maybe_conflicting",
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            for field, (allowed, invalid) in fields.items():
                with self.subTest(field=field):
                    self.assertTrue(allowed)
                    raw = deepcopy(self.raw)
                    raw["predicates"][0][field] = invalid
                    with self.assertRaisesRegex(PredicateRegistryError, field):
                        load_predicate_registry(self.write_registry(raw, directory))

    def test_phase3_gold_and_predictions_keep_validating(self) -> None:
        records = []
        for path, field in (
            (REPO_ROOT / "data/phase3/atomic_extraction_gold.jsonl", "expected_claims"),
            (REPO_ROOT / "results/phase3/atomic-extraction-v2/predictions.jsonl", "claims"),
        ):
            for line in path.read_text(encoding="utf-8").splitlines():
                records.extend(json.loads(line)[field])

        predicates = {record["predicate"] for record in records}
        self.assertEqual(len(predicates), 36)
        for record in records:
            with self.subTest(claim_id=record["claim_id"]):
                validate_atomic_claim(record)

    def test_benchmark_v1_predicates_and_object_shapes_are_covered(self) -> None:
        registry = load_default_predicate_registry()
        records = [
            json.loads(line)
            for line in (
                REPO_ROOT / "data/benchmark-v1/gold/claims.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        predicates = {record["predicate"] for record in records}

        self.assertEqual(len(predicates), 27)
        self.assertTrue(predicates <= registry.predicates)
        for record in records:
            definition = registry.by_predicate[record["predicate"]]
            with self.subTest(claim_id=record["claim_id"]):
                self.assertEqual(
                    validate_object_shape(record["object"], definition.object_shape),
                    (),
                )

    def test_runtime_rejects_unknown_predicates_and_wrong_object_shapes(self) -> None:
        with self.assertRaisesRegex(AtomicClaimValidationError, "predicate must be"):
            validate_atomic_claim(
                self.claim(predicate="unreviewed_predicate", object_value="value")
            )
        with self.assertRaisesRegex(AtomicClaimValidationError, "ISO date string"):
            validate_atomic_claim(
                self.claim(predicate="job_start_date", object_value=True)
            )
        with self.assertRaisesRegex(AtomicClaimValidationError, "title and location"):
            validate_atomic_claim(
                self.claim(
                    predicate="has_scheduled_event",
                    object_value={"title": "Review"},
                )
            )

    def test_prompt_rendering_is_deterministic_and_complete(self) -> None:
        registry = load_default_predicate_registry()
        first = render_registry_for_prompt(registry)
        second = render_registry_for_prompt(registry)

        self.assertEqual(first, second)
        self.assertEqual(len(first.splitlines()), 63)
        for definition in registry.definitions:
            line = next(
                line for line in first.splitlines()
                if line.startswith(f"{definition.predicate}:")
            )
            for field in (
                "family",
                "subject_scope",
                "object_shape",
                "temporal_behavior",
                "conflict_compatibility",
                "introduced_in",
            ):
                self.assertIn(f"{field}=", line)

    def test_same_version_semantic_drift_changes_hash_and_is_rejected(self) -> None:
        changed = deepcopy(self.raw)
        changed["predicates"][0]["temporal_behavior"] = "interval"
        with tempfile.TemporaryDirectory() as directory:
            registry = load_predicate_registry(self.write_registry(changed, directory))

        self.assertEqual(registry.registry_version, "predicate_registry_v1")
        self.assertNotEqual(registry.content_sha256, REGISTRY_SHA256)
        with patch("extraction.run_atomic.load_predicate_registry", return_value=registry):
            with self.assertRaisesRegex(AtomicPipelineError, "registry changed"):
                dry_run_atomic(repo_root=REPO_ROOT)

    def test_dry_run_freezes_registry_version_and_hash_without_gold_leakage(self) -> None:
        from io import StringIO

        dry_run = dry_run_atomic(repo_root=REPO_ROOT, stdout=StringIO())
        self.assertEqual(
            dry_run.record["predicate_registry_version"], "predicate_registry_v1"
        )
        self.assertEqual(
            dry_run.record["predicate_registry_sha256"], REGISTRY_SHA256
        )
        self.assertFalse(dry_run.record["oracle_or_gold_fields_in_prompts"])

        forbidden_keys = {
            "acceptable_answers",
            "evidence",
            "oracle_fact_ids",
            "reference_answer",
            "required_claim_ids",
        }
        self.assertFalse(forbidden_keys & set(self.raw))
        for entry in self.raw["predicates"]:
            self.assertFalse(forbidden_keys & set(entry))


if __name__ == "__main__":
    unittest.main()
