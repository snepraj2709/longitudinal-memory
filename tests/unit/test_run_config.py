from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest

from evaluation.history import (
    SOURCE_ORDERING_RULE,
    HistoryObservation,
    history_observation_sort_key,
)
from evaluation.run_config import (
    FULL_HISTORY_GENERATION_SETTINGS,
    PILOT_DATASET_FILES,
    CompletedStep3Run,
    RunConfigurationError,
    assert_no_controlled_drift,
    build_frozen_config,
    build_run_manifest,
    canonical_sha256,
    configuration_sha256,
    dataset_sha256,
    discover_pilot_jsonl_files,
    inspect_completed_step3_run,
    prompt_sha256,
    validate_frozen_config,
    validate_run_manifest,
    verify_manifest_artifacts,
)
from evaluation.smoke import SMOKE_CASE_IDS


class RunConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.repo_root = Path(self.temporary_directory.name)
        for index, relative_path in enumerate(PILOT_DATASET_FILES):
            path = self.repo_root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f'{{"record":{index}}}\n', encoding="utf-8")
        self.completed_run = CompletedStep3Run(
            provider="OpenAI",
            requested_model="gpt-4.1-2025-04-14",
            returned_model="gpt-4.1-2025-04-14",
            temperature=0.0,
            generation_settings=FULL_HISTORY_GENERATION_SETTINGS,
        )
        self.config = build_frozen_config(
            repo_root=self.repo_root,
            completed_run=self.completed_run,
        )

    def _config_record(self) -> dict[str, object]:
        record = dict(vars(self.config))
        record["generation_settings"] = dict(self.config.generation_settings)
        record["dataset_files"] = list(self.config.dataset_files)
        return record

    def _rehash(self, record: dict[str, object]) -> dict[str, object]:
        record["configuration_sha256"] = canonical_sha256(
            {key: value for key, value in record.items() if key != "configuration_sha256"}
        )
        return record

    def _valid_manifest(self) -> dict[str, object]:
        return {
            "manifest_version": "1",
            "run_id": "20260807T083918Z-gpt-4.1-2025-04-14",
            "run_date": "2026-08-07T08:39:18Z",
            "repository_commit": "a" * 40,
            "prompt_files_dirty": False,
            "dataset_files_dirty": False,
            "frozen_configuration_path": "configs/full_history_baseline_v1.json",
            "frozen_configuration_sha256": self.config.configuration_sha256,
            "output_artifacts": {
                "predictions": "results/run/predictions.jsonl",
                "diagnostics": "results/run/diagnostics.jsonl",
                "api_metadata": "results/run/api_metadata.jsonl",
                "report": "results/run/report.md",
            },
            "step3_exit_status": 0,
            "returned_model": "gpt-4.1-2025-04-14",
        }

    def test_valid_config_has_all_required_fields_and_exact_settings(self) -> None:
        validated = validate_frozen_config(self._config_record())

        self.assertEqual(validated.temperature, 0.0)
        self.assertEqual(dict(validated.generation_settings), FULL_HISTORY_GENERATION_SETTINGS)
        self.assertEqual(validated.dataset_files, PILOT_DATASET_FILES)

    def test_config_rejects_missing_and_unknown_fields_together(self) -> None:
        record = self._config_record()
        del record["provider"]
        record["extra"] = "value"

        with self.assertRaises(RunConfigurationError) as raised:
            validate_frozen_config(record)

        self.assertIn("unknown fields: extra", str(raised.exception))
        self.assertIn("missing fields: provider", str(raised.exception))

    def test_config_rejects_generation_setting_drift(self) -> None:
        record = self._config_record()
        record["generation_settings"]["max_output_tokens"] = 999
        self._rehash(record)

        with self.assertRaisesRegex(RunConfigurationError, "exactly match"):
            validate_frozen_config(record)

    def test_manifest_requires_timezone_aware_run_date(self) -> None:
        record = self._valid_manifest()
        record["run_date"] = "2026-08-07T08:39:18"

        with self.assertRaisesRegex(RunConfigurationError, "UTC offset"):
            validate_run_manifest(record, config=self.config)

    def test_dataset_hash_is_deterministic_and_input_order_invariant(self) -> None:
        forward = dataset_sha256(self.repo_root, PILOT_DATASET_FILES)
        reverse = dataset_sha256(self.repo_root, tuple(reversed(PILOT_DATASET_FILES)))

        self.assertEqual(forward, reverse)
        self.assertEqual(forward, dataset_sha256(self.repo_root, PILOT_DATASET_FILES))
        self.assertEqual(discover_pilot_jsonl_files(self.repo_root), PILOT_DATASET_FILES)

    def test_dataset_hash_changes_when_one_byte_changes(self) -> None:
        before = dataset_sha256(self.repo_root, PILOT_DATASET_FILES)
        path = self.repo_root / PILOT_DATASET_FILES[0]
        path.write_bytes(path.read_bytes() + b" ")

        self.assertNotEqual(before, dataset_sha256(self.repo_root, PILOT_DATASET_FILES))

    def test_prompt_hash_is_deterministic_and_covers_both_static_prompts(self) -> None:
        baseline = prompt_sha256("system", "template")

        self.assertEqual(baseline, prompt_sha256("system", "template"))
        self.assertNotEqual(baseline, prompt_sha256("system changed", "template"))
        self.assertNotEqual(baseline, prompt_sha256("system", "template changed"))

    def test_declared_source_ordering_rule_matches_history_sort(self) -> None:
        observed_at = datetime.fromisoformat("2026-01-01T10:00:00+05:30")
        observations = (
            HistoryObservation(
                observed_at=observed_at,
                source_type="conversation",
                source_id="source_b",
                message_id="msg_b",
                author_id="user",
                author_name="User",
                text="B",
            ),
            HistoryObservation(
                observed_at=observed_at,
                source_type="calendar",
                source_id="source_a",
                message_id=None,
                author_id="user",
                author_name="User",
                text="Calendar",
            ),
            HistoryObservation(
                observed_at=observed_at,
                source_type="conversation",
                source_id="source_a",
                message_id="msg_a",
                author_id="user",
                author_name="User",
                text="A",
            ),
        )

        ordered = sorted(observations, key=history_observation_sort_key)

        self.assertEqual(
            SOURCE_ORDERING_RULE,
            "observed_at ascending, then source_id ascending, then message_id ascending, "
            "with a calendar null message_id treated as an empty string for ordering.",
        )
        self.assertEqual(
            [(item.source_id, item.message_id) for item in ordered],
            [("source_a", None), ("source_a", "msg_a"), ("source_b", "msg_b")],
        )

    def test_configuration_hash_is_deterministic(self) -> None:
        self.assertEqual(
            self.config.configuration_sha256,
            configuration_sha256(self.config),
        )
        self.assertEqual(
            self.config.configuration_sha256,
            configuration_sha256(self._config_record()),
        )

    def test_same_version_rejects_controlled_drift(self) -> None:
        record = self._config_record()
        record["resolved_model"] = "gpt-other"
        changed = validate_frozen_config(self._rehash(record))

        with self.assertRaisesRegex(RunConfigurationError, "requires a new"):
            assert_no_controlled_drift(self.config, changed)

    def test_new_version_allows_intentional_change(self) -> None:
        record = self._config_record()
        record["configuration_version"] = "2"
        record["resolved_model"] = "gpt-other"
        changed = validate_frozen_config(self._rehash(record))

        assert_no_controlled_drift(self.config, changed)

    def test_secret_bearing_fields_and_values_are_rejected(self) -> None:
        record = self._config_record()
        record["api_key"] = "not-even-a-real-key"
        record["provider"] = "Bearer hidden"

        with self.assertRaises(RunConfigurationError) as raised:
            validate_frozen_config(record)

        self.assertIn("secret-bearing field", str(raised.exception))
        self.assertIn("appears to contain a secret", str(raised.exception))

    def test_manifest_requires_exact_artifact_references(self) -> None:
        record = self._valid_manifest()
        del record["output_artifacts"]["diagnostics"]
        record["output_artifacts"]["other"] = "results/run/other.txt"

        with self.assertRaises(RunConfigurationError) as raised:
            validate_run_manifest(record, config=self.config)

        self.assertIn("unknown fields: other", str(raised.exception))
        self.assertIn("missing fields: diagnostics", str(raised.exception))

    def test_build_manifest_links_successful_run_to_frozen_config(self) -> None:
        record = self._valid_manifest()
        manifest = build_run_manifest(
            run_id=record["run_id"],
            run_date=record["run_date"],
            repository_commit=record["repository_commit"],
            prompt_files_dirty=record["prompt_files_dirty"],
            dataset_files_dirty=record["dataset_files_dirty"],
            frozen_configuration_path=record["frozen_configuration_path"],
            config=self.config,
            output_artifacts=record["output_artifacts"],
            step3_exit_status=record["step3_exit_status"],
            returned_model=record["returned_model"],
        )

        self.assertEqual(
            manifest.frozen_configuration_sha256,
            self.config.configuration_sha256,
        )

    def test_manifest_artifact_references_must_exist(self) -> None:
        manifest = validate_run_manifest(self._valid_manifest(), config=self.config)

        with self.assertRaisesRegex(RunConfigurationError, "does not exist"):
            verify_manifest_artifacts(manifest, self.repo_root)

    def test_fake_or_dry_run_metadata_cannot_be_frozen(self) -> None:
        run_dir = self.repo_root / "results" / "fake"
        run_dir.mkdir(parents=True)
        predictions = [{"case_id": case_id} for case_id in SMOKE_CASE_IDS]
        diagnostics = [
            {
                "case_id": case_id,
                "valid_json": True,
                "valid_contract": True,
                "case_id_matches": True,
                "exact_evidence": True,
                "validation_error": None,
            }
            for case_id in SMOKE_CASE_IDS
        ]
        metadata = [
            {
                "case_id": case_id,
                "response_id": f"fake-{index}",
                "returned_model": "gpt-4.1-2025-04-14",
            }
            for index, case_id in enumerate(SMOKE_CASE_IDS)
        ]
        for filename, records in (
            ("predictions.jsonl", predictions),
            ("diagnostics.jsonl", diagnostics),
            ("api_metadata.jsonl", metadata),
        ):
            (run_dir / filename).write_text(
                "\n".join(json.dumps(item) for item in records) + "\n",
                encoding="utf-8",
            )
        (run_dir / "report.md").write_text(
            "Provider: OpenAI\n"
            "Model: gpt-4.1-2025-04-14\n"
            "Settings: `"
            + json.dumps(
                {**FULL_HISTORY_GENERATION_SETTINGS, "temperature": 0.0},
                sort_keys=True,
            )
            + "`\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(RunConfigurationError, "not a live response"):
            inspect_completed_step3_run(run_dir)


if __name__ == "__main__":
    unittest.main()
