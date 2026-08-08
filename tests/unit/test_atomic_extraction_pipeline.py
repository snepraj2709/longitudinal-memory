from __future__ import annotations

from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation.openai_client import OpenAIResponseMetadata
from extraction.gold import ATOMIC_GOLD_PATH, load_atomic_gold
from extraction.run_atomic import (
    ATOMIC_CASE_REFS,
    AtomicPipelineError,
    dry_run_atomic,
    execute_atomic_pipeline,
    main,
)
from extraction.scoring import score_atomic_extraction
from extraction.source import load_pilot_sources


REPO_ROOT = Path(__file__).resolve().parents[2]
B1_RESULT_DIR = REPO_ROOT / "results/pilot/b1-full-history"


class FakeAtomicPipelineClient:
    def __init__(
        self,
        *,
        model: str = "gpt-4.1-2025-04-14",
        provider_failures_at: set[int] | None = None,
        validation_failures_at: set[int] | None = None,
    ) -> None:
        self.model = model
        self.provider_failures_at = provider_failures_at or set()
        self.validation_failures_at = validation_failures_at or set()
        self.calls: list[tuple[str, str]] = []

    def complete_with_metadata(
        self, *, system_prompt: str, user_prompt: str
    ) -> tuple[str, OpenAIResponseMetadata]:
        self.calls.append((system_prompt, user_prompt))
        position = len(self.calls)
        if position in self.provider_failures_at:
            raise RuntimeError("sk-proj-" + "a" * 48)
        raw_response = (
            "sk-proj-" + "b" * 48
            if position in self.validation_failures_at
            else '{"claims":[]}'
        )
        return raw_response, OpenAIResponseMetadata(
            response_id=f"resp_fake_{position:03d}",
            returned_model=self.model,
            input_tokens=100,
            output_tokens=10,
            total_tokens=110,
        )


class AtomicExtractionPipelineTests(unittest.TestCase):
    def execute(
        self,
        output_dir: Path,
        client: FakeAtomicPipelineClient,
        *,
        started_at: str = "2026-08-08T10:00:00+00:00",
        completed_at: str = "2026-08-08T10:01:00+00:00",
    ) -> dict[str, object]:
        timestamps = iter(
            (datetime.fromisoformat(started_at), datetime.fromisoformat(completed_at))
        )
        return execute_atomic_pipeline(
            client=client,
            requested_model=client.model,
            output_dir=output_dir,
            repo_root=REPO_ROOT,
            now=lambda: next(timestamps),
        )

    def test_dry_run_makes_no_calls_or_writes(self) -> None:
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            result_dir = Path(directory) / "results"
            with (
                patch("extraction.run_atomic.DEFAULT_OUTPUT_DIR", result_dir),
                patch("extraction.run_atomic.OpenAIResponsesClient") as client_type,
            ):
                self.assertEqual(main(["--dry-run"], stdout=output), 0)

            client_type.assert_not_called()
            self.assertFalse(result_dir.exists())

        record = json.loads(output.getvalue())
        self.assertTrue(record["dry_run"])
        self.assertEqual(record["provider_calls"], 0)
        self.assertEqual(record["output_writes"], 0)
        self.assertEqual(
            [(item["case_id"], item["source_id"]) for item in record["case_order"]],
            list(ATOMIC_CASE_REFS),
        )

    def test_success_calls_ten_sources_in_order_then_loads_gold(self) -> None:
        client = FakeAtomicPipelineClient()
        real_loader = load_atomic_gold

        def load_gold_after_calls(*args: object, **kwargs: object) -> object:
            self.assertEqual(len(client.calls), 10)
            return real_loader(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "extraction.run_atomic.load_atomic_gold",
                side_effect=load_gold_after_calls,
            ) as gold_loader:
                self.execute(Path(directory) / "results", client)

        self.assertEqual(gold_loader.call_count, 1)
        self.assertEqual(len(client.calls), 10)
        prompted_sources = [
            json.loads(user_prompt.split("\n", 1)[1])["source_id"]
            for _, user_prompt in client.calls
        ]
        self.assertEqual(
            prompted_sources,
            [source_id for _, source_id in ATOMIC_CASE_REFS],
        )

        prompt_text = "\n".join(
            text for call in client.calls for text in call
        )
        self.assertNotIn("expected_claims", prompt_text)
        gold_cases = real_loader(
            REPO_ROOT / ATOMIC_GOLD_PATH,
            load_pilot_sources(),
        )
        for case in gold_cases:
            self.assertNotIn(case.case_id, prompt_text)
            for claim in case.expected_claims:
                self.assertNotIn(claim.claim_id, prompt_text)

    def test_success_writes_ordered_outputs_and_step_6_scores(self) -> None:
        client = FakeAtomicPipelineClient()
        before_b1 = self.hash_tree(B1_RESULT_DIR)
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "results"
            run = self.execute(output_dir, client)

            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {
                    "predictions.jsonl",
                    "case_scores.jsonl",
                    "scores.json",
                    "run.json",
                    "failures.jsonl",
                },
            )
            predictions = self.read_jsonl(output_dir / "predictions.jsonl")
            self.assertEqual(
                [(item["case_id"], item["source_id"]) for item in predictions],
                list(ATOMIC_CASE_REFS),
            )
            self.assertEqual(
                self.read_jsonl(output_dir / "case_scores.jsonl"),
                json.loads((output_dir / "scores.json").read_text())["case_results"],
            )
            self.assertEqual(
                (output_dir / "failures.jsonl").read_text(encoding="utf-8"),
                "",
            )

            gold_cases = load_atomic_gold(
                REPO_ROOT / ATOMIC_GOLD_PATH,
                load_pilot_sources(),
            )
            expected_scores = score_atomic_extraction(
                gold_cases,
                {case.case_id: () for case in gold_cases},
            )
            self.assertEqual(
                json.loads((output_dir / "scores.json").read_text()),
                expected_scores,
            )
            self.assertEqual(run, json.loads((output_dir / "run.json").read_text()))
            self.assertEqual(run["run_status"], "completed")
            self.assertEqual(run["prompt_version"], "atomic-extraction-v3")
            self.assertEqual(run["provider_requests_attempted"], 10)
            self.assertEqual(run["successful_cases"], 10)
            self.assertEqual(run["failed_cases"], 0)
            self.assertEqual(len(run["attempts"]), 10)
            self.assertEqual(run["resume_count"], 0)

        self.assertEqual(self.hash_tree(B1_RESULT_DIR), before_b1)

    def test_smoke_config_calls_only_three_frozen_development_cases(self) -> None:
        client = FakeAtomicPipelineClient()
        config_path = REPO_ROOT / "configs/extraction/atomic_extraction_smoke_v1.json"
        times = iter(
            (
                datetime.fromisoformat("2026-08-08T10:00:00+00:00"),
                datetime.fromisoformat("2026-08-08T10:01:00+00:00"),
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "smoke"
            run = execute_atomic_pipeline(
                client=client,
                requested_model=client.model,
                output_dir=output_dir,
                repo_root=REPO_ROOT,
                now=lambda: next(times),
                config_path=config_path,
            )
            scores = json.loads((output_dir / "scores.json").read_text())

        expected = ["conv_002", "email_005", "conv_010"]
        prompted = [
            json.loads(user_prompt.split("\n", 1)[1])["source_id"]
            for _, user_prompt in client.calls
        ]
        self.assertEqual(prompted, expected)
        self.assertEqual(run["planned_request_count"], 3)
        self.assertEqual(run["successful_cases"], 3)
        self.assertEqual(len(scores["case_results"]), 3)

    def test_failure_stops_without_retrying_or_scoring(self) -> None:
        client = FakeAtomicPipelineClient(provider_failures_at={4})
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "results"
            run = self.execute(output_dir, client)

        prompted_sources = [
            json.loads(user_prompt.split("\n", 1)[1])["source_id"]
            for _, user_prompt in client.calls
        ]
        expected_sources = [source_id for _, source_id in ATOMIC_CASE_REFS]
        self.assertEqual(prompted_sources, expected_sources[:4])
        self.assertEqual(run["run_status"], "completed_with_provider_failures")
        self.assertEqual(run["provider_requests_attempted"], 4)
        self.assertEqual(run["successful_cases"], 3)

    def test_failures_are_sanitized_and_not_scored(self) -> None:
        for failure_kind in ("provider", "validation"):
            with self.subTest(failure_kind=failure_kind):
                options = {f"{failure_kind}_failures_at": {4}}
                client = FakeAtomicPipelineClient(**options)
                with tempfile.TemporaryDirectory() as directory:
                    output_dir = Path(directory) / "results"
                    with patch("extraction.run_atomic.load_atomic_gold") as loader:
                        run = self.execute(output_dir, client)

                    loader.assert_not_called()
                    self.assertEqual(len(client.calls), 4)
                    self.assertEqual(
                        run["run_status"],
                        "completed_with_provider_failures"
                        if failure_kind == "provider"
                        else "failed_validation",
                    )
                    self.assertEqual(run["provider_requests_attempted"], 4)
                    self.assertEqual(run["successful_cases"], 3)
                    self.assertEqual(run["failed_cases"], 1)
                    self.assertFalse((output_dir / "scores.json").exists())
                    self.assertFalse((output_dir / "case_scores.jsonl").exists())

                    artifact_text = "\n".join(
                        path.read_text(encoding="utf-8")
                        for path in output_dir.iterdir()
                    )
                    self.assertNotIn("sk-proj-", artifact_text)
                    self.assertNotIn("raw_response", artifact_text)
                    failures = self.read_jsonl(output_dir / "failures.jsonl")
                    self.assertEqual(failures[0]["failure_stage"], failure_kind)
                    expected_diagnostics = (
                        []
                        if failure_kind == "provider"
                        else [
                            {
                                "code": "response_invalid_json",
                                "location": "response",
                            }
                        ]
                    )
                    self.assertEqual(
                        failures[0]["validation_diagnostics"],
                        expected_diagnostics,
                    )

    def test_non_empty_output_directory_is_not_overwritten(self) -> None:
        client = FakeAtomicPipelineClient()
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "results"
            output_dir.mkdir()
            marker = output_dir / "keep.txt"
            marker.write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(AtomicPipelineError, "refusing to overwrite"):
                self.execute(output_dir, client)

            self.assertEqual(client.calls, [])
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
            self.assertEqual(list(output_dir.iterdir()), [marker])

    def test_fake_runs_are_deterministic_except_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first_dir = Path(directory) / "first"
            second_dir = Path(directory) / "second"
            self.execute(first_dir, FakeAtomicPipelineClient())
            self.execute(
                second_dir,
                FakeAtomicPipelineClient(),
                started_at="2026-08-09T10:00:00+00:00",
                completed_at="2026-08-09T10:01:00+00:00",
            )

            for filename in (
                "predictions.jsonl",
                "case_scores.jsonl",
                "scores.json",
                "failures.jsonl",
            ):
                self.assertEqual(
                    (first_dir / filename).read_bytes(),
                    (second_dir / filename).read_bytes(),
                )

            first_run = json.loads((first_dir / "run.json").read_text())
            second_run = json.loads((second_dir / "run.json").read_text())
            for record in (first_run, second_run):
                record.pop("started_at")
                record.pop("completed_at")
            self.assertEqual(first_run, second_run)

    @staticmethod
    def read_jsonl(path: Path) -> list[dict[str, object]]:
        return [json.loads(line) for line in path.read_text().splitlines()]

    @staticmethod
    def hash_tree(path: Path) -> dict[str, str]:
        return {
            file.relative_to(path).as_posix(): hashlib.sha256(
                file.read_bytes()
            ).hexdigest()
            for file in sorted(path.rglob("*"))
            if file.is_file()
        }


if __name__ == "__main__":
    unittest.main()
