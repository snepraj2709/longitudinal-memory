from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evaluation.openai_client import OpenAIModelMismatchError, OpenAIResponseMetadata
from extraction.run_atomic import prepare_atomic_run
from extraction.run_phase4_input import (
    FULL_CONFIG_PATH,
    FULL_CONFIG_V1_PATH,
    QUALIFICATION_CONFIG_PATH,
    QUALIFICATION_CONFIG_V1_PATH,
    QUALIFICATION_ROOT,
    QUALIFICATION_ROOT_V1,
    Step35Budget,
    Step35RequestReservation,
    Step35RunError,
    _execute_model,
    _file_sha256,
    _full_budget,
    _load_stage_gold,
    _score_model,
    dry_run_step35,
    execute_full_phase4_input,
    execute_qualification,
    prepare_step35_stage,
)
from extraction.run_safety import (
    cost_text,
    count_step35_request_tokens,
    step35_cost,
    step35_request_sha256,
)


class FakeClient:
    def __init__(self, model: str, *, input_tokens: int = 100) -> None:
        self.model = model
        self.input_tokens = input_tokens
        self.calls = 0

    def complete_with_metadata(self, **_: object) -> object:
        self.calls += 1
        return (
            '{"claims":[]}',
            OpenAIResponseMetadata(
                response_id=f"response_{self.calls}",
                returned_model=self.model,
                input_tokens=self.input_tokens,
                output_tokens=10,
                total_tokens=self.input_tokens + 10,
            ),
        )


@contextmanager
def temporary_repo(*, copy_predecessor: bool = False) -> object:
    repository = Path(".").resolve()
    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "configs").symlink_to(repository / "configs", target_is_directory=True)
        (root / "data").symlink_to(repository / "data", target_is_directory=True)
        phase3 = root / "results/phase3"
        phase3.mkdir(parents=True)
        predecessor = phase3 / QUALIFICATION_ROOT_V1.name
        if copy_predecessor:
            shutil.copytree(repository / QUALIFICATION_ROOT_V1, predecessor)
        else:
            predecessor.symlink_to(
                repository / QUALIFICATION_ROOT_V1, target_is_directory=True
            )
        yield root


def qualification_clients() -> dict[str, FakeClient]:
    return {
        "gpt41": FakeClient("gpt-4.1-2025-04-14"),
        "gpt41-mini": FakeClient("gpt-4.1-mini-2025-04-14"),
    }


def write_json(path: Path, record: object) -> None:
    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for item in records
        ),
        encoding="utf-8",
    )


class Phase4InputRunnerTests(unittest.TestCase):
    def test_dry_run_has_no_calls_writes_or_any_gold_read(self) -> None:
        stdout = io.StringIO()
        gold = Path("data/scaled-v1/gold/claims.jsonl").resolve()
        original = Path.read_bytes

        def deny_gold(path: Path) -> bytes:
            if path.resolve() == gold:
                raise AssertionError("dry-run must not read or hash gold")
            return original(path)

        with patch.object(Path, "read_bytes", deny_gold), patch(
            "extraction.run_phase4_input.load_scaled_development_gold",
            side_effect=AssertionError("dry-run must not load gold"),
        ):
            record = dry_run_step35(stdout=stdout)

        self.assertEqual(record["provider_calls"], 0)
        self.assertEqual(record["output_writes"], 0)
        self.assertTrue(record["fits_cost_cap"])
        self.assertEqual(record["new_request_attempt_count"], 28)
        self.assertEqual(record["new_retry_request_count"], 0)
        self.assertEqual(record["prior_spend_usd"], "0.019646")
        self.assertEqual(record["current_run_hard_maximum_cost_usd"], "0.172894")
        self.assertEqual(record["cumulative_hard_maximum_cost_usd"], "0.192540")
        self.assertTrue(record["recovery"]["prior_spend_counted_once"])
        self.assertEqual(record["token_counter"]["package_version"], "0.13.0")
        self.assertNotIn("OPENAI_API_KEY", stdout.getvalue())

    def test_frozen_models_cases_settings_and_prompt_hashes(self) -> None:
        qualification = prepare_step35_stage(QUALIFICATION_CONFIG_PATH)
        full = prepare_step35_stage(FULL_CONFIG_PATH)

        self.assertEqual((len(qualification.sources), len(full.sources)), (4, 20))
        self.assertEqual(qualification.config.maximum_retry_requests, 0)
        self.assertEqual(full.config.maximum_retry_requests, 0)
        self.assertEqual(
            [model.resolved_model for model in qualification.config.models],
            ["gpt-4.1-2025-04-14", "gpt-4.1-mini-2025-04-14"],
        )
        self.assertEqual(
            qualification.config.prompt_sha256,
            "57ec5307bf3cdf691e303ccd65602a0d3856f95b442c8408e44fd74e9d05ac5b",
        )
        self.assertEqual(
            prepare_atomic_run(validate_gold=False).config.prompt_sha256,
            "e399c3c6108cf2103e844f762c8b34339f5a096a16bd5c91c5a5500c6482bff2",
        )
        registry = json.loads(
            Path("configs/extraction/predicate_registry_v1.json").read_text()
        )
        self.assertEqual(
            hashlib.sha256(
                json.dumps(
                    registry,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "64991ee0b9e52e61d5238b2c404447d22d633e3adf0fb858757dd2fa5c574b4d",
        )

    def test_v1_configs_and_failed_release_remain_byte_identical(self) -> None:
        paths = [
            FULL_CONFIG_V1_PATH,
            QUALIFICATION_CONFIG_V1_PATH,
            *(path for path in QUALIFICATION_ROOT_V1.rglob("*") if path.is_file()),
        ]
        before = {path.as_posix(): _file_sha256(path) for path in paths}

        dry_run_step35(stdout=io.StringIO())

        self.assertEqual(
            before,
            {path.as_posix(): _file_sha256(path) for path in paths},
        )

    def test_config_allowlist_rejects_arbitrary_and_mixed_families(self) -> None:
        with self.assertRaisesRegex(Step35RunError, "allowlist"):
            dry_run_step35(
                qualification_config="configs/extraction/not-frozen.json",
                stdout=io.StringIO(),
            )
        with self.assertRaisesRegex(Step35RunError, "cannot be mixed"):
            dry_run_step35(
                qualification_config=QUALIFICATION_CONFIG_V1_PATH,
                full_config=FULL_CONFIG_PATH,
                stdout=io.StringIO(),
            )

    def test_recovery_predecessor_tamper_blocks_before_any_call(self) -> None:
        with temporary_repo(copy_predecessor=True) as root:
            failure_path = root / QUALIFICATION_ROOT_V1 / "gpt41/failures.jsonl"
            failure_path.write_text(failure_path.read_text() + "\n", encoding="utf-8")
            clients = qualification_clients()
            with self.assertRaisesRegex(Step35RunError, "predecessor artifact changed"):
                execute_qualification(repo_root=root, clients=clients)
            self.assertEqual(sum(client.calls for client in clients.values()), 0)

    def test_each_model_has_its_own_exact_count_hash_and_full_reserve(self) -> None:
        plan = prepare_step35_stage(QUALIFICATION_CONFIG_PATH)
        for model in plan.config.models:
            for index, prompt in enumerate(plan.prompts):
                exact = count_step35_request_tokens(
                    model=model.requested_model,
                    system_prompt=plan.system_prompt,
                    user_prompt=prompt,
                    text_format=plan.text_format,
                    generation_settings=plan.config.generation_settings,
                )
                request_hash = step35_request_sha256(
                    model=model.requested_model,
                    system_prompt=plan.system_prompt,
                    user_prompt=prompt,
                    text_format=plan.text_format,
                    generation_settings=plan.config.generation_settings,
                )
                self.assertEqual(plan.estimated_input_tokens[model.label][index], exact)
                self.assertEqual(
                    plan.maximum_input_tokens[model.label][index], exact + 256
                )
                self.assertEqual(plan.request_body_sha256[model.label][index], request_hash)
        self.assertEqual(
            tuple(
                mini - full
                for full, mini in zip(
                    plan.estimated_input_tokens["gpt41"],
                    plan.estimated_input_tokens["gpt41-mini"],
                )
            ),
            (1, 1, 1, 1),
        )

    def test_qualification_opens_gold_only_after_all_eight_predictions(self) -> None:
        with temporary_repo() as root:
            clients = qualification_clients()
            gold = (root / "data/scaled-v1/gold/claims.jsonl").resolve()
            original_read_bytes = Path.read_bytes
            original_loader = __import__(
                "extraction.run_phase4_input", fromlist=["load_scaled_development_gold"]
            ).load_scaled_development_gold

            def guarded_read(path: Path) -> bytes:
                if path.resolve() == gold:
                    self.assertEqual(sum(client.calls for client in clients.values()), 8)
                return original_read_bytes(path)

            def guarded_load(*args: object, **kwargs: object) -> object:
                self.assertEqual(sum(client.calls for client in clients.values()), 8)
                return original_loader(*args, **kwargs)

            with patch.object(Path, "read_bytes", guarded_read), patch(
                "extraction.run_phase4_input.load_scaled_development_gold",
                side_effect=guarded_load,
            ):
                decision = execute_qualification(repo_root=root, clients=clients)

            output = root / QUALIFICATION_ROOT
            self.assertTrue(decision["passed"])
            self.assertEqual(decision["provider_request_count"], 8)
            self.assertEqual(decision["budget"]["prior_spend_usd"], "0.019646")
            self.assertEqual(
                Decimal(decision["budget"]["cumulative_actual_spend_usd"]),
                Decimal("0.019646") + Decimal(decision["actual_cost_usd"]),
            )
            self.assertEqual(
                {item.name for item in output.iterdir()},
                {
                    "gpt41",
                    "gpt41-mini",
                    "decision.json",
                    "comparison.json",
                    "findings.md",
                    "final_preflight.json",
                    "manifest.json",
                },
            )
            hashes = {
                path.relative_to(output).as_posix(): _file_sha256(path)
                for path in output.rglob("*")
                if path.is_file()
            }
            execute_qualification(repo_root=root, clients=clients, resume=True)
            self.assertEqual(sum(client.calls for client in clients.values()), 8)
            self.assertEqual(
                hashes,
                {
                    path.relative_to(output).as_posix(): _file_sha256(path)
                    for path in output.rglob("*")
                    if path.is_file()
                },
            )

    def test_model_mismatch_stops_and_charges_request_maximum(self) -> None:
        class MismatchedClient:
            calls = 0

            def complete_with_metadata(self, **_: object) -> object:
                self.calls += 1
                raise OpenAIModelMismatchError("different snapshot")

        plan = prepare_step35_stage(QUALIFICATION_CONFIG_PATH)
        model = plan.config.models[0]
        client = MismatchedClient()
        with TemporaryDirectory() as directory:
            output = Path(directory) / "model"
            result = _execute_model(plan, model, client, output, resume=False)
            failure = json.loads((output / "failures.jsonl").read_text())

        self.assertEqual(client.calls, 1)
        self.assertFalse(result["generation_complete"])
        self.assertEqual(result["failure_counts"]["execution"], 1)
        self.assertEqual(
            failure["provider_metadata"]["charged_input_tokens"],
            plan.maximum_input_tokens[model.label][0],
        )
        self.assertEqual(failure["provider_metadata"]["charged_output_tokens"], 1200)

    def test_actual_usage_over_reservation_stops_before_next_call(self) -> None:
        plan = prepare_step35_stage(QUALIFICATION_CONFIG_PATH)
        model = plan.config.models[0]
        client = FakeClient(
            model.resolved_model,
            input_tokens=plan.maximum_input_tokens[model.label][0] + 1,
        )
        with TemporaryDirectory() as directory:
            result = _execute_model(
                plan, model, client, Path(directory) / "model", resume=False
            )

        self.assertEqual(client.calls, 1)
        self.assertFalse(result["generation_complete"])
        self.assertEqual(result["failures"][0]["stage"], "budget")

    def test_missing_usage_charges_the_whole_request_maximum(self) -> None:
        plan = prepare_step35_stage(QUALIFICATION_CONFIG_PATH)
        model = plan.config.models[0]

        class MissingUsageClient:
            calls = 0

            def complete_with_metadata(self, **_: object) -> object:
                self.calls += 1
                return (
                    '{"claims":[]}',
                    OpenAIResponseMetadata(
                        response_id="response_missing_usage",
                        returned_model=model.resolved_model,
                        input_tokens=None,
                        output_tokens=10,
                        total_tokens=None,
                    ),
                )

        client = MissingUsageClient()
        with TemporaryDirectory() as directory:
            output = Path(directory) / "model"
            _execute_model(plan, model, client, output, resume=False)
            first = json.loads((output / "predictions.jsonl").read_text().splitlines()[0])
        self.assertEqual(
            first["provider_metadata"]["charged_input_tokens"],
            plan.maximum_input_tokens[model.label][0],
        )
        self.assertEqual(first["provider_metadata"]["charged_output_tokens"], 1200)

    def test_cumulative_cap_blocks_before_first_call(self) -> None:
        plan = prepare_step35_stage(QUALIFICATION_CONFIG_PATH)
        model = plan.config.models[0]
        reservation = Step35RequestReservation(
            model,
            plan.sources[0].source_id,
            plan.maximum_input_tokens[model.label][0],
        )
        budget = Step35Budget(
            prior_spend_usd=Decimal(0),
            authorization_usd=reservation.maximum_cost_usd - Decimal("0.000001"),
            reservations=(reservation,),
        )
        client = FakeClient(model.resolved_model)
        with TemporaryDirectory() as directory, self.assertRaises(Step35RunError):
            _execute_model(
                plan,
                model,
                client,
                Path(directory) / "model",
                resume=False,
                budget=budget,
            )
        self.assertEqual(client.calls, 0)

    def test_resume_rejects_hash_tamper_and_reordered_predictions(self) -> None:
        plan = prepare_step35_stage(QUALIFICATION_CONFIG_PATH)
        model = plan.config.models[0]
        with TemporaryDirectory() as directory:
            output = Path(directory) / "model"
            _execute_model(
                plan, model, FakeClient(model.resolved_model), output, resume=False
            )
            predictions = json.loads(
                "[" + ",".join((output / "predictions.jsonl").read_text().splitlines()) + "]"
            )
            predictions.reverse()
            write_jsonl(output / "predictions.jsonl", predictions)
            with self.assertRaisesRegex(Step35RunError, "artifact changed"):
                _execute_model(plan, model, object(), output, resume=True)

            run = json.loads((output / "run.json").read_text())
            run["artifact_sha256"]["predictions.jsonl"] = _file_sha256(
                output / "predictions.jsonl"
            )
            run["successful_source_ids"] = [item["source_id"] for item in predictions]
            run["attempted_source_ids"] = [item["source_id"] for item in predictions]
            write_json(output / "run.json", run)
            with self.assertRaisesRegex(Step35RunError, "reordered"):
                _execute_model(plan, model, object(), output, resume=True)

    def test_resume_rejects_duplicate_failure_ids(self) -> None:
        plan = prepare_step35_stage(QUALIFICATION_CONFIG_PATH)
        model = plan.config.models[0]

        class FailedClient:
            def complete_with_metadata(self, **_: object) -> object:
                raise OpenAIModelMismatchError("different snapshot")

        with TemporaryDirectory() as directory:
            output = Path(directory) / "model"
            _execute_model(plan, model, FailedClient(), output, resume=False)
            failure = json.loads((output / "failures.jsonl").read_text())
            write_jsonl(output / "failures.jsonl", [failure, failure])
            run = json.loads((output / "run.json").read_text())
            metadata = failure["provider_metadata"]
            run["artifact_sha256"]["failures.jsonl"] = _file_sha256(
                output / "failures.jsonl"
            )
            run["failure_count"] = 2
            run["attempted_source_ids"] = [failure["source_id"], failure["source_id"]]
            run["input_tokens_charged"] = 2 * metadata["charged_input_tokens"]
            run["output_tokens_charged"] = 2 * metadata["charged_output_tokens"]
            run["cost_usd"] = cost_text(
                step35_cost(
                    model,
                    run["input_tokens_charged"],
                    run["output_tokens_charged"],
                )
            )
            write_json(output / "run.json", run)
            with self.assertRaisesRegex(Step35RunError, "multiple failures"):
                _execute_model(plan, model, object(), output, resume=True)

    def test_completed_checkpoint_is_rescored_and_rejects_score_tamper(self) -> None:
        plan = prepare_step35_stage(QUALIFICATION_CONFIG_PATH)
        model = plan.config.models[0]
        with TemporaryDirectory() as directory:
            output = Path(directory) / "model"
            generated = _execute_model(
                plan, model, FakeClient(model.resolved_model), output, resume=False
            )
            scored = _score_model(plan, model, generated, _load_stage_gold(plan), output)
            scores = json.loads((output / "scores.json").read_text())
            scores["total_unsupported_claims"] = 999
            write_json(output / "scores.json", scores)
            run = json.loads((output / "run.json").read_text())
            run["artifact_sha256"]["scores.json"] = _file_sha256(output / "scores.json")
            write_json(output / "run.json", run)
            with self.assertRaisesRegex(Step35RunError, "refusing to overwrite"):
                _score_model(plan, model, scored, _load_stage_gold(plan), output)

    def test_final_rejects_edited_qualification_decision_before_call(self) -> None:
        with temporary_repo() as root:
            execute_qualification(repo_root=root, clients=qualification_clients())
            decision_path = root / QUALIFICATION_ROOT / "decision.json"
            decision = json.loads(decision_path.read_text())
            decision["passed"] = False
            write_json(decision_path, decision)
            client = FakeClient("gpt-4.1-mini-2025-04-14")
            with self.assertRaisesRegex(Step35RunError, "artifact changed"):
                execute_full_phase4_input(repo_root=root, client=client)
            self.assertEqual(client.calls, 0)

    def test_resume_finishes_finalization_without_replaying_completed_cases(self) -> None:
        with temporary_repo() as root:
            decision = execute_qualification(
                repo_root=root, clients=qualification_clients()
            )
            plan = prepare_step35_stage(FULL_CONFIG_PATH, root)
            model = plan.config.models[0]
            client = FakeClient(model.resolved_model)
            output = root / model.output_directory
            generated = _execute_model(
                plan,
                model,
                client,
                output,
                resume=False,
                budget=_full_budget(plan, Decimal(decision["actual_cost_usd"])),
            )
            _score_model(plan, model, generated, _load_stage_gold(plan), output)
            self.assertFalse((output / "manifest.json").exists())

            result = execute_full_phase4_input(
                repo_root=root, client=client, resume=True
            )

            self.assertTrue(result["generation_complete"])
            self.assertEqual(client.calls, 20)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertIn("configurations", manifest)
            self.assertIn("rendered_prompts", manifest)
            self.assertIn("model", manifest)
            self.assertTrue(
                manifest["reverse_source_to_claim_evidence_index"]["valid"]
            )
            self.assertTrue((output / "known_limitations.json").is_file())
            self.assertTrue((output / "evidence_index.json").is_file())

    def test_resume_refuses_unexpected_nonempty_output(self) -> None:
        plan = prepare_step35_stage(QUALIFICATION_CONFIG_PATH)
        model = plan.config.models[0]
        with TemporaryDirectory() as directory:
            output = Path(directory) / "model"
            _execute_model(
                plan, model, FakeClient(model.resolved_model), output, resume=False
            )
            (output / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
            with self.assertRaisesRegex(Step35RunError, "unexpected artifacts"):
                _execute_model(plan, model, object(), output, resume=True)


if __name__ == "__main__":
    unittest.main()
