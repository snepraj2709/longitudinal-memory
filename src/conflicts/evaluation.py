"""Leakage-safe deterministic evaluation for conflict candidate generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Callable, Mapping, Sequence

from storage.contracts import (
    ClaimRecord,
    ClaimVersionRecord,
    EvidenceLinkRecord,
    ExtractionVersionRecord,
    LIFECYCLE_STATUSES,
    MemoryUser,
    SourceEventRecord,
    SourceSpanRecord,
    safe_json,
)
from storage.migrations import apply_migrations
from storage.repository import StorageRepository

from .candidates import (
    CandidatePair,
    CandidateRequest,
    ConflictCandidateService,
    ELIGIBLE_STATUSES,
)


DATASET_VERSION = "candidate_development_v1"
SPLIT = "development"
DATASET_ROOT = Path("data/conflicts/candidate-development-v1")
RESULT_ROOT = Path("results/conflicts/candidate-generation-development-v1")
DATASET_MANIFEST_SHA256 = "9dd85b3bf49d04ebdd3e0f3a6ea05da4b01fff9901be0d5d9019272bc728f2bb"
RUNTIME_SHA256 = "19aa0171b3ce655f06725e4c555a8e0a5d4b3c9c3c25bc9cc3d7de35e4368a33"
GOLD_SHA256 = "e22bff6a26c34ede3b57076d9bddf74849cbd7f62b3a39457ecf30976559c924"
SCORER_ONLY_PROTECTED_PATH = "data/phase4/temporal-development-v1/gold/cases.jsonl"
PROTECTED_SHA256 = {
    "compose.yaml": "c53c4d37a256e2203f32566a3196c246cd5b2bb67ad8e1839095c9c1c8b0f51a",
    "configs/extraction/predicate_registry_v2.json": "15349ed1f623442dcafedfddfbf9809ea7f44497d5eed0d76bfeff89f57ecfd1",
    "data/phase4/temporal-development-v1/gold/cases.jsonl": "1afce9b37f441925826b0511c8e32b004d825b29e2c8d614f10a5c7ca04ceefa",
    "data/phase4/temporal-development-v1/manifest.json": "785f17876b56ebdf29b8765e104e0160c19befae5271687869e4db9726034f8e",
    "data/phase4/temporal-development-v1/runtime/cases.jsonl": "4c67e1a01f0513512f9c1c3d65bacf8a943f66d037c369182a24adae9b656416",
    "data/scaled-v1/manifest.json": "e3b4386b7063b3c2d65b45574b2e5665fc5094a8330ffd16ea83781744b9a5d3",
    "data/scaled-v1/runtime/sources.jsonl": "a5cbdf38faf22689726c5d5998ea58e2b9e8a19acfae9318511064b5e29235de",
    "data/scaled-v1/runtime/users.jsonl": "13e118ad1e8ecda61616eec51d6ff896ee37321f48a8c187d6c460af898cc3ef",
    "docs/memory-evaluation-steps.md": "bf89021a98273e623edbe27318c9b1cadfb8bed023f5e256a2f58b13e27913ba",
    "migrations/0001_phase4_storage.sql": "6f8a84ce1f78adeccfd7ff15d830dbcf844c34159a0ed34ce11cea4313e95359",
    "migrations/0002_ingestion_reprocessing.sql": "52900567c16e67c654d6fb845b615043beb5d9bf68851eb58231c5740a41253b",
    "migrations/0003_temporal_lifecycle.sql": "2d4e888b262e3dab1a86464fa9de6d33d8818d8978004c5923a5a5e69226fc8e",
    "preference.md": "bf6dfc6ea0b23e9ff1c52b4dbf1debce6ebe495070e826743ffa2d56681a18b8",
    "requirements-storage.txt": "d375af9a0f805ccfd0fbf9a4943cfc6f44b4d7371e6a2376f6b8318d3c73e3d6",
    "results/phase3/phase4-input-development-gpt41-fallback-v1/claims.jsonl": "509c51229eb8a6e13e898a28fec594d0118c29b6f5a93fc917fb0d1af34b4ff7",
    "results/phase3/phase4-input-development-gpt41-fallback-v1/evidence_index.json": "311107940b64e22d9ba8af77e17d04e17f4298e8b5797e6f7234261e852ca0b6",
    "results/phase3/phase4-input-development-gpt41-fallback-v1/failures.jsonl": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "results/phase3/phase4-input-development-gpt41-fallback-v1/manifest.json": "f5127cfdb7720ecf84e320da613396d11d71629b709813e2e25ea2ab00df0876",
    "results/phase3/phase4-input-development-gpt41-fallback-v1/scores.json": "76c96cc9590da0ac40d31a6ff5ab1d96e3decacb732a9bfc9b531e58e8750b4b",
    "results/phase4/step4.2-ingestion-v1/findings.md": "097af6341fd97031ef27802f2bfe79263e60fb377926bb811e94ff07e141010d",
    "results/phase4/step4.2-ingestion-v1/manifest.json": "5c5e27587642af9115e0b5454292afb4b211043ec641b57b97c91573f0796ed2",
    "results/phase4/step4.3-temporal-lifecycle-v1/findings.md": "274b79a17d589474b70683aa77cbb2936ba0eb3d8fd5a8e4bd4c77887a6f9250",
    "results/phase4/step4.3-temporal-lifecycle-v1/manifest.json": "68a527421761dcdc960862f39f589c0283a274e062afe6ca86dc01bbce1c67ad",
    "results/phase4/step4.4-temporal-evaluation-v1/failures.jsonl": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "results/phase4/step4.4-temporal-evaluation-v1/findings.md": "1317ef934f24f8b3f7eb08b49703bde0bc23855ae1cde4e419d1d223d047159f",
    "results/phase4/step4.4-temporal-evaluation-v1/manifest.json": "8f7cc49fbe5620094c618eaaaa98c27ce7a337fdb2747ca9918d8bcfe6d4d644",
    "results/phase4/step4.4-temporal-evaluation-v1/predictions.jsonl": "641e2b1221f6b0123c7521b95b997fa7d4a321faf7aa49f4fc8ab1713726c8b1",
    "results/phase4/step4.4-temporal-evaluation-v1/run.json": "1613ed40d8e9a73c2263aa651400e2240fda9a3ca46174e76d903d49e44cb285",
    "results/phase4/step4.4-temporal-evaluation-v1/scores.json": "23ce6b37522e1367d91e72959b3acfb1a6558597a2667f53c0da9bd970a53d9e",
    "src/evaluation/run_temporal.py": "de52805b3b56a831ff49852f7dbb3f4aedcc3b4999eb6cc3c0a0affe1654cde5",
    "src/evaluation/temporal.py": "997b3e9f5e5080e00b8c65ff3ca662abec5a130fa13b4e07e7f96d3f493eb531",
    "src/ingestion/service.py": "4c2d98417f69896c37506f0b879dae7f306c9d7d61e103036553387ff2a1784e",
    "src/storage/contracts.py": "23122b62d4831bb1ca9a881a6396627d2de609ef954e8c583e599a80989de660",
    "src/storage/repository.py": "c3602486372cccf9b360ad692a05c09b30a6405eefbbb5ebea22a47fd6dd9d3e",
    "src/temporal/__init__.py": "24c692f46335c104e5338d7c639a1890bef83196ebcfd993549f4f574b7508e3",
    "src/temporal/contracts.py": "b0bef262e1332e6435daf7e2dcec602f73d38cfbbe2abec1174d64558b1fda0e",
    "src/temporal/service.py": "7a0836035f1b3e3845dd21f90370d20525ceb8f3508d9c18b17124d1effff61f",
    "tests/integration/test_phase4_storage.py": "ec072b86396f8d1ae1271590d4b4d53870c538bba6e6b10a79b916915c1ee154",
    "tests/integration/test_temporal_evaluation.py": "b826c85ec4d22549ed0f9e8996ac38642e6b1ff56a6b4f86d8cc3ca7af42ea63",
    "tests/integration/test_temporal_service.py": "173c9b4a2d72c22c45a3d05b284b95b31c81b4989b476ef99406bc6801216f90",
    "tests/unit/test_temporal_contracts.py": "f65d619f7f29e0a1cf1c802fdb936b3d4b44005575a68d1eb17035d88b9caafb",
    "tests/unit/test_temporal_evaluation.py": "acfc0957002d5554e08a32ace1bb69760acb3ba3216bd1bf35aa143a7be1c077",
}
REFERENCE_RUNTIME_SHA256 = "4c67e1a01f0513512f9c1c3d65bacf8a943f66d037c369182a24adae9b656416"
DEVELOPMENT_USERS = ("user_001", "user_002")
CASE_IDS = (
    "c51_u1_same_family_repeat",
    "c51_u1_separate_periods",
    "c51_u1_ninety_day_boundary",
    "c51_u1_lexical_threshold",
    "c51_u2_shared_entity_overlap",
    "c51_u2_mixed_unknown",
    "c51_u2_approximate",
    "c51_u2_cross_user_distractor",
)
SIGNAL_COUNT_KEYS = (
    "same_subject",
    "same_predicate_family",
    "shared_entity",
    "temporal_overlap",
    "temporal_within_gap",
    "approximate_time",
    "lexical_at_or_above_threshold",
)
SANITIZED_TOKEN = re.compile(r"^[a-z0-9_:-]+$")
RUNTIME_FIELDS = frozenset(
    {
        "case_id", "dataset_version", "split", "user_id", "transaction_as_of",
        "incoming_claim_ids", "tags", "claims",
    }
)
CLAIM_FIELDS = frozenset(
    {
        "claim_id", "reference_case_id", "reference_claim_id", "user_id",
        "subject_id", "predicate", "object_json", "lifecycle_status",
        "valid_from", "valid_to", "time_precision", "transaction_from",
        "transaction_to", "evidence_refs",
    }
)
EVIDENCE_FIELDS = frozenset({"reference_case_id", "source_id", "span_id"})
GOLD_FIELDS = frozenset(
    {"case_id", "dataset_version", "split", "user_id", "review_status", "required_pairs"}
)
PAIR_FIELDS = frozenset({"left_claim_id", "right_claim_id"})
MANIFEST_FIELDS = frozenset(
    {
        "dataset_version", "review_status", "case_count", "user_counts",
        "case_ids", "runtime", "gold", "reference_runtime", "candidate_config",
        "predicate_registry", "source_boundary",
    }
)


class ConflictEvaluationError(ValueError):
    """Reject malformed evaluation data or leakage-prone sequencing."""


class _RollbackCandidateCase(Exception):
    def __init__(self, prediction: "CandidatePrediction") -> None:
        self.prediction = prediction


@dataclass(frozen=True)
class CandidateEvidenceRef:
    reference_case_id: str
    source_id: str
    span_id: str


@dataclass(frozen=True)
class CandidateRuntimeClaim:
    claim_id: str
    reference_case_id: str
    reference_claim_id: str
    user_id: str
    subject_id: str
    predicate: str
    object_json: object
    lifecycle_status: str
    valid_from: date | datetime | None
    valid_to: date | datetime | None
    time_precision: str
    transaction_from: datetime
    transaction_to: datetime | None
    evidence_refs: tuple[CandidateEvidenceRef, ...]


@dataclass(frozen=True)
class CandidateRuntimeCase:
    case_id: str
    dataset_version: str
    split: str
    user_id: str
    transaction_as_of: datetime
    incoming_claim_ids: tuple[str, ...]
    tags: tuple[str, ...]
    claims: tuple[CandidateRuntimeClaim, ...]


@dataclass(frozen=True)
class CandidateGoldCase:
    case_id: str
    dataset_version: str
    split: str
    user_id: str
    review_status: str
    required_pairs: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class CandidatePrediction:
    case_id: str
    user_id: str
    pairs: tuple[CandidatePair, ...]

    def __post_init__(self) -> None:
        _token(self.case_id, "case_id")
        _token(self.user_id, "user_id")
        if not isinstance(self.pairs, tuple):
            raise ConflictEvaluationError("pairs must be a tuple")
        keys = [(pair.left_claim_id, pair.right_claim_id) for pair in self.pairs]
        if keys != sorted(set(keys)):
            raise ConflictEvaluationError("prediction pairs must be sorted and unique")


@dataclass(frozen=True)
class CandidateFailure:
    failure_id: str
    case_id: str
    user_id: str
    code: str
    location: str

    def __post_init__(self) -> None:
        for name in ("failure_id", "case_id", "user_id", "code", "location"):
            _token(getattr(self, name), name)


@dataclass(frozen=True)
class CandidateScorecard:
    dataset_version: str
    case_count: int
    prediction_count: int
    failure_count: int
    candidate_recall: Mapping[str, object]
    required_pair_count: int
    generated_pair_count: int
    all_possible_same_user_pair_count: int
    pair_reduction: Mapping[str, object]
    candidate_counts_by_signal: Mapping[str, int]
    cross_user_candidate_count: int


def load_candidate_dataset_runtime(
    dataset_root: str | Path,
) -> tuple[Mapping[str, object], tuple[CandidateRuntimeCase, ...]]:
    """Load and hash only runtime-side inputs; never open the separate gold file."""

    root = Path(dataset_root)
    manifest_path = root / "manifest.json"
    manifest = _read_json_object(manifest_path, "candidate dataset manifest")
    if set(manifest) != MANIFEST_FIELDS:
        raise ConflictEvaluationError("candidate dataset manifest fields changed")
    if (
        manifest["dataset_version"] != DATASET_VERSION
        or manifest["review_status"] != "approved"
        or manifest["case_count"] != 8
        or manifest["user_counts"] != {"user_001": 4, "user_002": 4}
        or tuple(manifest["case_ids"]) != CASE_IDS
        or manifest["source_boundary"]
        != "step4.4_runtime_only_no_temporal_gold_scaled_gold_oracle_review_or_test_users"
    ):
        raise ConflictEvaluationError("candidate dataset manifest contract changed")
    for field in ("runtime", "gold", "reference_runtime", "candidate_config", "predicate_registry"):
        value = manifest[field]
        if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
            raise ConflictEvaluationError(f"manifest {field} binding changed")
        if not _sha256_text(value["sha256"]):
            raise ConflictEvaluationError(f"manifest {field} hash is invalid")
    runtime_path = root / str(manifest["runtime"]["path"])
    _require_hash(runtime_path, str(manifest["runtime"]["sha256"]))
    reference_path = root.parents[1] / str(manifest["reference_runtime"]["path"])
    _require_hash(reference_path, str(manifest["reference_runtime"]["sha256"]))
    repository_root = root.parents[2]
    _require_hash(
        repository_root / str(manifest["candidate_config"]["path"]),
        str(manifest["candidate_config"]["sha256"]),
    )
    _require_hash(
        repository_root / str(manifest["predicate_registry"]["path"]),
        str(manifest["predicate_registry"]["sha256"]),
    )
    cases = load_candidate_runtime(runtime_path, reference_runtime_path=reference_path)
    return manifest, cases


def load_candidate_runtime(
    path: str | Path,
    *,
    reference_runtime_path: str | Path,
) -> tuple[CandidateRuntimeCase, ...]:
    reference = _load_reference_runtime(reference_runtime_path)
    cases: list[CandidateRuntimeCase] = []
    for position, raw in enumerate(_read_jsonl(path)):
        _exact(raw, RUNTIME_FIELDS, f"runtime[{position}]")
        case_id = _token(raw["case_id"], "case_id")
        user_id = _development_user(raw["user_id"])
        if raw["dataset_version"] != DATASET_VERSION or raw["split"] != SPLIT:
            raise ConflictEvaluationError("runtime dataset identity changed")
        transaction_as_of = _aware(raw["transaction_as_of"], "transaction_as_of")
        incoming = _string_tuple(raw["incoming_claim_ids"], "incoming_claim_ids")
        if not incoming or len(incoming) != len(set(incoming)):
            raise ConflictEvaluationError("incoming_claim_ids must be nonempty and unique")
        tags = _string_tuple(raw["tags"], "tags")
        if not tags or len(tags) != len(set(tags)):
            raise ConflictEvaluationError("runtime tags must be nonempty and unique")
        claim_values = raw["claims"]
        if not isinstance(claim_values, list) or len(claim_values) < 3:
            raise ConflictEvaluationError("runtime cases require a pair and distractors")
        claims = tuple(
            _runtime_claim(item, reference, f"runtime[{position}].claims[{index}]")
            for index, item in enumerate(claim_values)
        )
        claim_ids = [claim.claim_id for claim in claims]
        if claim_ids != sorted(set(claim_ids)):
            raise ConflictEvaluationError("runtime claim IDs must be sorted and unique")
        by_id = {claim.claim_id: claim for claim in claims}
        if any(
            claim_id not in by_id
            or by_id[claim_id].user_id != user_id
            or by_id[claim_id].lifecycle_status not in ELIGIBLE_STATUSES
            or not _transaction_contains(by_id[claim_id], transaction_as_of)
            for claim_id in incoming
        ):
            raise ConflictEvaluationError("incoming claim must be visible and eligible")
        cases.append(
            CandidateRuntimeCase(
                case_id, DATASET_VERSION, SPLIT, user_id, transaction_as_of,
                incoming, tags, claims,
            )
        )
    _validate_release_cases(cases)
    return tuple(cases)


def load_candidate_gold(path: str | Path) -> tuple[CandidateGoldCase, ...]:
    cases: list[CandidateGoldCase] = []
    for position, raw in enumerate(_read_jsonl(path)):
        _exact(raw, GOLD_FIELDS, f"gold[{position}]")
        case_id = _token(raw["case_id"], "case_id")
        user_id = _development_user(raw["user_id"])
        if (
            raw["dataset_version"] != DATASET_VERSION
            or raw["split"] != SPLIT
            or raw["review_status"] != "approved"
        ):
            raise ConflictEvaluationError("gold review contract changed")
        pairs_value = raw["required_pairs"]
        if not isinstance(pairs_value, list) or len(pairs_value) != 1:
            raise ConflictEvaluationError("each gold case requires one reviewed pair")
        pairs: list[tuple[str, str]] = []
        for index, item in enumerate(pairs_value):
            _exact(item, PAIR_FIELDS, f"gold[{position}].required_pairs[{index}]")
            left = _token(item["left_claim_id"], "left_claim_id")
            right = _token(item["right_claim_id"], "right_claim_id")
            if left >= right:
                raise ConflictEvaluationError("required pair IDs must be canonical")
            pairs.append((left, right))
        cases.append(
            CandidateGoldCase(
                case_id, DATASET_VERSION, SPLIT, user_id, "approved", tuple(pairs)
            )
        )
    _validate_release_cases(cases)
    return tuple(cases)


def score_candidate_generation(
    runtime: Sequence[CandidateRuntimeCase],
    predictions: Sequence[CandidatePrediction],
    failures: Sequence[CandidateFailure],
    gold: Sequence[CandidateGoldCase],
) -> CandidateScorecard:
    _validate_accounting(runtime, predictions, failures)
    if [case.case_id for case in gold] != [case.case_id for case in runtime]:
        raise ConflictEvaluationError("runtime and gold case order differ")
    runtime_by_id = {case.case_id: case for case in runtime}
    gold_by_id = {case.case_id: case for case in gold}
    prediction_by_id = {item.case_id: item for item in predictions}
    required_count = sum(len(case.required_pairs) for case in gold)
    hits = 0
    generated_count = 0
    possible_count = 0
    same_user_generated_count = 0
    cross_user_count = 0
    signal_counts = {key: 0 for key in SIGNAL_COUNT_KEYS}
    for case in runtime:
        gold_case = gold_by_id[case.case_id]
        if gold_case.user_id != case.user_id:
            raise ConflictEvaluationError("runtime and gold users differ")
        claims = {claim.claim_id: claim for claim in case.claims}
        eligible_ids = {
            claim.claim_id
            for claim in case.claims
            if claim.user_id == case.user_id
            and claim.lifecycle_status in ELIGIBLE_STATUSES
            and _transaction_contains(claim, case.transaction_as_of)
            and claim.evidence_refs
        }
        possible_count += len(eligible_ids) * (len(eligible_ids) - 1) // 2
        for left, right in gold_case.required_pairs:
            if left not in eligible_ids or right not in eligible_ids:
                raise ConflictEvaluationError("required pair is not same-user visible runtime data")
        prediction = prediction_by_id.get(case.case_id)
        if prediction is None:
            continue
        if prediction.user_id != case.user_id:
            raise ConflictEvaluationError("prediction user differs from runtime")
        pair_keys = {(pair.left_claim_id, pair.right_claim_id) for pair in prediction.pairs}
        if any(not ({left, right} & set(case.incoming_claim_ids)) for left, right in pair_keys):
            raise ConflictEvaluationError("generated pair does not involve an incoming claim")
        hits += sum(pair in pair_keys for pair in gold_case.required_pairs)
        generated_count += len(prediction.pairs)
        for pair in prediction.pairs:
            if pair.user_id != case.user_id:
                raise ConflictEvaluationError("candidate pair user differs from runtime")
            left = claims.get(pair.left_claim_id)
            right = claims.get(pair.right_claim_id)
            if left is None or right is None:
                raise ConflictEvaluationError("candidate pair references an unknown runtime claim")
            cross_user = left.user_id != case.user_id or right.user_id != case.user_id
            cross_user_count += int(cross_user)
            same_user_generated_count += int(not cross_user)
            signals = pair.signals
            signal_counts["same_subject"] += int(signals.same_subject)
            signal_counts["same_predicate_family"] += int(signals.same_predicate_family)
            signal_counts["shared_entity"] += int(bool(signals.shared_entities))
            signal_counts["temporal_overlap"] += int(signals.temporal_relation == "overlap")
            signal_counts["temporal_within_gap"] += int(signals.temporal_relation == "within_gap")
            signal_counts["approximate_time"] += int(signals.approximate_time)
            signal_counts["lexical_at_or_above_threshold"] += int(signals.lexical_jaccard >= 0.5)
    return CandidateScorecard(
        dataset_version=DATASET_VERSION,
        case_count=len(runtime),
        prediction_count=len(predictions),
        failure_count=len(failures),
        candidate_recall=_metric(hits, required_count, "no_required_pairs"),
        required_pair_count=required_count,
        generated_pair_count=generated_count,
        all_possible_same_user_pair_count=possible_count,
        pair_reduction=_metric(
            possible_count - same_user_generated_count,
            possible_count,
            "no_possible_same_user_pairs",
        ),
        candidate_counts_by_signal=signal_counts,
        cross_user_candidate_count=cross_user_count,
    )


def run_candidate_cases(
    connection: object,
    cases: Sequence[CandidateRuntimeCase],
    *,
    repo_root: str | Path,
    reference_runtime_path: str | Path,
) -> tuple[tuple[CandidatePrediction, ...], tuple[CandidateFailure, ...]]:
    """Run each development case in an isolated rollback transaction."""

    reference = _load_reference_execution_catalog(reference_runtime_path)
    predictions: list[CandidatePrediction] = []
    failures: list[CandidateFailure] = []
    for case in cases:
        try:
            with connection.transaction():
                _insert_runtime_case(connection, case, reference)
                pairs = ConflictCandidateService(
                    connection, repo_root=repo_root
                ).generate(
                    CandidateRequest(
                        case.user_id,
                        case.transaction_as_of,
                        case.incoming_claim_ids,
                    )
                )
                raise _RollbackCandidateCase(
                    CandidatePrediction(case.case_id, case.user_id, pairs)
                )
        except _RollbackCandidateCase as completed:
            predictions.append(completed.prediction)
        except Exception:
            failures.append(
                CandidateFailure(
                    failure_id="failure_"
                    + hashlib.sha256(
                        f"{case.case_id}:case_execution_failed".encode("utf-8")
                    ).hexdigest(),
                    case_id=case.case_id,
                    user_id=case.user_id,
                    code="case_execution_failed",
                    location="case",
                )
            )
    return tuple(predictions), tuple(failures)


def execute_candidate_evaluation(
    connection: object,
    *,
    repo_root: str | Path = ".",
    result_root: str | Path | None = None,
) -> Mapping[str, object]:
    """Execute and freeze the deterministic eight-case development release."""

    root = Path(repo_root).resolve()
    output = root / RESULT_ROOT if result_root is None else Path(result_root).resolve()
    _require_empty_output(output)
    _verify_protected_inputs(root, include_scorer_only=False)
    dataset_root = root / DATASET_ROOT
    _require_hash(dataset_root / "manifest.json", DATASET_MANIFEST_SHA256)
    manifest, runtime = load_candidate_dataset_runtime(dataset_root)
    if (
        manifest["runtime"]["sha256"] != RUNTIME_SHA256
        or manifest["gold"]["sha256"] != GOLD_SHA256
        or manifest["reference_runtime"]["sha256"] != REFERENCE_RUNTIME_SHA256
    ):
        raise ConflictEvaluationError("candidate release input hashes changed")
    apply_migrations(connection, root / "migrations")
    _require_clean_database(connection)
    reference_path = root / "data" / str(manifest["reference_runtime"]["path"])
    predictions, failures = run_candidate_cases(
        connection,
        runtime,
        repo_root=root,
        reference_runtime_path=reference_path,
    )
    gold_path = dataset_root / str(manifest["gold"]["path"])
    scorecard = persist_outputs_before_gold(
        output,
        runtime,
        predictions,
        failures,
        gold_path,
        expected_gold_sha256=GOLD_SHA256,
    )
    _verify_protected_inputs(root, include_scorer_only=True)
    run_path = output / "run.json"
    _write_exclusive(
        run_path,
        _json_bytes(
            {
                "artifact_version": "candidate_generation_development_v1",
                "dataset_version": DATASET_VERSION,
                "starting_commit": "f0f658a356179f363f31a21bab146b446b144f72",
                "case_count": len(runtime),
                "prediction_count": len(predictions),
                "failure_count": len(failures),
                "retries": 0,
                "llm_requests": 0,
                "hosted_writes": 0,
                "gold_opened_after_persisted_results": True,
                "relation_labels_generated": False,
            }
        ),
    )
    findings_path = output / "findings.md"
    _write_exclusive(findings_path, _findings(scorecard).encode("utf-8"))
    artifact_paths = (
        output / "predictions.jsonl",
        output / "failures.jsonl",
        output / "scores.json",
        run_path,
        findings_path,
    )
    implementation_paths = (
        "Makefile",
        "src/conflicts/__init__.py",
        "src/conflicts/candidates.py",
        "src/conflicts/evaluation.py",
        "tests/unit/test_conflict_candidates.py",
        "tests/unit/test_conflict_candidate_evaluation.py",
        "tests/integration/test_conflict_candidates.py",
    )
    release_manifest = {
        "artifact_version": "candidate_generation_development_v1",
        "guidance_version": "step-5.1-guidance-v1",
        "decision_envelope_sha256": "e14a27ee4e7f5b2ddb58ca986aa61f4fbe523a974dbc0c0d1adddd1d5c3b92ec",
        "dataset": {
            "manifest_sha256": DATASET_MANIFEST_SHA256,
            "runtime_sha256": RUNTIME_SHA256,
            "gold_sha256": GOLD_SHA256,
            "reference_runtime_sha256": REFERENCE_RUNTIME_SHA256,
            "case_count": 8,
            "user_counts": {"user_001": 4, "user_002": 4},
        },
        "execution": {
            "prediction_count": len(predictions),
            "failure_count": len(failures),
            "gold_access_boundary": "after_predictions_and_failures_persisted",
            "cross_user_candidate_count": scorecard.cross_user_candidate_count,
            "relation_labels_generated": False,
            "retries": 0,
            "llm_requests": 0,
            "hosted_writes": 0,
        },
        "implementation_hashes": {
            relative: file_sha256(root / relative) for relative in implementation_paths
        },
        "input_hashes": {
            "candidate_config": str(manifest["candidate_config"]["sha256"]),
            "predicate_registry": str(manifest["predicate_registry"]["sha256"]),
        },
        "protected_inputs": dict(sorted(PROTECTED_SHA256.items())),
        "artifacts": {path.name: file_sha256(path) for path in artifact_paths},
        "known_limitations": [
            "This development release has eight reviewed pairs and is not a production workload.",
            "Candidate recall measures whether required pairs survive linking; it does not measure precision.",
            "The linker proposes pairs only. It does not assign relation labels or change lifecycle state.",
        ],
    }
    _write_exclusive(output / "manifest.json", _json_bytes(release_manifest))
    return release_manifest


def persist_outputs_before_gold(
    output_dir: str | Path,
    runtime: Sequence[CandidateRuntimeCase],
    predictions: Sequence[CandidatePrediction],
    failures: Sequence[CandidateFailure],
    gold_path: str | Path,
    *,
    expected_gold_sha256: str | None = None,
    gold_loader: Callable[[str | Path], tuple[CandidateGoldCase, ...]] = load_candidate_gold,
) -> CandidateScorecard:
    """Persist complete outcomes before the first scorer-only gold access."""

    _validate_accounting(runtime, predictions, failures)
    root = Path(output_dir)
    _require_empty_output(root)
    root.mkdir(parents=True, exist_ok=True)
    prediction_path = root / "predictions.jsonl"
    failure_path = root / "failures.jsonl"
    _write_exclusive(prediction_path, serialize_jsonl(predictions))
    _write_exclusive(failure_path, serialize_jsonl(failures))
    _verify_persisted_accounting(runtime, prediction_path, failure_path)
    if expected_gold_sha256 is not None:
        _require_hash(Path(gold_path), expected_gold_sha256)
    gold = gold_loader(gold_path)
    scorecard = score_candidate_generation(runtime, predictions, failures, gold)
    _write_exclusive(root / "scores.json", _json_bytes(asdict(scorecard)))
    return scorecard


def serialize_jsonl(values: Sequence[object]) -> bytes:
    records = [asdict(value) for value in values]
    records.sort(key=lambda item: (str(item.get("case_id", "")), str(item.get("failure_id", ""))))
    return b"".join(_json_bytes(record) for record in records)


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _runtime_claim(
    raw: object,
    reference: Mapping[tuple[str, str], Mapping[str, object]],
    location: str,
) -> CandidateRuntimeClaim:
    if not isinstance(raw, dict):
        raise ConflictEvaluationError(f"{location} must be an object")
    _exact(raw, CLAIM_FIELDS, location)
    claim_id = _token(raw["claim_id"], "claim_id")
    reference_case_id = _token(raw["reference_case_id"], "reference_case_id")
    reference_claim_id = _token(raw["reference_claim_id"], "reference_claim_id")
    user_id = _development_user(raw["user_id"])
    subject_id = _token(raw["subject_id"], "subject_id")
    predicate = _token(raw["predicate"], "predicate")
    if raw["lifecycle_status"] not in LIFECYCLE_STATUSES:
        raise ConflictEvaluationError("runtime lifecycle status is invalid")
    precision = raw["time_precision"]
    valid_from, valid_to = _valid_boundaries(raw["valid_from"], raw["valid_to"], precision)
    transaction_from = _aware(raw["transaction_from"], "transaction_from")
    transaction_to = None if raw["transaction_to"] is None else _aware(raw["transaction_to"], "transaction_to")
    if transaction_to is not None and transaction_to <= transaction_from:
        raise ConflictEvaluationError("runtime transaction interval is invalid")
    evidence_values = raw["evidence_refs"]
    if not isinstance(evidence_values, list) or not evidence_values:
        raise ConflictEvaluationError("runtime claims require evidence references")
    evidence: list[CandidateEvidenceRef] = []
    for index, item in enumerate(evidence_values):
        if not isinstance(item, dict):
            raise ConflictEvaluationError("evidence reference must be an object")
        _exact(item, EVIDENCE_FIELDS, f"{location}.evidence_refs[{index}]")
        evidence.append(
            CandidateEvidenceRef(
                _token(item["reference_case_id"], "reference_case_id"),
                _token(item["source_id"], "source_id"),
                _token(item["span_id"], "span_id"),
            )
        )
    if len(evidence) != len(set(evidence)):
        raise ConflictEvaluationError("evidence references must be unique")
    ref = reference.get((reference_case_id, reference_claim_id))
    if ref is None or ref["user_id"] != user_id:
        raise ConflictEvaluationError("claim reference is missing or cross-user")
    allowed_evidence = ref["evidence"]
    if any(
        item.reference_case_id != reference_case_id
        or (item.source_id, item.span_id) not in allowed_evidence
        for item in evidence
    ):
        raise ConflictEvaluationError("evidence reference is not owned by the source claim")
    return CandidateRuntimeClaim(
        claim_id, reference_case_id, reference_claim_id, user_id, subject_id,
        predicate, safe_json(raw["object_json"], "object_json"),
        str(raw["lifecycle_status"]), valid_from, valid_to, str(precision),
        transaction_from, transaction_to, tuple(evidence),
    )


def _load_reference_runtime(
    path: str | Path,
) -> Mapping[tuple[str, str], Mapping[str, object]]:
    result: dict[tuple[str, str], Mapping[str, object]] = {}
    for case in _read_jsonl(path):
        case_id = case.get("case_id")
        user_id = case.get("user_id")
        spans = {
            item["span_id"]: item["source_id"]
            for item in case.get("spans", [])
            if isinstance(item, dict) and "span_id" in item and "source_id" in item
        }
        for claim in case.get("claims", []):
            if not isinstance(claim, dict) or not isinstance(claim.get("claim_id"), str):
                raise ConflictEvaluationError("reference runtime claim is invalid")
            evidence = frozenset(
                (spans[span_id], span_id)
                for span_id in claim.get("span_ids", [])
                if span_id in spans
            )
            if len(evidence) != len(claim.get("span_ids", [])):
                raise ConflictEvaluationError("reference runtime evidence is invalid")
            key = (str(case_id), claim["claim_id"])
            if key in result:
                raise ConflictEvaluationError("reference runtime claim IDs repeat within a case")
            result[key] = {"user_id": user_id, "evidence": evidence}
    return result


def _load_reference_execution_catalog(
    path: str | Path,
) -> Mapping[tuple[str, str, str], Mapping[str, object]]:
    catalog: dict[tuple[str, str, str], Mapping[str, object]] = {}
    for case in _read_jsonl(path):
        case_id = str(case.get("case_id"))
        user_id = _development_user(case.get("user_id"))
        sources = {
            item["source_id"]: item
            for item in case.get("sources", [])
            if isinstance(item, dict) and isinstance(item.get("source_id"), str)
        }
        for span in case.get("spans", []):
            if not isinstance(span, dict):
                raise ConflictEvaluationError("reference runtime span is invalid")
            source = sources.get(span.get("source_id"))
            if source is None:
                raise ConflictEvaluationError("reference runtime source is missing")
            source_ref = str(source.get("source_ref"))
            source_type = next(
                (
                    item
                    for item in ("conversation", "email", "calendar", "chat")
                    if f"_{item}_" in source_ref
                ),
                None,
            )
            if source_type is None:
                raise ConflictEvaluationError("reference runtime source type is invalid")
            key = (case_id, str(source["source_id"]), str(span.get("span_id")))
            catalog[key] = {
                "user_id": user_id,
                "source_type": source_type,
                "source_ref": source_ref,
                "reference_ingested_at": _aware(
                    source.get("ingested_at"), "reference ingested_at"
                ),
                "message_id": span.get("message_id"),
                "speaker_id": span.get("speaker_id"),
                "quote": span.get("quote"),
            }
    return catalog


def _insert_runtime_case(
    connection: object,
    case: CandidateRuntimeCase,
    reference: Mapping[tuple[str, str, str], Mapping[str, object]],
) -> None:
    repository = StorageRepository(connection)
    created_at = datetime.fromisoformat("2026-01-01T00:00:00+00:00")
    for user_id in sorted({claim.user_id for claim in case.claims}):
        repository.insert_user(MemoryUser(user_id, created_at))
    repository.insert_extraction_version(
        ExtractionVersionRecord(
            version_id="candidate_eval_v1",
            model_version="deterministic",
            prompt_version="none",
            prompt_hash="0" * 64,
            schema_version="candidate_runtime_v1",
            schema_hash="0" * 64,
            registry_version="predicate_registry_v2",
            registry_hash="0" * 64,
            input_manifest_hash="0" * 64,
            created_at=created_at,
        )
    )
    source_claim_times: dict[tuple[str, str], list[datetime]] = {}
    source_details: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    span_details: dict[tuple[str, str], Mapping[str, object]] = {}
    for claim in case.claims:
        for evidence in claim.evidence_refs:
            detail = reference.get(
                (evidence.reference_case_id, evidence.source_id, evidence.span_id)
            )
            if detail is None or detail["user_id"] != claim.user_id:
                raise ConflictEvaluationError("execution evidence reference is invalid")
            source_key = (claim.user_id, evidence.source_id)
            source_claim_times.setdefault(source_key, []).append(claim.transaction_from)
            source_details.setdefault(source_key, []).append(detail)
            span_key = (claim.user_id, evidence.span_id)
            if span_key in span_details and span_details[span_key] != detail:
                raise ConflictEvaluationError("execution span reference changed")
            span_details[span_key] = detail
    for (user_id, source_id), details in sorted(source_details.items()):
        unique_details = {
            (
                str(item["source_type"]),
                str(item["source_ref"]),
                item["reference_ingested_at"],
            )
            for item in details
        }
        if len(unique_details) != 1:
            raise ConflictEvaluationError("execution source reference changed")
        source_type, source_ref, reference_ingested_at = next(iter(unique_details))
        ingested_at = min(reference_ingested_at, min(source_claim_times[(user_id, source_id)]))
        quotes = sorted({str(item["quote"]) for item in details})
        raw_content = "\n".join(quotes)
        repository.insert_source_event(
            SourceEventRecord(
                source_id=source_id,
                user_id=user_id,
                source_type=source_type,
                session_id=None,
                idempotency_key=f"candidate_eval:{case.case_id}:{source_id}",
                produced_at=ingested_at,
                ingested_at=ingested_at,
                raw_content=raw_content,
                participants=[user_id],
                metadata={"reference_source": source_ref},
                content_hash=hashlib.sha256(raw_content.encode("utf-8")).hexdigest(),
            )
        )
    for (user_id, span_id), detail in sorted(span_details.items()):
        source_id = next(
            evidence.source_id
            for claim in case.claims
            if claim.user_id == user_id
            for evidence in claim.evidence_refs
            if evidence.span_id == span_id
        )
        repository.insert_source_span(
            SourceSpanRecord(
                span_id=span_id,
                user_id=user_id,
                source_id=source_id,
                message_id=detail["message_id"],
                speaker_id=str(detail["speaker_id"]),
                verbatim_quote=str(detail["quote"]),
            )
        )
    for claim in case.claims:
        date_from = claim.valid_from if type(claim.valid_from) is date else None
        date_to = claim.valid_to if type(claim.valid_to) is date else None
        timestamp_from = claim.valid_from if isinstance(claim.valid_from, datetime) else None
        timestamp_to = claim.valid_to if isinstance(claim.valid_to, datetime) else None
        repository.insert_claim(
            ClaimRecord(
                claim_id=claim.claim_id,
                user_id=claim.user_id,
                subject_id=claim.subject_id,
                speaker_id=claim.subject_id,
                predicate=claim.predicate,
                predicate_registry_version="predicate_registry_v2",
                object_json=claim.object_json,
                polarity="positive",
                epistemic_status="asserted",
                valid_from_date=date_from,
                valid_from_timestamp=timestamp_from,
                valid_to_date=date_to,
                valid_to_timestamp=timestamp_to,
                time_precision=claim.time_precision,
                extraction_confidence=1,
                memory_kind="durative",
                sensitivity="standard",
                extraction_version_id="candidate_eval_v1",
            )
        )
        repository.insert_claim_version(
            ClaimVersionRecord(
                version_id=f"candidate_eval_version_{claim.claim_id}",
                user_id=claim.user_id,
                claim_id=claim.claim_id,
                lifecycle_status=claim.lifecycle_status,
                transaction_from=claim.transaction_from,
                transaction_to=claim.transaction_to,
                belief_confidence=None,
                valid_from_date=date_from,
                valid_from_timestamp=timestamp_from,
                valid_to_date=date_to,
                valid_to_timestamp=timestamp_to,
                time_precision=claim.time_precision,
            )
        )
        for evidence in claim.evidence_refs:
            repository.insert_evidence_link(
                EvidenceLinkRecord(
                    user_id=claim.user_id,
                    claim_id=claim.claim_id,
                    span_id=evidence.span_id,
                    support_type="supports",
                    extraction_confidence=1,
                )
            )


def _require_clean_database(connection: object) -> None:
    for table in (
        "memory_users", "source_events", "source_spans", "extraction_versions",
        "processing_attempts", "claims", "claim_versions", "evidence_links",
        "claim_extractions", "processing_outbox", "source_tombstones",
        "lifecycle_transitions",
    ):
        row = connection.execute(f"SELECT EXISTS (SELECT 1 FROM {table} LIMIT 1)").fetchone()
        if row is None or row[0]:
            raise ConflictEvaluationError("candidate evaluation requires a clean database")


def _findings(scorecard: CandidateScorecard) -> str:
    recall = scorecard.candidate_recall["value"]
    reduction = scorecard.pair_reduction["value"]
    return (
        "# Candidate generation findings\n\n"
        f"The linker returned all {scorecard.required_pair_count} required pairs "
        f"(recall {recall}). It emitted {scorecard.generated_pair_count} pairs from "
        f"{scorecard.all_possible_same_user_pair_count} possible same-user combinations, "
        f"a reduction of {reduction}. No cross-user pair appeared.\n\n"
        "Each prediction records its signals and visible source IDs. The release has no "
        "relation labels and does not change lifecycle state. This is a small development "
        "check, so it does not estimate production precision.\n"
    )


def _validate_release_cases(cases: Sequence[object]) -> None:
    if tuple(case.case_id for case in cases) != CASE_IDS:
        raise ConflictEvaluationError("candidate development case order changed")
    counts = {user_id: 0 for user_id in DEVELOPMENT_USERS}
    for case in cases:
        counts[case.user_id] += 1
    if counts != {"user_001": 4, "user_002": 4}:
        raise ConflictEvaluationError("candidate development user counts changed")


def _validate_accounting(
    runtime: Sequence[CandidateRuntimeCase],
    predictions: Sequence[CandidatePrediction],
    failures: Sequence[CandidateFailure],
) -> None:
    expected = {case.case_id: case.user_id for case in runtime}
    outcomes = [(item.case_id, item.user_id) for item in (*predictions, *failures)]
    if len(outcomes) != len(set(case_id for case_id, _ in outcomes)):
        raise ConflictEvaluationError("runtime cases require exactly one outcome")
    if {case_id for case_id, _ in outcomes} != set(expected):
        raise ConflictEvaluationError("prediction and failure accounting is incomplete")
    if any(expected[case_id] != user_id for case_id, user_id in outcomes):
        raise ConflictEvaluationError("outcome user differs from runtime")


def _verify_persisted_accounting(
    runtime: Sequence[CandidateRuntimeCase],
    prediction_path: Path,
    failure_path: Path,
) -> None:
    actual: list[str] = []
    for path in (prediction_path, failure_path):
        for item in _read_jsonl(path):
            case_id = item.get("case_id")
            if not isinstance(case_id, str):
                raise ConflictEvaluationError("persisted outcome is malformed")
            actual.append(case_id)
    expected = {case.case_id for case in runtime}
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ConflictEvaluationError("persisted outcomes do not cover runtime cases")


def _valid_boundaries(
    start: object, end: object, precision: object
) -> tuple[date | datetime | None, date | datetime | None]:
    if precision == "unknown":
        if start is not None or end is not None:
            raise ConflictEvaluationError("unknown valid time cannot have boundaries")
        return None, None
    if precision == "timestamp":
        parsed_start = None if start is None else _aware(start, "valid_from")
        parsed_end = None if end is None else _aware(end, "valid_to")
    elif precision in {"day", "month", "year", "approximate"}:
        parsed_start = None if start is None else _date(start, "valid_from")
        parsed_end = None if end is None else _date(end, "valid_to")
    else:
        raise ConflictEvaluationError("time_precision is invalid")
    if parsed_start is None and parsed_end is None:
        raise ConflictEvaluationError("known valid time requires a boundary")
    if parsed_start is not None and parsed_end is not None and parsed_start > parsed_end:
        raise ConflictEvaluationError("valid-time boundaries are unordered")
    return parsed_start, parsed_end


def _transaction_contains(claim: CandidateRuntimeClaim, value: datetime) -> bool:
    return claim.transaction_from <= value and (
        claim.transaction_to is None or value < claim.transaction_to
    )


def _metric(numerator: int, denominator: int, null_reason: str) -> Mapping[str, object]:
    if denominator == 0:
        return {"value": None, "numerator": 0, "denominator": 0, "null_reason": null_reason}
    return {
        "value": round(numerator / denominator, 6),
        "numerator": numerator,
        "denominator": denominator,
        "null_reason": None,
    }


def _read_jsonl(path: str | Path) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    try:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ConflictEvaluationError("JSONL records must be objects")
                records.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise ConflictEvaluationError("evaluation JSONL is unreadable") from error
    return tuple(records)


def _read_json_object(path: Path, name: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConflictEvaluationError(f"{name} is unreadable") from error
    if not isinstance(value, dict):
        raise ConflictEvaluationError(f"{name} must be an object")
    return value


def _exact(value: Mapping[str, object], expected: frozenset[str], location: str) -> None:
    if set(value) != expected:
        raise ConflictEvaluationError(f"{location} fields changed")


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ConflictEvaluationError(f"{name} must be a string list")
    return tuple(value)


def _development_user(value: object) -> str:
    if value not in DEVELOPMENT_USERS:
        raise ConflictEvaluationError("only development users are allowed")
    return str(value)


def _token(value: object, name: str) -> str:
    if not isinstance(value, str) or not SANITIZED_TOKEN.fullmatch(value):
        raise ConflictEvaluationError(f"{name} must be a sanitized token")
    return value


def _aware(value: object, name: str) -> datetime:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ConflictEvaluationError(f"{name} must be an aware timestamp") from error
    else:
        parsed = value
    if not isinstance(parsed, datetime) or parsed.utcoffset() is None:
        raise ConflictEvaluationError(f"{name} must be an aware timestamp")
    return parsed


def _date(value: object, name: str) -> date:
    if not isinstance(value, str):
        raise ConflictEvaluationError(f"{name} must be an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ConflictEvaluationError(f"{name} must be an ISO date") from error
    return parsed


def _require_hash(path: Path, expected: str) -> None:
    if not _sha256_text(expected) or file_sha256(path) != expected:
        raise ConflictEvaluationError(f"protected input hash changed: {path.name}")


def _verify_protected_inputs(root: Path, *, include_scorer_only: bool) -> None:
    for relative, expected in PROTECTED_SHA256.items():
        if relative == SCORER_ONLY_PROTECTED_PATH and not include_scorer_only:
            continue
        _require_hash(root / relative, expected)


def _require_empty_output(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ConflictEvaluationError("candidate result directory must be empty")


def _sha256_text(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _write_exclusive(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise ConflictEvaluationError("evaluation artifacts are immutable") from error
