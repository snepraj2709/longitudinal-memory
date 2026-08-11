"""Build a deterministic, sanitized demo bundle from versioned artifacts."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable, Mapping


CONFIG = Path("configs/demo/demo_v1.json")
QUESTIONS = Path("data/pilot/evaluation/eval_questions.jsonl")
ANSWERS = Path("data/pilot/evaluation/eval_answer.jsonl")
PREDICTIONS = Path("results/pilot/b1-full-history/predictions.jsonl")
FAILURES = Path("results/pilot/b1-full-history/failure_analysis.jsonl")
SOURCES = (
    Path("data/pilot/sources/conversations.jsonl"),
    Path("data/pilot/sources/emails.jsonl"),
    Path("data/pilot/sources/calendar.jsonl"),
)
B1_SCORES = Path("results/pilot/b1-full-history/scores.json")
RETRIEVAL_SCORES = Path("results/retrieval/retrieval-quality-development-v1/scorecard.json")
ABSTENTION_SCORES = Path("results/abstention/b7-evaluation-development-v1/scorecard.json")
OPENAI_RUN = Path("results/evaluation/openai-step10.3-interrupted-v1/manifest.json")
QWEN_CONFIG = Path("configs/evaluation/qwen35_27b_fp8_v1.json")
QWEN3_RUN = Path("results/evaluation/qwen3-8b-vllm-dev-v1/run-manifest.json")
QWEN3_METRICS = Path("results/evaluation/qwen3-8b-vllm-dev-v1/scores/metrics.jsonl")
DEFAULT_OUTPUT = Path("results/demo/demo-v1/bundle.json")


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected object rows in {path}")
        rows.append(value)
    return rows


def _by(rows: Iterable[dict[str, object]], key: str) -> dict[str, dict[str, object]]:
    return {str(row[key]): row for row in rows}


def _source_index(root: Path) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    sources = _by((row for path in SOURCES for row in _jsonl(root / path)), "source_id")
    messages: dict[str, dict[str, object]] = {}
    for source_id, source in sources.items():
        people = {
            str(item["participant_id"]): str(item["display_name"])
            for item in source.get("participants", [])
            if isinstance(item, dict)
        }
        payload = source.get("payload", {})
        if not isinstance(payload, dict):
            continue
        for message in payload.get("messages", []):
            if not isinstance(message, dict):
                continue
            message_id = str(message["message_id"])
            speaker_id = str(message.get("speaker_id") or message.get("from_id") or "unknown")
            messages[f"{source_id}:{message_id}"] = {
                "evidence_id": f"{source_id}:{message_id}",
                "source_id": source_id,
                "message_id": message_id,
                "source_type": source["source_type"],
                "observed_at": message.get("sent_at") or source["created_at"],
                "speaker": people.get(speaker_id, speaker_id),
                "text": message.get("content") or message.get("body") or "",
            }
    return sources, messages


def _timeline(evidence_ids: list[str], messages: Mapping[str, dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for evidence_id in dict.fromkeys(evidence_ids):
        if evidence_id not in messages:
            raise ValueError(f"unknown demo evidence: {evidence_id}")
        rows.append(dict(messages[evidence_id]))
    return sorted(rows, key=lambda item: (str(item["observed_at"]), str(item["evidence_id"])))


def _review(failure: dict[str, object] | None, answer: dict[str, object]) -> dict[str, object]:
    if failure is None:
        return {
            "result": "correct",
            "failure_type": None,
            "explanation": "The model abstained because the source history does not contain the answer.",
            "expected_answer": answer["reference_answer"],
        }
    return {
        "result": failure["result"],
        "failure_type": failure["primary_classification"],
        "explanation": failure["concise_explanation"],
        "expected_answer": failure["reference_answer"],
    }


def _scorecards(root: Path) -> list[dict[str, object]]:
    b1 = _json(root / B1_SCORES)
    retrieval = _json(root / RETRIEVAL_SCORES)
    abstention = _json(root / ABSTENTION_SCORES)
    qwen3_metrics = _jsonl(root / QWEN3_METRICS)
    overall = {
        str(row["baseline_id"]): row
        for row in retrieval["quality_rows"]
        if row["slice_dimension"] == "overall" and row["slice_value"] == "all"
    }
    abstention_metrics = {
        f"{row['baseline_id']}:{row['metric_name']}": row["value"]
        for row in abstention["metrics"]
        if row["scope"] == "baseline"
    }
    rows = [
        {
            "series_id": "historical-development",
            "baseline_id": "B1",
            "label": "Full history",
            "status": "pilot_scored",
            "answer_accuracy": b1["answer_correctness"]["strict_accuracy"]["value"],
            "evidence_recall": b1["evidence_message"]["micro_recall"]["value"],
            "abstention_precision": b1["abstention"]["precision"]["value"],
            "recall_at_10": None,
            "latency_ms": None,
            "failures": b1["execution_failure_count"],
        }
    ]
    for baseline in ("B2", "B3", "B4"):
        metrics = overall[baseline]["metrics"]
        latency = next(
            item for item in retrieval["latency_rows"]
            if item["baseline_id"] == baseline and item["slice_dimension"] == "overall"
        )
        rows.append({
            "series_id": "historical-development",
            "baseline_id": baseline,
            "label": {"B2": "Atomic memory", "B3": "Session summaries", "B4": "Hybrid retrieval"}[baseline],
            "status": "development_scored",
            "answer_accuracy": None,
            "evidence_recall": None,
            "abstention_precision": None,
            "recall_at_10": float(metrics["recall_at_10"]["value"]),
            "latency_ms": float(latency["mean_ms"]),
            "failures": retrieval["runtime_failure_count"],
        })
    for baseline in ("B5", "B6", "B7"):
        rows.append({
            "series_id": "historical-development",
            "baseline_id": baseline,
            "label": {"B5": "Conflict aware", "B6": "Lifecycle aware", "B7": "Evidence gated"}[baseline],
            "status": "development_partial" if baseline in ("B6", "B7") else "not_scored",
            "answer_accuracy": None,
            "evidence_recall": None,
            "abstention_precision": float(abstention_metrics[f"{baseline}:abstention_precision"])
            if baseline in ("B6", "B7") else None,
            "recall_at_10": None,
            "latency_ms": None,
            "failures": None,
        })
    labels = {
        "B0": "No memory",
        "B1": "Full history",
        "B2": "Atomic memory",
        "B3": "Session summaries",
        "B4": "Hybrid retrieval",
        "B5": "Conflict aware",
        "B6": "Lifecycle aware",
        "B7": "Evidence gated",
    }
    for baseline in labels:
        rows.append({
            "series_id": "qwen3-8b-vllm-dev-v1",
            "baseline_id": baseline,
            "label": labels[baseline],
            "status": "development_scored",
            "answer_accuracy": _weighted_metric(qwen3_metrics, baseline, "strict_correctness"),
            "evidence_recall": _weighted_metric(qwen3_metrics, baseline, "source_message_recall"),
            "abstention_precision": _weighted_metric(qwen3_metrics, baseline, "abstention_accuracy"),
            "recall_at_10": _weighted_metric(qwen3_metrics, baseline, "recall_at_10"),
            "latency_ms": None,
            "failures": int(_metric_numerator(qwen3_metrics, baseline, "execution_failures") or 0),
        })
    return rows


def _metric_numerator(rows: list[dict[str, object]], baseline: str, metric: str) -> float | None:
    selected = [
        row for row in rows
        if row["baseline_id"] == baseline and row["metric"] == metric and int(row["denominator"]) > 0
    ]
    if not selected:
        return None
    return sum(float(row["numerator"]) for row in selected)


def _weighted_metric(rows: list[dict[str, object]], baseline: str, metric: str) -> float | None:
    selected = [
        row for row in rows
        if row["baseline_id"] == baseline and row["metric"] == metric and int(row["denominator"]) > 0
    ]
    denominator = sum(float(row["denominator"]) for row in selected)
    if denominator == 0:
        return None
    return sum(float(row["numerator"]) for row in selected) / denominator


def build_bundle(repo_root: Path) -> dict[str, object]:
    root = repo_root.resolve()
    config = _json(root / CONFIG)
    questions = _by(_jsonl(root / QUESTIONS), "case_id")
    answers = _by(_jsonl(root / ANSWERS), "case_id")
    predictions = _by(_jsonl(root / PREDICTIONS), "case_id")
    failures = _by(_jsonl(root / FAILURES), "case_id")
    _, messages = _source_index(root)
    cases = []
    for case_config in config["cases"]:
        case_id = str(case_config["case_id"])
        question, answer, prediction = questions[case_id], answers[case_id], predictions[case_id]
        states = case_config["memory_states"]
        evidence_ids = [str(item) for state in states for item in state["evidence"]]
        predicted_evidence = [
            f"{item['source_id']}:{item['message_id']}"
            for item in prediction.get("evidence", [])
        ]
        evidence_ids.extend(predicted_evidence)
        cases.append({
            "case_id": case_id,
            "title": case_config["title"],
            "summary": case_config["summary"],
            "capability": question["capability"],
            "difficulty": question["difficulty"],
            "as_of": question["as_of"],
            "question": question["question"],
            "source_events": _timeline(evidence_ids, messages),
            "memory_states": states,
            "answer": {
                "status": prediction["status"],
                "body": prediction["answer"],
                "confidence": prediction["confidence"],
                "evidence": predicted_evidence,
                "abstention_reason": prediction["abstention_reason"],
            },
            "review": _review(failures.get(case_id), answer),
        })
    openai = _json(root / OPENAI_RUN)
    qwen = _json(root / QWEN_CONFIG)
    qwen3 = _json(root / QWEN3_RUN)
    inputs = [CONFIG, QUESTIONS, ANSWERS, PREDICTIONS, FAILURES, *SOURCES, B1_SCORES,
              RETRIEVAL_SCORES, ABSTENTION_SCORES, OPENAI_RUN, QWEN_CONFIG, QWEN3_RUN,
              QWEN3_METRICS]
    input_digest = sha256()
    for path in inputs:
        raw = (root / path).read_bytes()
        input_digest.update(len(raw).to_bytes(8, "big"))
        input_digest.update(raw)
    return {
        "schema_version": "demo_bundle_v1",
        "bundle_version": config["bundle_version"],
        "build_input_sha256": input_digest.hexdigest(),
        "cases": cases,
        "runs": [
            {
                "series_id": "openai-gpt41-v1",
                "model": "GPT-4.1 mini extraction + GPT-4.1 answers",
                "status": "interrupted_not_scored",
                "requests": openai["historical_provider_request_count"],
                "input_tokens": openai["historical_input_token_count"],
                "output_tokens": openai["historical_output_token_count"],
                "cost": f"${openai['historical_openai_spend_usd']}",
                "note": "Extraction completed. B0 answer attempts produced no valid predictions or score.",
            },
            {
                "series_id": qwen["series_id"],
                "model": qwen["model"]["hugging_face_id"],
                "status": qwen["status"],
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost": "INR 0",
                "note": "Pinned and ready for a separately approved JarvisLabs run.",
            },
            {
                "series_id": qwen3["series_id"],
                "model": qwen3["model"]["hugging_face_id"],
                "status": qwen3["status"],
                "requests": qwen3["execution_metadata"]["provider_request_count"],
                "input_tokens": qwen3["execution_metadata"]["input_tokens"],
                "output_tokens": qwen3["execution_metadata"]["output_tokens"],
                "cost": f"INR {qwen3['execution_metadata']['gpu_cost_inr']:.4f}",
                "note": "Completed Qwen3-8B development B0-B7 via JarvisLabs L4 vLLM. Judge diagnostics are separate and uncalibrated.",
            },
        ],
        "scorecards": _scorecards(root),
    }


def write_bundle(repo_root: Path, output: Path = DEFAULT_OUTPUT) -> Path:
    destination = repo_root.resolve() / output
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(build_bundle(repo_root), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    destination.write_text(payload, encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", default=DEFAULT_OUTPUT.as_posix())
    args = parser.parse_args()
    path = write_bundle(Path(args.repo_root), Path(args.output))
    print(path)


if __name__ == "__main__":
    main()
