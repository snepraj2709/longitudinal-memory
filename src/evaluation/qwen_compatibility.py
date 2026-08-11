"""Build the 12-request, runtime-only Qwen Stage 1 compatibility pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from extraction.predicate_registry import load_predicate_registry
from extraction.prompt import build_atomic_extraction_prompt, get_atomic_extraction_system_prompt
from extraction.scaled_source import _adapt_source
from extraction.schema import atomic_extraction_text_format


RUNTIME = Path("data/scaled-v1/runtime")
PROMPTS = Path("configs/evaluation/frozen_prompts_v1.json")
REGISTRY = Path("configs/extraction/predicate_registry_v2.json")
DEFAULT_OUTPUT = Path("results/evaluation/qwen35-27b-fp8-v1/stage1/compatibility-requests.jsonl")
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "seed": 42}
BODY_FIELD = {"qa": "answer", "summary": "summary", "interactive": "response"}


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _answer_schema(task: str) -> dict[str, object]:
    body = BODY_FIELD[task]
    properties: dict[str, object] = {
        "status": {"type": "string", "enum": ["answered", "abstained", "disputed", "partially_answered"]},
        body: {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "statements": {"type": "array", "items": {"type": "string"}},
        "citations": {"type": "array", "items": {"type": "object", "properties": {
            "source_id": {"type": "string"}, "message_id": {"type": "string"}, "quote": {"type": "string"},
        }, "required": ["source_id", "message_id", "quote"], "additionalProperties": False}},
        "unresolved_parts": {"type": "array", "items": {"type": "string"}},
        "abstention_reason": {"type": ["string", "null"]},
    }
    fields = list(properties)
    return {"type": "object", "properties": properties, "required": fields, "additionalProperties": False}


def _response_format(name: str, schema: Mapping[str, object]) -> dict[str, object]:
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}


def _answer_user_prompt(task: str, case: Mapping[str, object]) -> str:
    fields = {
        "qa": ("case_id", "user_id", "as_of", "question"),
        "summary": ("case_id", "user_id", "as_of", "instruction"),
        "interactive": ("case_id", "user_id", "as_of", "scenario", "initial_user_message"),
    }[task]
    body = BODY_FIELD[task]
    return json.dumps({
        "runtime_case": {name: case[name] for name in fields},
        "context_records": [],
        "response_format": "JSON object",
        "output_contract": {
            "exact_fields": ["status", body, "confidence", "statements", "citations", "unresolved_parts", "abstention_reason"],
            "rules": "B0 has no memory context. Abstain without inventing evidence.",
        },
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_requests(repo_root: Path) -> list[dict[str, object]]:
    root = repo_root.resolve()
    users = _jsonl(root / RUNTIME / "users.jsonl")
    names = {str(row["user_id"]): str(row["display_name"]) for row in users}
    sources = _jsonl(root / RUNTIME / "sources.jsonl")
    registry = load_predicate_registry(root / REGISTRY)
    extraction_system = get_atomic_extraction_system_prompt("atomic-extraction-v3", registry=registry)
    raw_format = atomic_extraction_text_format(registry)
    extraction_format = _response_format(str(raw_format["name"]), raw_format["schema"])
    requests = []
    for index, source in enumerate(sources[:3], 1):
        adapted = _adapt_source(source, (str(source["user_id"]), str(source["source_id"])), names)
        requests.append({
            "request_id": f"compat-extraction-{index:02d}", "task": "extraction",
            "record_id": source["source_id"], "messages": [
                {"role": "system", "content": extraction_system},
                {"role": "user", "content": build_atomic_extraction_prompt(adapted, include_speaker_name=False)},
            ],
            "response_format": extraction_format, "required_output_fields": ["claims"],
            "max_output_tokens": 1200, "sampling": SAMPLING,
        })
    prompt_rows = json.loads((root / PROMPTS).read_text(encoding="utf-8"))["tasks"]
    prompts = {str(item["task"]): str(item["system_instruction"]) for item in prompt_rows}
    paths = {"qa": "qa.jsonl", "summary": "summaries.jsonl", "interactive": "interactive.jsonl"}
    for task, filename in paths.items():
        cases = _jsonl(root / RUNTIME / filename)
        for index, case in enumerate(cases[:3], 1):
            fields = ["status", BODY_FIELD[task], "confidence", "statements", "citations", "unresolved_parts", "abstention_reason"]
            requests.append({
                "request_id": f"compat-{task}-{index:02d}", "task": task,
                "record_id": case["case_id"], "baseline_id": "B0", "messages": [
                    {"role": "system", "content": prompts[task]},
                    {"role": "user", "content": _answer_user_prompt(task, case)},
                ],
                "response_format": _response_format(f"qwen_{task}_v1", _answer_schema(task)),
                "required_output_fields": fields, "max_output_tokens": 1000, "sampling": SAMPLING,
            })
    if len(requests) != 12:
        raise ValueError("compatibility request selection changed")
    return requests


def write_requests(repo_root: Path, output: Path = DEFAULT_OUTPUT) -> Path:
    destination = repo_root.resolve() / output
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in build_requests(repo_root))
    if destination.exists():
        if destination.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"immutable compatibility pack changed: {destination}")
        return destination
    with destination.open("x", encoding="utf-8", newline="") as handle:
        handle.write(payload)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", default=DEFAULT_OUTPUT.as_posix())
    args = parser.parse_args()
    print(write_requests(Path(args.repo_root), Path(args.output)))


if __name__ == "__main__":
    main()
