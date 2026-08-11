"""Measure Qwen tokenization, inference throughput, and GPU cost on private vLLM."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .frozen_answer_contracts import validate_answer_output
from .qwen_series import BudgetGate, append_checkpoint, write_immutable_json
from .vllm_client import VLLMClient, VLLMResponseError


@dataclass(frozen=True)
class TokenCount:
    count: int
    elapsed_seconds: float


class VLLMTokenizer:
    def __init__(self, *, base_url: str, model: str, api_key: str | None, timeout_seconds: float = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def count(self, messages: list[Mapping[str, object]]) -> TokenCount:
        payload = {
            "model": self.model,
            "messages": messages,
            "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            f"{self.base_url}/tokenize",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        began = time.monotonic()
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                value = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            raise VLLMResponseError(f"vLLM tokenizer returned HTTP {error.code}", status_code=error.code) from error
        except (URLError, json.JSONDecodeError) as error:
            raise VLLMResponseError(f"vLLM tokenizer failed: {error}") from error
        elapsed = time.monotonic() - began
        count = value.get("count") if isinstance(value, dict) else None
        if not isinstance(count, int) or count < 0:
            raise VLLMResponseError("vLLM tokenizer response count is missing")
        return TokenCount(count=count, elapsed_seconds=elapsed)


def run_compatibility_preflight(
    *,
    requests_path: Path,
    output_dir: Path,
    base_url: str,
    model: str,
    api_key: str | None,
    hourly_rate_inr: float,
    stage_cap_inr: float,
    setup_seconds: float,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"immutable output directory already exists: {output_dir}")
    rows = [json.loads(line) for line in requests_path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 12:
        raise ValueError("Stage 1 compatibility pack must contain exactly 12 requests")
    output_dir.mkdir(parents=True)
    tokenizer = VLLMTokenizer(base_url=base_url, model=model, api_key=api_key)
    checkpoint = output_dir / "responses.jsonl"
    began = time.monotonic()
    input_tokens = output_tokens = valid_responses = 0
    for position, row in enumerate(rows, 1):
        elapsed_cost = hourly_rate_inr * (setup_seconds + time.monotonic() - began) / 3600
        request_reserve = hourly_rate_inr * 120 / 3600
        BudgetGate(1, stage_cap_inr, stage_cap_inr, elapsed_cost, elapsed_cost).assert_can_spend(request_reserve)
        messages = row["messages"]
        count = tokenizer.count(messages)
        client = VLLMClient(
            base_url=base_url,
            model=model,
            api_key=api_key,
            temperature=float(row["sampling"]["temperature"]),
            max_output_tokens=int(row["max_output_tokens"]),
            text_format=row["response_format"],
            payload_options={
                "top_p": row["sampling"]["top_p"],
                "top_k": row["sampling"]["top_k"],
                "presence_penalty": row["sampling"]["presence_penalty"],
                "seed": row["sampling"]["seed"],
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        inference_started = time.monotonic()
        raw, metadata = client.complete_with_metadata(
            system_prompt=str(messages[0]["content"]),
            user_prompt=str(messages[1]["content"]),
        )
        inference_seconds = time.monotonic() - inference_started
        try:
            parsed = json.loads(raw)
            valid = isinstance(parsed, dict) and set(row["required_output_fields"]) == set(parsed)
            if valid and row["task"] == "extraction":
                valid = isinstance(parsed["claims"], list)
            elif valid:
                validate_answer_output(parsed, task=str(row["task"]), evidence_index={})
        except json.JSONDecodeError:
            valid = False
        except Exception:
            valid = False
        input_tokens += metadata.input_tokens or 0
        output_tokens += metadata.output_tokens or 0
        valid_responses += int(valid)
        append_checkpoint(checkpoint, {
            "position": position,
            "request_id": row["request_id"],
            "task": row["task"],
            "tokenized_input_count": count.count,
            "provider_input_tokens": metadata.input_tokens,
            "provider_output_tokens": metadata.output_tokens,
            "tokenizer_seconds": round(count.elapsed_seconds, 6),
            "inference_seconds": round(inference_seconds, 6),
            "valid": valid,
            "response": parsed if valid else raw,
        })
    inference_wall_seconds = time.monotonic() - began
    gpu_seconds = setup_seconds + inference_wall_seconds
    manifest = {
        "schema_version": "qwen_compatibility_preflight_v1",
        "status": "completed" if valid_responses == len(rows) else "completed_with_invalid_responses",
        "request_count": len(rows),
        "valid_response_count": valid_responses,
        "valid_response_rate": valid_responses / len(rows),
        "setup_seconds": setup_seconds,
        "inference_wall_seconds": round(inference_wall_seconds, 6),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "output_tokens_per_second": round(output_tokens / inference_wall_seconds, 4) if inference_wall_seconds else None,
        "hourly_rate_inr": hourly_rate_inr,
        "gpu_cost_inr": round(gpu_seconds * hourly_rate_inr / 3600, 4),
        "automatic_retry_count": 0,
    }
    write_immutable_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="qwen35-27b-fp8-v1")
    parser.add_argument("--api-key")
    parser.add_argument("--hourly-rate-inr", required=True, type=float)
    parser.add_argument("--stage-cap-inr", default=200.0, type=float)
    parser.add_argument("--setup-seconds", required=True, type=float)
    args = parser.parse_args()
    result = run_compatibility_preflight(
        requests_path=Path(args.requests), output_dir=Path(args.output),
        base_url=args.base_url, model=args.model, api_key=args.api_key,
        hourly_rate_inr=args.hourly_rate_inr, stage_cap_inr=args.stage_cap_inr,
        setup_seconds=args.setup_seconds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
