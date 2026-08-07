"""OpenAI Responses API adapter for the full-history smoke test."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Callable, Mapping, Sequence, TextIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .history import load_evaluation_questions, load_history_observations
from .smoke import (
    ModelRunConfig,
    SMOKE_CASE_IDS,
    run_smoke_test,
    write_smoke_test_artifacts,
)


_RESPONSES_URL = "https://api.openai.com/v1/responses"


class OpenAIResponseError(RuntimeError):
    """Raised when the provider response cannot be used without guessing."""


@dataclass(frozen=True)
class OpenAIResponseMetadata:
    """Non-secret provider metadata retained for one API response."""

    response_id: str
    returned_model: str
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


class OpenAIResponsesClient:
    """Minimal standard-library adapter for the OpenAI Responses API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        temperature: float,
        max_output_tokens: int,
        timeout_seconds: float = 120,
        transport: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key must be non-empty")
        if not model.strip():
            raise ValueError("model must be non-empty")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        self._api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self._transport = transport or self._post_response
        self.response_metadata: list[OpenAIResponseMetadata] = []

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        """Call OpenAI once and return exactly one raw output-text value."""

        payload: dict[str, object] = {
            "model": self.model,
            "instructions": system_prompt,
            "input": user_prompt,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "store": False,
            "text": {"format": {"type": "json_object"}},
        }
        response = self._transport(payload)
        raw_text, metadata = _parse_response(response, self.model)
        self.response_metadata.append(metadata)
        return raw_text

    def _post_response(
        self, payload: Mapping[str, object]
    ) -> Mapping[str, object]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            _RESPONSES_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:1000]
            raise OpenAIResponseError(
                f"OpenAI API returned HTTP {error.code}: {detail}"
            ) from error
        except URLError as error:
            raise OpenAIResponseError(f"OpenAI API request failed: {error.reason}") from error

        try:
            parsed = json.loads(response_body)
        except json.JSONDecodeError as error:
            raise OpenAIResponseError(
                f"OpenAI API returned invalid JSON: {error.msg}"
            ) from error
        if not isinstance(parsed, dict):
            raise OpenAIResponseError("OpenAI API response must be an object")
        return parsed


def _parse_response(
    response: Mapping[str, object], expected_model: str
) -> tuple[str, OpenAIResponseMetadata]:
    status = response.get("status")
    if status != "completed":
        raise OpenAIResponseError(f"OpenAI response status was {status!r}")

    returned_model = response.get("model")
    if not isinstance(returned_model, str) or not returned_model.strip():
        raise OpenAIResponseError("OpenAI response model is missing")
    if returned_model != expected_model:
        raise OpenAIResponseError(
            f"OpenAI returned model {returned_model!r}, expected {expected_model!r}"
        )

    output = response.get("output")
    if not isinstance(output, list):
        raise OpenAIResponseError("OpenAI response output must be a list")

    texts: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "refusal":
                raise OpenAIResponseError("OpenAI response contained a refusal")
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                texts.append(part["text"])

    if len(texts) != 1:
        raise OpenAIResponseError(
            f"OpenAI response must contain exactly one output_text value, got {len(texts)}"
        )

    response_id = response.get("id")
    if not isinstance(response_id, str) or not response_id.strip():
        raise OpenAIResponseError("OpenAI response id is missing")
    usage = response.get("usage")
    usage_record = usage if isinstance(usage, dict) else {}
    metadata = OpenAIResponseMetadata(
        response_id=response_id,
        returned_model=returned_model,
        input_tokens=_optional_integer(usage_record.get("input_tokens")),
        output_tokens=_optional_integer(usage_record.get("output_tokens")),
        total_tokens=_optional_integer(usage_record.get("total_tokens")),
    )
    return texts[0], metadata


def _optional_integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def load_env_value(path: str | Path, name: str) -> str:
    """Read one env value without printing or loading unrelated settings."""

    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise OpenAIResponseError(f"could not read {path}: {error}") from error

    matches: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() != name:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        matches.append(value)

    if not matches or not matches[0]:
        raise OpenAIResponseError(f"{name} is missing or empty in {path}")
    if len(matches) > 1:
        raise OpenAIResponseError(f"{name} appears more than once in {path}")
    return matches[0]


def _write_provider_metadata(
    path: Path, metadata: Sequence[OpenAIResponseMetadata]
) -> None:
    lines = [
        json.dumps(
            {
                "case_id": case_id,
                "response_id": item.response_id,
                "returned_model": item.returned_model,
                "input_tokens": item.input_tokens,
                "output_tokens": item.output_tokens,
                "total_tokens": item.total_tokens,
            },
            separators=(",", ":"),
        )
        for case_id, item in zip(SMOKE_CASE_IDS, metadata)
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the five-case full-history smoke test with OpenAI."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--max-output-tokens", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--source-dir", type=Path, default=Path("data/pilot/sources")
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/pilot/evaluation/eval_questions.jsonl"),
    )
    return parser


def main(argv: Sequence[str] | None = None, stdout: TextIO | None = None) -> int:
    """Run exactly five model calls and write the Step 3 artifacts."""

    stdout = stdout or sys.stdout
    args = _build_parser().parse_args(argv)
    api_key = load_env_value(args.env_file, "OPENAI_API_KEY")
    client = OpenAIResponsesClient(
        api_key=api_key,
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    )
    model_config = ModelRunConfig(
        provider="OpenAI",
        model=args.model,
        settings={
            "api": "responses",
            "temperature": args.temperature,
            "max_output_tokens": args.max_output_tokens,
            "store": False,
            "text_format": "json_object",
        },
    )
    observations = load_history_observations(args.source_dir)
    questions = load_evaluation_questions(args.questions)
    run = run_smoke_test(questions, observations, client, model_config)
    artifacts = write_smoke_test_artifacts(run, args.output_dir)
    metadata_path = args.output_dir / "api_metadata.jsonl"
    _write_provider_metadata(metadata_path, client.response_metadata)

    print(f"exit_code={run.exit_code}", file=stdout)
    print(f"predictions={artifacts.predictions_path}", file=stdout)
    print(f"diagnostics={artifacts.diagnostics_path}", file=stdout)
    print(f"report={artifacts.report_path}", file=stdout)
    print(f"api_metadata={metadata_path}", file=stdout)
    return run.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
