"""OpenAI Responses API adapter for the full-history smoke test."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import re
import sys
import time
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

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        provider_code: str | None = None,
        provider_param: str | None = None,
        provider_reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.provider_code = provider_code
        self.provider_param = provider_param
        self.provider_reason = provider_reason


class OpenAIModelMismatchError(OpenAIResponseError):
    """Raised when the provider resolves to a different model snapshot."""


@dataclass(frozen=True)
class OpenAIRateLimitMetadata:
    """Rate-limit capacity reported by one successful API response."""

    remaining_requests: int | None = None
    remaining_tokens: int | None = None
    reset_requests_seconds: float | None = None
    reset_tokens_seconds: float | None = None
    remaining_project_tokens: int | None = None
    reset_project_tokens_seconds: float | None = None


@dataclass(frozen=True)
class OpenAIResponseMetadata:
    """Non-secret provider metadata retained for one API response."""

    response_id: str
    returned_model: str
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    request_id: str | None = None
    rate_limits: OpenAIRateLimitMetadata | None = None
    pacing_delay_seconds: float = 0.0


@dataclass(frozen=True)
class _OpenAITransportResponse:
    body: Mapping[str, object]
    request_id: str | None
    rate_limits: OpenAIRateLimitMetadata


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
        transport: Callable[[Mapping[str, object]], object] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        pacing_safety_factor: float = 1.1,
        minimum_request_interval_seconds: float = 15.0,
        text_format: Mapping[str, object] | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key must be non-empty")
        if not model.strip():
            raise ValueError("model must be non-empty")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if pacing_safety_factor < 1:
            raise ValueError("pacing_safety_factor must be at least 1")
        if minimum_request_interval_seconds < 0:
            raise ValueError("minimum_request_interval_seconds cannot be negative")
        if text_format is not None and not isinstance(text_format, Mapping):
            raise ValueError("text_format must be an object")
        self._api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self._transport = transport or self._post_response
        self._sleep = sleep
        self._pacing_safety_factor = pacing_safety_factor
        self._minimum_request_interval_seconds = minimum_request_interval_seconds
        self._text_format = deepcopy(
            dict(text_format) if text_format is not None else {"type": "json_object"}
        )
        self._requests_started = 0
        self.response_metadata: list[OpenAIResponseMetadata] = []
        self._pacing_metadata: OpenAIResponseMetadata | None = None

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        """Call OpenAI once and return exactly one raw output-text value."""

        raw_text, _ = self.complete_with_metadata(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return raw_text

    def complete_with_metadata(
        self, *, system_prompt: str, user_prompt: str
    ) -> tuple[str, OpenAIResponseMetadata]:
        """Call OpenAI once and return raw output plus non-secret metadata."""

        payload: dict[str, object] = {
            "model": self.model,
            "instructions": system_prompt,
            "input": user_prompt,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "store": False,
            "text": {"format": deepcopy(self._text_format)},
        }
        pacing_delay = self._pace_before_request()
        self._requests_started += 1
        transport_response = self._transport(payload)
        request_id: str | None = None
        rate_limits: OpenAIRateLimitMetadata | None = None
        if isinstance(transport_response, _OpenAITransportResponse):
            response = transport_response.body
            request_id = transport_response.request_id
            rate_limits = transport_response.rate_limits
        elif isinstance(transport_response, Mapping):
            response = transport_response
        else:
            raise OpenAIResponseError("OpenAI transport response must be an object")
        raw_text, metadata = _parse_response(response, self.model)
        metadata = replace(
            metadata,
            request_id=request_id,
            rate_limits=rate_limits,
            pacing_delay_seconds=pacing_delay,
        )
        self.response_metadata.append(metadata)
        self._pacing_metadata = metadata
        return raw_text, metadata

    def restore_pacing_metadata(self, metadata: OpenAIResponseMetadata) -> None:
        """Restore the last saved response headers before a resumed request."""

        if metadata.returned_model != self.model:
            raise OpenAIModelMismatchError(
                "saved pacing metadata does not match the requested model"
            )
        self._pacing_metadata = metadata
        self._requests_started = max(self._requests_started, 1)

    def _pace_before_request(self) -> float:
        delays: list[float] = []
        if self._requests_started:
            delays.append(self._minimum_request_interval_seconds)
        if self._pacing_metadata is None:
            delay = max(delays, default=0.0)
            if delay > 0:
                self._sleep(delay)
            return delay
        previous = self._pacing_metadata
        limits = previous.rate_limits
        if limits is None:
            delay = max(delays, default=0.0)
            if delay > 0:
                self._sleep(delay)
            return delay

        previous_input = previous.input_tokens or 0
        required_tokens = (
            math.ceil(previous_input * self._pacing_safety_factor)
            + self.max_output_tokens
        )
        if limits.remaining_requests is not None and limits.remaining_requests < 1:
            delays.append(
                limits.reset_requests_seconds
                if limits.reset_requests_seconds is not None
                else 60.0
            )
        if limits.remaining_tokens is not None and limits.remaining_tokens < required_tokens:
            delays.append(
                limits.reset_tokens_seconds
                if limits.reset_tokens_seconds is not None
                else 60.0
            )
        if (
            limits.remaining_project_tokens is not None
            and limits.remaining_project_tokens < required_tokens
        ):
            delays.append(
                limits.reset_project_tokens_seconds
                if limits.reset_project_tokens_seconds is not None
                else 60.0
            )
        delay = max(delays, default=0.0)
        if delay > 0:
            delay += 0.25
            self._sleep(delay)
        return delay

    def _post_response(
        self, payload: Mapping[str, object]
    ) -> object:
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
                request_id = response.headers.get("x-request-id")
                rate_limits = _rate_limit_metadata(response.headers)
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:1000]
            provider_code, provider_param, provider_reason = _provider_error_metadata(detail)
            suffix = ", ".join(
                value for value in (
                    provider_code,
                    f"param={provider_param}" if provider_param else None,
                    f"reason={provider_reason}" if provider_reason else None,
                )
                if value is not None
            )
            raise OpenAIResponseError(
                f"OpenAI API returned HTTP {error.code}"
                + (f" ({suffix})" if suffix else ""),
                status_code=error.code,
                provider_code=provider_code,
                provider_param=provider_param,
                provider_reason=provider_reason,
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
        return _OpenAITransportResponse(
            body=parsed,
            request_id=request_id,
            rate_limits=rate_limits,
        )


def _provider_error_metadata(
    detail: str,
) -> tuple[str | None, str | None, str | None]:
    """Return only allowlisted diagnostic fields from a provider error body."""

    try:
        parsed = json.loads(detail)
    except json.JSONDecodeError:
        return None, None, None
    if not isinstance(parsed, Mapping) or not isinstance(parsed.get("error"), Mapping):
        return None, None, None
    error = parsed["error"]
    code = _safe_error_token(error.get("code")) or _safe_error_token(error.get("type"))
    return (
        code,
        _safe_error_token(error.get("param")),
        _provider_message_category(error.get("message")),
    )


def _safe_error_token(value: object) -> str | None:
    if not isinstance(value, str) or not value or len(value) > 80:
        return None
    return value if re.fullmatch(r"[A-Za-z0-9_.-]+", value) else None


def _provider_message_category(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    message = value.lower()
    if "json" in message and any(token in message for token in ("mention", "must", "contain")):
        return "json_instruction_missing"
    if any(token in message for token in ("safety", "flagged", "usage polic")):
        return "input_safety_rejected"
    if "unsupported" in message:
        return "unsupported_value"
    if "invalid" in message and "input" in message:
        return "invalid_input"
    return None


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
        raise OpenAIModelMismatchError(
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


def _rate_limit_metadata(headers: Mapping[str, str]) -> OpenAIRateLimitMetadata:
    return OpenAIRateLimitMetadata(
        remaining_requests=_header_integer(headers, "x-ratelimit-remaining-requests"),
        remaining_tokens=_header_integer(headers, "x-ratelimit-remaining-tokens"),
        reset_requests_seconds=_parse_reset_duration(
            headers.get("x-ratelimit-reset-requests")
        ),
        reset_tokens_seconds=_parse_reset_duration(
            headers.get("x-ratelimit-reset-tokens")
        ),
        remaining_project_tokens=_header_integer(
            headers, "x-ratelimit-remaining-project-tokens"
        ),
        reset_project_tokens_seconds=_parse_reset_duration(
            headers.get("x-ratelimit-reset-project-tokens")
        ),
    )


def _header_integer(headers: Mapping[str, str], name: str) -> int | None:
    value = headers.get(name)
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")


def _parse_reset_duration(value: str | None) -> float | None:
    if value is None:
        return None
    position = 0
    seconds = 0.0
    multipliers = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    for match in _DURATION_PART.finditer(value.strip()):
        if match.start() != position:
            return None
        seconds += float(match.group(1)) * multipliers[match.group(2)]
        position = match.end()
    return seconds if position == len(value.strip()) and position > 0 else None


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
