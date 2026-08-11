"""Private vLLM adapter for the OpenAI-compatible chat completions protocol.

The Qwen comparison series runs against a private vLLM server instead of
OpenAI. This seam is the provider-neutral boundary: it speaks the standard
``POST {base}/v1/chat/completions`` protocol and returns the same shared
provider metadata shape (``OpenAIResponseMetadata``) that every frozen runner
already uses to build ``ProviderRecord`` rows. No existing runner or frozen
artifact is changed by this module.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import json
import time
from typing import Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .openai_client import (
    OpenAIResponseMetadata,
    _optional_integer,
    _provider_error_metadata,
)


class VLLMResponseError(RuntimeError):
    """Raised when a private vLLM response cannot be used without guessing."""

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


class VLLMModelMismatchError(VLLMResponseError):
    """Raised when the vLLM server resolves to a different model snapshot."""


class VLLMTransportError(VLLMResponseError):
    """Raised only when a request failed before any response body was received."""


@dataclass(frozen=True)
class _VLLMTransportResponse:
    body: Mapping[str, object]
    request_id: str | None


class VLLMClient:
    """Minimal standard-library adapter for a private vLLM server."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        temperature: float,
        max_output_tokens: int,
        api_key: str | None = None,
        timeout_seconds: float = 120,
        transport: Callable[[Mapping[str, object]], object] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        minimum_request_interval_seconds: float = 0.0,
        text_format: Mapping[str, object] | None = None,
        payload_options: Mapping[str, object] | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url must be non-empty")
        if not model.strip():
            raise ValueError("model must be non-empty")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if minimum_request_interval_seconds < 0:
            raise ValueError("minimum_request_interval_seconds cannot be negative")
        if text_format is not None and not isinstance(text_format, Mapping):
            raise ValueError("text_format must be an object")
        if payload_options is not None and not isinstance(payload_options, Mapping):
            raise ValueError("payload_options must be an object")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._transport = transport or self._post_completion
        self._sleep = sleep
        self._minimum_request_interval_seconds = minimum_request_interval_seconds
        self._text_format = deepcopy(
            dict(text_format) if text_format is not None else {"type": "json_object"}
        )
        self._payload_options = deepcopy(dict(payload_options or {}))
        protected = {"model", "messages", "max_tokens", "stream", "response_format"}
        overlap = protected.intersection(self._payload_options)
        if overlap:
            raise ValueError(
                "payload_options cannot replace protected fields: "
                + ", ".join(sorted(overlap))
            )
        self._requests_started = 0
        self.response_metadata: list[OpenAIResponseMetadata] = []

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        """Call the private vLLM server once and return exactly one raw output text."""

        raw_text, _ = self.complete_with_metadata(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return raw_text

    def complete_with_metadata(
        self, *, system_prompt: str, user_prompt: str
    ) -> tuple[str, OpenAIResponseMetadata]:
        """Call the private vLLM server once and return raw output plus metadata."""

        payload: dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "stream": False,
            "response_format": deepcopy(self._text_format),
        }
        payload.update(deepcopy(self._payload_options))
        pacing_delay = self._pace_before_request()
        self._requests_started += 1
        transport_response = self._transport(payload)
        request_id: str | None = None
        if isinstance(transport_response, _VLLMTransportResponse):
            response = transport_response.body
            request_id = transport_response.request_id
        elif isinstance(transport_response, Mapping):
            response = transport_response
        else:
            raise VLLMResponseError("vLLM transport response must be an object")
        raw_text, metadata = _parse_completion(response, self.model)
        metadata = replace(metadata, pacing_delay_seconds=pacing_delay, request_id=request_id)
        self.response_metadata.append(metadata)
        return raw_text, metadata

    def _pace_before_request(self) -> float:
        delay = self._minimum_request_interval_seconds if self._requests_started else 0.0
        if delay > 0:
            self._sleep(delay)
        return delay

    def _post_completion(self, payload: Mapping[str, object]) -> object:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = Request(
            f"{self.base_url}/v1/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
                request_id = response.headers.get("x-request-id")
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
            raise VLLMResponseError(
                f"vLLM API returned HTTP {error.code}"
                + (f" ({suffix})" if suffix else ""),
                status_code=error.code,
                provider_code=provider_code,
                provider_param=provider_param,
                provider_reason=provider_reason,
            ) from error
        except URLError as error:
            raise VLLMTransportError(f"vLLM API request failed: {error.reason}") from error
        except TimeoutError as error:
            raise VLLMTransportError("vLLM API request timed out before a response") from error

        try:
            parsed = json.loads(response_body)
        except json.JSONDecodeError as error:
            raise VLLMResponseError(
                f"vLLM API returned invalid JSON: {error.msg}"
            ) from error
        if not isinstance(parsed, dict):
            raise VLLMResponseError("vLLM API response must be an object")
        return _VLLMTransportResponse(body=parsed, request_id=request_id)


def _parse_completion(
    response: Mapping[str, object], expected_model: str
) -> tuple[str, OpenAIResponseMetadata]:
    returned_model = response.get("model")
    if not isinstance(returned_model, str) or not returned_model.strip():
        raise VLLMResponseError("vLLM response model is missing")
    if returned_model != expected_model:
        raise VLLMModelMismatchError(
            f"vLLM returned model {returned_model!r}, expected {expected_model!r}"
        )

    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise VLLMResponseError(
            f"vLLM response must contain exactly one choice, got {len(choices) if isinstance(choices, list) else 'none'}"
        )
    message = choices[0]
    if not isinstance(message, dict) or not isinstance(message.get("message"), dict):
        raise VLLMResponseError("vLLM response choice message is missing")
    content = message["message"].get("content")
    if not isinstance(content, str):
        raise VLLMResponseError("vLLM response choice must contain exactly one content string")

    response_id = response.get("id")
    if not isinstance(response_id, str) or not response_id.strip():
        raise VLLMResponseError("vLLM response id is missing")
    usage = response.get("usage")
    usage_record = usage if isinstance(usage, dict) else {}
    input_tokens = _optional_integer(usage_record.get("prompt_tokens"))
    output_tokens = _optional_integer(usage_record.get("completion_tokens"))
    total_tokens = _optional_integer(usage_record.get("total_tokens"))
    if any(value is None for value in (input_tokens, output_tokens, total_tokens)):
        raise VLLMResponseError("vLLM response token usage is missing")
    return content, OpenAIResponseMetadata(
        response_id=response_id,
        returned_model=returned_model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )
