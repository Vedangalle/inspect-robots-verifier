"""Judge backends: an OpenAI-compatible wire client and deterministic replay."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from inspect_robots_verifier._json import digest_json, jsonable
from inspect_robots_verifier._png import png_data_url
from inspect_robots_verifier.models import Judgement, JudgeRequest, JudgeRun

Transport = Callable[[str, Mapping[str, str], bytes, float], tuple[int, bytes]]


@runtime_checkable
class JudgeBackend(Protocol):
    """Backend contract for one structured visual judgement."""

    @property
    def backend_name(self) -> str:
        """Stable wire/backend identifier."""
        ...

    @property
    def model(self) -> str:
        """Exact model identifier submitted to the backend."""
        ...

    def judge(self, request: JudgeRequest) -> JudgeRun:
        """Judge one request and return replayable response provenance."""
        ...


class OpenAICompatibleJudge:
    """Blocking vision judge over the OpenAI Chat Completions wire format."""

    backend_name = "openai-compatible-chat"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        backoff_s: float = 1.0,
        temperature: float = 0.0,
        json_mode: bool = True,
        transport: Transport | None = None,
    ):
        if not model:
            raise ValueError("model must not be empty")
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._model = model
        self._max_retries = max_retries
        self._backoff_s = backoff_s
        self._temperature = temperature
        self._json_mode = json_mode
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._timeout_s = timeout_s
        self._transport = transport or _urlopen_transport

    @property
    def model(self) -> str:
        """Exact model identifier submitted to the backend."""
        return self._model

    def judge(self, request: JudgeRequest) -> JudgeRun:
        """Submit one evidence sequence and parse its structured judgement."""
        content: list[dict[str, Any]] = [{"type": "text", "text": request.user_prompt}]
        for frame in request.evidence.frames:
            content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            f"frame_id={frame.frame_id} camera={frame.camera} "
                            f"step={frame.step} phase={frame.phase}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": png_data_url(frame.image)},
                    },
                ]
            )
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": self._temperature,
        }
        if self._json_mode:
            body["response_format"] = {"type": "json_object"}

        payload = self._post(body)
        decoded = _decode_content(payload)
        judgement = Judgement.from_mapping(decoded)
        return JudgeRun(
            judgement=judgement,
            backend=self.backend_name,
            model=self._model,
            request_sha256=request.sha256,
            response_sha256=digest_json(payload),
            response=jsonable(payload),
        )

    def _post(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        last_error = "unknown error"
        for attempt in range(self._max_retries):
            try:
                status, response_body = self._transport(
                    self._url,
                    self._headers,
                    json.dumps(body).encode(),
                    self._timeout_s,
                )
            except OSError as exc:
                last_error = str(exc)
            else:
                if status == 200:
                    try:
                        payload = json.loads(response_body)
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        raise RuntimeError(f"judge returned invalid HTTP JSON: {exc}") from None
                    if not isinstance(payload, Mapping):
                        raise RuntimeError("judge response must be a JSON object")
                    return payload
                text = response_body.decode(errors="replace")
                last_error = f"HTTP {status}: {text[:500]}"
                if status != 429 and status < 500:
                    raise RuntimeError(f"judge request rejected — {last_error}")
            if attempt + 1 < self._max_retries:
                time.sleep(self._backoff_s * 2**attempt)
        raise RuntimeError(
            f"judge request failed after {self._max_retries} attempts — {last_error}"
        )

    def close(self) -> None:
        """Retain context-manager symmetry; the stdlib transport owns no pool."""

    def __enter__(self) -> OpenAICompatibleJudge:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class ReplayJudge:
    """Deterministic backend for tests and exact offline pipeline replay."""

    backend_name = "replay"

    def __init__(
        self,
        judgements: Sequence[Judgement | Mapping[str, Any]],
        *,
        model: str = "recorded",
    ):
        self._judgements = [
            item if isinstance(item, Judgement) else Judgement.from_mapping(item)
            for item in judgements
        ]
        self._model = model
        self._cursor = 0

    @property
    def model(self) -> str:
        """Recorded model identifier."""
        return self._model

    def judge(self, request: JudgeRequest) -> JudgeRun:
        """Return the next recorded judgement."""
        if self._cursor >= len(self._judgements):
            raise RuntimeError("replay backend has no judgement left")
        judgement = self._judgements[self._cursor]
        self._cursor += 1
        response = {"replay_index": self._cursor - 1, "judgement": judgement.as_dict()}
        return JudgeRun(
            judgement=judgement,
            backend=self.backend_name,
            model=self._model,
            request_sha256=request.sha256,
            response_sha256=digest_json(response),
            response=response,
        )


def _decode_content(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError("judge response omitted choices[0].message.content") from None
    if isinstance(content, list):
        content = "".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, Mapping) and item.get("type") in ("text", "output_text")
        )
    if not isinstance(content, str):
        raise RuntimeError("judge response content must be text")
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines)
    try:
        decoded = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"judge returned invalid JSON: {exc}") from None
    if not isinstance(decoded, Mapping):
        raise RuntimeError("judge content must decode to a JSON object")
    return decoded


def _urlopen_transport(
    url: str,
    headers: Mapping[str, str],
    body: bytes,
    timeout_s: float,
) -> tuple[int, bytes]:
    request = Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urlopen(request, timeout=timeout_s) as response:
            return response.status, response.read()
    except HTTPError as exc:
        return exc.code, exc.read()
