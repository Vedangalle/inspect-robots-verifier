from __future__ import annotations

import json
from collections.abc import Mapping

import pytest

from conftest import trial
from inspect_robots_verifier.backend import OpenAICompatibleJudge, ReplayJudge
from inspect_robots_verifier.evidence import EvidenceSampler
from inspect_robots_verifier.models import Judgement
from inspect_robots_verifier.prompt import build_request


def payload(verdict: str = "success") -> dict[str, object]:
    return {
        "verdict": verdict,
        "progress": 1.0,
        "success_probability": 0.96,
        "confidence": 0.9,
        "failure_mode": None,
        "evidence": ["front:t0:initial", "front:t4:terminal"],
        "rationale": "The cube is visibly in the bowl.",
    }


def request() -> object:
    return build_request(EvidenceSampler().sample(trial(), None))


def test_openai_compatible_backend_sends_ordered_images_and_parses_json() -> None:
    seen: dict[str, object] = {}

    def handler(
        url: str, headers: Mapping[str, str], body: bytes, timeout: float
    ) -> tuple[int, bytes]:
        seen["url"] = url
        seen["authorization"] = headers["Authorization"]
        seen["body"] = json.loads(body)
        seen["timeout"] = timeout
        response = {
            "id": "run-1",
            "choices": [{"message": {"content": json.dumps(payload())}}],
        }
        return (
            200,
            json.dumps(response).encode(),
        )

    backend = OpenAICompatibleJudge(
        base_url="https://judge.test/v1",
        api_key="secret",
        model="vision-test",
        transport=handler,
    )
    judge_request = request()
    run = backend.judge(judge_request)  # type: ignore[arg-type]
    body = seen["body"]
    assert isinstance(body, dict)
    assert body["model"] == "vision-test"
    assert body["response_format"] == {"type": "json_object"}
    user_content = body["messages"][1]["content"]
    images = [part for part in user_content if part["type"] == "image_url"]
    assert len(images) == len(judge_request.evidence.frames)  # type: ignore[attr-defined]
    assert seen["authorization"] == "Bearer secret"
    assert run.judgement.verdict == "success"
    assert run.request_sha256 == judge_request.sha256  # type: ignore[attr-defined]
    assert len(run.response_sha256) == 64
    backend.close()


def test_backend_accepts_fenced_json_and_content_parts() -> None:
    response = "```json\n" + json.dumps(payload("partial")) + "\n```"

    def handler(_: str, __: Mapping[str, str], ___: bytes, ____: float) -> tuple[int, bytes]:
        return (
            200,
            json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": [
                                    {"type": "text", "text": response},
                                    {"type": "ignored", "text": "no"},
                                ]
                            }
                        }
                    ]
                }
            ).encode(),
        )

    with OpenAICompatibleJudge(
        base_url="https://judge.test/v1",
        api_key="",
        model="vision-test",
        json_mode=False,
        transport=handler,
    ) as backend:
        assert backend.judge(request()).judgement.verdict == "partial"  # type: ignore[arg-type]


def test_backend_retries_transient_response() -> None:
    calls = 0

    def handler(_: str, __: Mapping[str, str], ___: bytes, ____: float) -> tuple[int, bytes]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return 500, b"temporary"
        return (
            200,
            json.dumps({"choices": [{"message": {"content": json.dumps(payload())}}]}).encode(),
        )

    backend = OpenAICompatibleJudge(
        base_url="https://judge.test/v1",
        api_key="",
        model="vision-test",
        max_retries=2,
        backoff_s=0,
        transport=handler,
    )
    backend.judge(request())  # type: ignore[arg-type]
    assert calls == 2


@pytest.mark.parametrize(
    ("status", "body", "message"),
    [
        (400, "bad request", "rejected"),
        (200, {"choices": []}, "omitted"),
        (200, {"choices": [{"message": {"content": "not json"}}]}, "invalid JSON"),
        (200, {"choices": [{"message": {"content": "[]"}}]}, "JSON object"),
    ],
)
def test_backend_rejects_invalid_responses(status: int, body: object, message: str) -> None:
    def handler(_: str, __: Mapping[str, str], ___: bytes, ____: float) -> tuple[int, bytes]:
        if isinstance(body, str):
            return status, body.encode()
        return status, json.dumps(body).encode()

    backend = OpenAICompatibleJudge(
        base_url="https://judge.test/v1",
        api_key="",
        model="vision-test",
        transport=handler,
    )
    with pytest.raises(RuntimeError, match=message):
        backend.judge(request())  # type: ignore[arg-type]


def test_backend_validates_configuration_and_retry_exhaustion() -> None:
    with pytest.raises(ValueError, match="model"):
        OpenAICompatibleJudge(base_url="https://test", api_key="", model="")
    with pytest.raises(ValueError, match="max_retries"):
        OpenAICompatibleJudge(base_url="https://test", api_key="", model="x", max_retries=0)
    with pytest.raises(ValueError, match="timeout"):
        OpenAICompatibleJudge(base_url="https://test", api_key="", model="x", timeout_s=0)

    backend = OpenAICompatibleJudge(
        base_url="https://judge.test/v1",
        api_key="",
        model="x",
        max_retries=1,
        transport=lambda *_: (429, b"later"),
    )
    with pytest.raises(RuntimeError, match="after 1 attempts"):
        backend.judge(request())  # type: ignore[arg-type]


def test_replay_backend_preserves_provenance_and_exhausts() -> None:
    judgement = Judgement.from_mapping(payload())
    replay = ReplayJudge([judgement, payload("partial")], model="recorded-model")
    first = replay.judge(request())  # type: ignore[arg-type]
    second = replay.judge(request())  # type: ignore[arg-type]
    assert replay.backend_name == "replay"
    assert replay.model == "recorded-model"
    assert first.response["replay_index"] == 0
    assert second.judgement.verdict == "partial"
    with pytest.raises(RuntimeError, match="no judgement left"):
        replay.judge(request())  # type: ignore[arg-type]
