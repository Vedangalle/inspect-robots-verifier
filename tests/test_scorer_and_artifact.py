from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import trial
from inspect_robots.rollout import TrialRecord
from inspect_robots_verifier.artifact import ArtifactWriter, _safe
from inspect_robots_verifier.backend import ReplayJudge
from inspect_robots_verifier.models import Judgement
from inspect_robots_verifier.scorer import ProcessAwareVerifier, process_vlm_scorer


def judgement(
    verdict: str = "success",
    *,
    probability: float = 0.95,
    confidence: float = 0.9,
    progress: float = 1.0,
    evidence: tuple[str, ...] = ("front:t0:initial", "front:t4:terminal"),
    failure_mode: str | None = None,
) -> Judgement:
    return Judgement(
        verdict=verdict,  # type: ignore[arg-type]
        progress=progress,
        success_probability=probability,
        confidence=confidence,
        failure_mode=failure_mode,
        evidence=evidence,
        rationale=f"observable {verdict}",
    )


def test_scorer_aggregates_and_writes_replay_grade_artifact(tmp_path: Path) -> None:
    record = trial()
    scorer = ProcessAwareVerifier(
        ReplayJudge(
            [
                judgement(probability=0.91),
                judgement(probability=0.95),
                judgement(probability=0.93),
            ],
            model="judge-v1",
        ),
        artifact_root=tmp_path,
        samples=3,
    )
    score = scorer(record, None)
    assert score.value == "success"
    assert score.metadata["aggregate"]["success_probability"] == pytest.approx(0.93)
    assert score.metadata["aggregate"]["disagreement"] == pytest.approx(0.04)
    artifact_path = Path(score.metadata["artifact"]["path"])
    assert artifact_path.exists()
    manifest = json.loads(artifact_path.read_text())
    assert manifest["schema_version"] == 1
    assert manifest["algorithm_version"] == "process-aware-v1"
    assert manifest["artifact_sha256"] == score.metadata["artifact"]["sha256"]
    assert manifest["evidence"]["sha256"] == score.metadata["artifact"]["evidence_sha256"]
    assert len(manifest["runs"]) == 3
    for item in manifest["evidence"]["frames"]:
        pixels = np.load(artifact_path.parent / item["file"], allow_pickle=False)
        assert pixels.dtype == np.uint8
    assert record.metadata["verifier_artifacts"] == [score.metadata["artifact"]]


@pytest.mark.parametrize(
    ("judgements", "expected_reason"),
    [
        (
            [
                judgement("success", probability=0.9),
                judgement("failure", probability=0.1),
            ],
            "spread",
        ),
        ([judgement(confidence=0.2)], "confidence"),
        ([judgement("unscorable", confidence=0.3)], "unscorable"),
        ([judgement(probability=0.6)], "below"),
        ([judgement(evidence=("hallucinated",))], "unknown evidence"),
    ],
)
def test_scorer_abstains_on_uncertainty(
    tmp_path: Path, judgements: list[Judgement], expected_reason: str
) -> None:
    score = ProcessAwareVerifier(
        ReplayJudge(judgements),
        artifact_root=tmp_path,
        samples=len(judgements),
    )(trial(), None)
    assert score.value == "unscorable"
    assert score.metadata["aggregate"]["abstained"] is True
    assert expected_reason in str(score.explanation)


def test_scorer_abstains_without_images_without_calling_backend(tmp_path: Path) -> None:
    score = ProcessAwareVerifier(
        ReplayJudge([]),
        artifact_root=tmp_path,
    )(TrialRecord(scene_id="no images", epoch=0, seed=None), None)
    assert score.value == "unscorable"
    assert score.metadata["samples"] == 0
    assert "No camera evidence" in str(score.explanation)
    assert Path(score.metadata["artifact"]["path"]).exists()


@pytest.mark.parametrize(
    ("output", "expected"),
    [("verdict", "partial"), ("progress", 0.55), ("success", False)],
)
def test_output_modes(tmp_path: Path, output: str, expected: object) -> None:
    score = ProcessAwareVerifier(
        ReplayJudge([judgement("partial", progress=0.55, probability=0.3)]),
        artifact_root=tmp_path / output,
        output=output,  # type: ignore[arg-type]
    )(trial(), None)
    assert score.value == expected


def test_success_output_is_true_only_for_accepted_success(tmp_path: Path) -> None:
    score = ProcessAwareVerifier(
        ReplayJudge([judgement()]),
        artifact_root=tmp_path,
        output="success",
    )(trial(), None)
    assert score.value is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"samples": 0}, "samples"),
        ({"min_confidence": -1}, "min_confidence"),
        ({"max_disagreement": 2}, "max_disagreement"),
        ({"success_threshold": -0.1}, "success_threshold"),
        ({"output": "bad"}, "output"),
    ],
)
def test_scorer_rejects_invalid_configuration(
    tmp_path: Path, kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ProcessAwareVerifier(
            ReplayJudge([judgement()]),
            artifact_root=tmp_path,
            **kwargs,  # type: ignore[arg-type]
        )


def test_factory_resolves_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("INSPECT_ROBOTS_VERIFIER_MODEL", "provider/model")
    monkeypatch.setenv("INSPECT_ROBOTS_VERIFIER_BASE_URL", "https://local.test/v1")
    monkeypatch.setenv("TEST_KEY", "key")
    scorer = process_vlm_scorer(
        api_key_env="TEST_KEY",
        artifact_root=str(tmp_path),
        frames_per_camera=2,
        max_cameras=1,
        output="progress",
    )
    assert scorer.backend.model == "provider/model"
    assert scorer.output == "progress"


def test_factory_requires_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INSPECT_ROBOTS_VERIFIER_MODEL", raising=False)
    with pytest.raises(ValueError, match="no verifier model"):
        process_vlm_scorer()


def test_artifact_writer_and_safe_name_guards(tmp_path: Path) -> None:
    assert _safe("a/b c") == "a-b-c"
    assert _safe("///") == "scene"
    with pytest.raises(ValueError, match="request"):
        ArtifactWriter(tmp_path).write(
            (), (), ProcessAwareVerifier(ReplayJudge([]))._aggregate((), evidence_ids=set())
        )
