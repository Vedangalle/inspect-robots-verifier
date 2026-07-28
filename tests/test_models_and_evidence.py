from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import frame, trial
from inspect_robots.frames import FrameRef
from inspect_robots.rollout import StepRecord, TrialRecord
from inspect_robots.scene import Target
from inspect_robots.types import Action, Observation, StepResult
from inspect_robots_verifier._json import canonical_json, digest_json, jsonable
from inspect_robots_verifier._png import encode_png, png_data_url
from inspect_robots_verifier.evidence import (
    EvidenceSampler,
    SamplingConfig,
    _quantile_indices,
)
from inspect_robots_verifier.models import EvidenceFrame, Judgement, image_digest


def test_json_normalization_and_digest_are_stable(tmp_path: Path) -> None:
    value = {
        "z": np.array([1, 2]),
        "a": np.float64(0.25),
        "path": tmp_path,
        "nonfinite": float("inf"),
        "tuple": (True, None),
    }
    normalized = jsonable(value)
    assert list(normalized) == ["a", "nonfinite", "path", "tuple", "z"]
    assert normalized["nonfinite"] == "inf"
    assert canonical_json(value) == canonical_json(normalized)
    assert digest_json(value) == digest_json(normalized)
    assert json.loads(canonical_json(value)) == normalized
    assert jsonable(object()).startswith("<object object at")


def test_evidence_frame_validates_pixels_and_digest() -> None:
    image = frame(3)
    evidence = EvidenceFrame("front:t0:initial", "front", 0, "initial", "obs", image)
    assert evidence.sha256 == image_digest(image)
    assert evidence.manifest()["shape"] == [3, 4, 3]
    with pytest.raises(TypeError, match="uint8"):
        EvidenceFrame("bad", "front", 0, "initial", "obs", image.astype(np.float32))
    with pytest.raises(ValueError, match="2-D or 3-D"):
        EvidenceFrame("bad", "front", 0, "initial", "obs", np.zeros(2, dtype=np.uint8))
    with pytest.raises(ValueError, match="channel count"):
        EvidenceFrame("bad", "front", 0, "initial", "obs", np.zeros((2, 2, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="empty"):
        EvidenceFrame("bad", "front", 0, "initial", "obs", np.zeros((0, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="digest"):
        EvidenceFrame("bad", "front", 0, "initial", "obs", image, sha256="bad")


def test_sampler_selects_initial_process_terminal_per_camera() -> None:
    record = trial(steps=6, cameras=("wrist", "front"))
    bundle = EvidenceSampler(SamplingConfig(frames_per_camera=3, max_cameras=2)).sample(
        record,
        Target("object_in_receptacle", {"object": "red cube", "receptacle": "bowl"}),
    )
    assert [frame.camera for frame in bundle.frames] == ["front"] * 3 + ["wrist"] * 3
    assert [frame.phase for frame in bundle.frames[:3]] == ["initial", "process", "terminal"]
    assert [frame.step for frame in bundle.frames[:3]] == [0, 3, 6]
    assert np.all(bundle.frames[0].image == 20)
    assert np.all(bundle.frames[2].image == 26)
    assert bundle.instruction == "put the red cube in the bowl"
    assert bundle.target == {
        "kind": "object_in_receptacle",
        "spec": {"object": "red cube", "receptacle": "bowl"},
    }
    assert len(bundle.sha256) == 64


def test_sampler_loads_frame_refs_and_limits_cameras(tmp_path: Path) -> None:
    stored = tmp_path / "frame.npy"
    np.save(stored, frame(9))
    record = TrialRecord(
        scene_id="ref",
        epoch=0,
        seed=None,
        steps=[
            StepRecord(
                t=0,
                observation=Observation(instruction=None),
                action=Action(data=np.zeros(1)),
                result=StepResult(
                    observation=Observation(
                        images={"z": frame(10)},
                        instruction="instruction from result",
                    )
                ),
                image_refs={
                    "b": FrameRef(camera="b", t=0, path=str(stored)),
                    "a": FrameRef(camera="a", t=0, path=str(stored)),
                },
            )
        ],
    )
    bundle = EvidenceSampler(SamplingConfig(frames_per_camera=2, max_cameras=2)).sample(
        record, None
    )
    assert [item.camera for item in bundle.frames] == ["a", "b"]
    assert bundle.instruction == "instruction from result"
    assert bundle.target is None
    assert all(np.all(item.image == 9) for item in bundle.frames)


def test_sampler_handles_no_steps_and_deduplicates_terminal_frame() -> None:
    empty = EvidenceSampler().sample(TrialRecord("empty", 0, None), None)
    assert empty.frames == ()
    assert empty.instruction == ""

    same = frame(4)
    record = TrialRecord(
        scene_id="same",
        epoch=0,
        seed=0,
        steps=[
            StepRecord(
                0,
                Observation(images={"front": same}),
                Action(np.zeros(1)),
                StepResult(Observation(images={"front": same})),
            )
        ],
    )
    bundle = EvidenceSampler(SamplingConfig(frames_per_camera=2)).sample(record, None)
    assert len(bundle.frames) == 2
    assert {item.source for item in bundle.frames} == {
        "pre_action_observation",
        "post_action_observation",
    }


@pytest.mark.parametrize(
    ("length", "budget", "expected"),
    [(2, 5, [0, 1]), (7, 3, [0, 3, 6]), (10, 4, [0, 3, 6, 9])],
)
def test_quantile_indices(length: int, budget: int, expected: list[int]) -> None:
    assert _quantile_indices(length, budget) == expected


def test_sampling_config_rejects_invalid_budgets() -> None:
    with pytest.raises(ValueError, match="frames_per_camera"):
        SamplingConfig(frames_per_camera=1)
    with pytest.raises(ValueError, match="max_cameras"):
        SamplingConfig(max_cameras=0)


def test_judgement_validation_and_mapping() -> None:
    judgement = Judgement.from_mapping(
        {
            "verdict": "partial",
            "progress": 0.6,
            "success_probability": 0.2,
            "confidence": 0.8,
            "failure_mode": "dropped",
            "evidence": ["front:t0:initial"],
            "rationale": "Object was lifted then dropped.",
        }
    )
    assert judgement.as_dict()["failure_mode"] == "dropped"
    with pytest.raises(ValueError, match="unsupported verdict"):
        Judgement("maybe", 0.0, 0.0, 0.0, None, (), "no")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="progress"):
        Judgement("failure", -0.1, 0.0, 0.5, None, (), "no")
    with pytest.raises(ValueError, match="rationale"):
        Judgement("failure", 0.0, 0.0, 0.5, None, (), " ")
    with pytest.raises(ValueError, match="unscorable"):
        Judgement("unscorable", 0.0, 0.0, 0.9, None, (), "occluded")
    with pytest.raises(TypeError, match="sequence"):
        Judgement.from_mapping(
            {
                "verdict": "failure",
                "progress": 0,
                "success_probability": 0,
                "confidence": 0.8,
                "evidence": "frame",
                "rationale": "failed",
            }
        )


def test_png_encoder_supports_grayscale_rgb_and_rgba() -> None:
    for image in (
        np.zeros((2, 3), dtype=np.uint8),
        np.zeros((2, 3, 3), dtype=np.uint8),
        np.zeros((2, 3, 4), dtype=np.uint8),
    ):
        assert encode_png(image).startswith(b"\x89PNG\r\n\x1a\n")
        assert png_data_url(image).startswith("data:image/png;base64,")
    with pytest.raises(ValueError, match="channel count"):
        encode_png(np.zeros((2, 3, 2), dtype=np.uint8))
