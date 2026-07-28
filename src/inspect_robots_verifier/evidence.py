"""Deterministic multi-view trajectory evidence selection."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from inspect_robots.scene import Target
from inspect_robots_verifier._json import jsonable
from inspect_robots_verifier.models import EvidenceBundle, EvidenceFrame, Phase

if TYPE_CHECKING:
    from inspect_robots.rollout import StepRecord, TrialRecord


@dataclass(frozen=True)
class SamplingConfig:
    """Evidence budget applied independently to each selected camera."""

    frames_per_camera: int = 5
    max_cameras: int = 4

    def __post_init__(self) -> None:
        if self.frames_per_camera < 2:
            raise ValueError("frames_per_camera must be at least 2")
        if self.max_cameras < 1:
            raise ValueError("max_cameras must be at least 1")


@dataclass(frozen=True)
class _Candidate:
    camera: str
    step: int
    source: str
    image: npt.NDArray[np.uint8]


class EvidenceSampler:
    """Select initial, process, and terminal evidence without model input."""

    def __init__(self, config: SamplingConfig | None = None):
        self.config = config or SamplingConfig()

    def sample(self, record: TrialRecord, target: Target | None) -> EvidenceBundle:
        """Construct a deterministic evidence bundle from an immutable trial."""
        by_camera: dict[str, list[_Candidate]] = defaultdict(list)
        for step in record.steps:
            for camera, image in _observation_images(step).items():
                by_camera[camera].append(
                    _Candidate(camera, step.t, "pre_action_observation", image)
                )
        if record.steps:
            last = record.steps[-1]
            for camera, image in sorted(last.result.observation.images.items()):
                by_camera[camera].append(
                    _Candidate(camera, last.t + 1, "post_action_observation", image)
                )

        selected_cameras = sorted(by_camera)[: self.config.max_cameras]
        frames: list[EvidenceFrame] = []
        for camera in selected_cameras:
            candidates = _deduplicate(by_camera[camera])
            indices = _quantile_indices(len(candidates), self.config.frames_per_camera)
            for ordinal, index in enumerate(indices):
                candidate = candidates[index]
                phase: Phase
                if ordinal == 0:
                    phase = "initial"
                elif ordinal == len(indices) - 1:
                    phase = "terminal"
                else:
                    phase = "process"
                frame_id = f"{camera}:t{candidate.step}:{phase}"
                frames.append(
                    EvidenceFrame(
                        frame_id=frame_id,
                        camera=camera,
                        step=candidate.step,
                        phase=phase,
                        source=candidate.source,
                        image=candidate.image,
                    )
                )

        instruction = _instruction(record)
        target_manifest: Mapping[str, Any] | None = None
        if target is not None:
            target_manifest = {
                "kind": target.kind,
                "spec": jsonable(target.spec),
            }
        return EvidenceBundle(
            scene_id=record.scene_id,
            epoch=record.epoch,
            seed=record.seed,
            instruction=instruction,
            target=target_manifest,
            frames=tuple(frames),
        )


def _observation_images(step: StepRecord) -> dict[str, npt.NDArray[np.uint8]]:
    images = {
        name: np.asarray(image, dtype=np.uint8) for name, image in step.observation.images.items()
    }
    if step.image_refs is not None:
        for name, reference in step.image_refs.items():
            images.setdefault(name, reference.load())
    return dict(sorted(images.items()))


def _deduplicate(candidates: list[_Candidate]) -> list[_Candidate]:
    unique: list[_Candidate] = []
    seen: set[tuple[int, str, bytes]] = set()
    for candidate in candidates:
        token = (candidate.step, candidate.source, candidate.image.tobytes())
        if token not in seen:
            seen.add(token)
            unique.append(candidate)
    return unique


def _quantile_indices(length: int, budget: int) -> list[int]:
    if length <= budget:
        return list(range(length))
    raw = [round(index * (length - 1) / (budget - 1)) for index in range(budget)]
    return list(dict.fromkeys(raw))


def _instruction(record: TrialRecord) -> str:
    for step in record.steps:
        if step.observation.instruction:
            return str(step.observation.instruction)
        if step.result.observation.instruction:
            return str(step.result.observation.instruction)
    return ""
