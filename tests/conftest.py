from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from inspect_robots.rollout import StepRecord, TrialRecord
from inspect_robots.types import Action, Observation, StepResult


def frame(value: int, *, shape: tuple[int, int, int] = (3, 4, 3)) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint8)


def trial(
    *,
    steps: int = 4,
    cameras: tuple[str, ...] = ("front",),
    instruction: str = "put the red cube in the bowl",
) -> TrialRecord:
    record = TrialRecord(scene_id="scene/one", epoch=2, seed=7)
    for t in range(steps):
        images: Mapping[str, np.ndarray[Any, np.dtype[np.uint8]]] = {
            camera: frame(t + index * 20) for index, camera in enumerate(cameras)
        }
        result_images: Mapping[str, np.ndarray[Any, np.dtype[np.uint8]]] = {
            camera: frame(t + 1 + index * 20) for index, camera in enumerate(cameras)
        }
        record.steps.append(
            StepRecord(
                t=t,
                observation=Observation(images=images, instruction=instruction),
                action=Action(data=np.zeros(2, dtype=np.float64)),
                result=StepResult(
                    observation=Observation(images=result_images, instruction=instruction),
                    terminated=t == steps - 1,
                    termination_reason="success" if t == steps - 1 else None,
                ),
            )
        )
    record.terminated = bool(steps)
    record.termination_reason = "success" if steps else None
    return record
