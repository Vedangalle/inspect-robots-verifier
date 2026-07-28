"""Run the complete verifier pipeline without a model API or robot hardware."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from inspect_robots.rollout import StepRecord, TrialRecord
from inspect_robots.scene import Target
from inspect_robots.types import Action, Observation, StepResult
from inspect_robots_verifier import Judgement, ProcessAwareVerifier, ReplayJudge


def _image(red_x: int) -> np.ndarray:
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[20:30, red_x : red_x + 10, 0] = 255
    image[18:34, 45:60, 1] = 120
    return image


def _trial() -> TrialRecord:
    record = TrialRecord(scene_id="demo-cube-to-bowl", epoch=0, seed=11)
    positions = (4, 16, 30, 47)
    for step, position in enumerate(positions[:-1]):
        result_position = positions[step + 1]
        record.steps.append(
            StepRecord(
                t=step,
                observation=Observation(
                    images={"front": _image(position)},
                    instruction="place the red cube in the green bowl",
                ),
                action=Action(data=np.array([1.0, 0.0], dtype=np.float64)),
                result=StepResult(
                    observation=Observation(
                        images={"front": _image(result_position)},
                        instruction="place the red cube in the green bowl",
                    ),
                    terminated=step == len(positions) - 2,
                    termination_reason="success" if step == len(positions) - 2 else None,
                ),
            )
        )
    record.terminated = True
    record.termination_reason = "success"
    return record


def main() -> None:
    frame_ids = ("front:t0:initial", "front:t3:terminal")
    runs = [
        Judgement(
            verdict="success",
            progress=1.0,
            success_probability=probability,
            confidence=0.92,
            failure_mode=None,
            evidence=frame_ids,
            rationale="The red cube moves from outside to visibly inside the green bowl.",
        )
        for probability in (0.91, 0.94, 0.93)
    ]
    scorer = ProcessAwareVerifier(
        ReplayJudge(runs, model="recorded-vision-judge"),
        artifact_root=Path("artifacts/demo"),
        samples=3,
    )
    score = scorer(
        _trial(),
        Target(
            "object_in_receptacle",
            {"object": "red cube", "receptacle": "green bowl"},
        ),
    )
    print(
        json.dumps(
            {
                "value": score.value,
                "explanation": score.explanation,
                "metadata": score.metadata,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
