"""Typed evidence, request, and judgement contracts."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from inspect_robots_verifier._json import digest_json, jsonable

Phase = Literal["initial", "process", "terminal"]
Verdict = Literal["success", "partial", "failure", "unscorable"]

_VERDICTS = frozenset({"success", "partial", "failure", "unscorable"})


def image_digest(image: npt.NDArray[np.uint8]) -> str:
    """Hash an image with its dtype and shape, not only its byte payload."""
    contiguous = np.ascontiguousarray(image)
    header = f"{contiguous.dtype.str}:{','.join(map(str, contiguous.shape))}".encode()
    return hashlib.sha256(header + b"\0" + contiguous.tobytes()).hexdigest()


@dataclass(frozen=True, eq=False)
class EvidenceFrame:
    """One selected trajectory frame supplied to a visual judge."""

    frame_id: str
    camera: str
    step: int
    phase: Phase
    source: str
    image: npt.NDArray[np.uint8] = field(repr=False)
    sha256: str = ""

    def __post_init__(self) -> None:
        image = np.asarray(self.image)
        if image.dtype != np.uint8:
            raise TypeError(f"evidence image must be uint8, got {image.dtype}")
        if image.ndim not in (2, 3):
            raise ValueError(f"evidence image must be 2-D or 3-D, got shape {image.shape}")
        if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
            raise ValueError(f"evidence image has unsupported channel count: {image.shape[2]}")
        if image.size == 0:
            raise ValueError("evidence image must not be empty")
        expected = image_digest(image)
        if self.sha256 and self.sha256 != expected:
            raise ValueError("evidence image digest does not match its byte content")
        object.__setattr__(self, "image", np.ascontiguousarray(image))
        object.__setattr__(self, "sha256", expected)

    def manifest(self) -> dict[str, Any]:
        """Return the JSON-safe identity of this frame, excluding pixels."""
        return {
            "frame_id": self.frame_id,
            "camera": self.camera,
            "step": self.step,
            "phase": self.phase,
            "source": self.source,
            "shape": list(self.image.shape),
            "dtype": str(self.image.dtype),
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class EvidenceBundle:
    """Deterministic, task-conditioned evidence selected from one trial."""

    scene_id: str
    epoch: int
    seed: int | None
    instruction: str
    target: Mapping[str, Any] | None
    frames: tuple[EvidenceFrame, ...]

    def manifest(self) -> dict[str, Any]:
        """Return the canonical evidence manifest."""
        return {
            "scene_id": self.scene_id,
            "epoch": self.epoch,
            "seed": self.seed,
            "instruction": self.instruction,
            "target": jsonable(self.target),
            "frames": [frame.manifest() for frame in self.frames],
        }

    @property
    def sha256(self) -> str:
        """Digest the task context and exact selected-frame identities."""
        return digest_json(self.manifest())


@dataclass(frozen=True)
class JudgeRequest:
    """One complete, hashable request to a judge backend."""

    system_prompt: str
    user_prompt: str
    evidence: EvidenceBundle
    sample_index: int = 0

    def manifest(self) -> dict[str, Any]:
        """Return the request without inline image bytes."""
        return {
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "evidence": self.evidence.manifest(),
            "sample_index": self.sample_index,
        }

    @property
    def sha256(self) -> str:
        """Digest the full logical request."""
        return digest_json(self.manifest())


@dataclass(frozen=True)
class Judgement:
    """A structured visual judgement with bounded uncertainty fields."""

    verdict: Verdict
    progress: float
    success_probability: float
    confidence: float
    failure_mode: str | None
    evidence: tuple[str, ...]
    rationale: str

    def __post_init__(self) -> None:
        if self.verdict not in _VERDICTS:
            raise ValueError(f"unsupported verdict {self.verdict!r}")
        for name, value in (
            ("progress", self.progress),
            ("success_probability", self.success_probability),
            ("confidence", self.confidence),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and within [0, 1], got {value!r}")
        if not self.rationale.strip():
            raise ValueError("rationale must not be empty")
        if self.verdict == "unscorable" and self.confidence > 0.5:
            raise ValueError("an unscorable judgement cannot claim confidence above 0.5")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Judgement:
        """Validate a decoded model response."""
        evidence = value.get("evidence", ())
        if not isinstance(evidence, Sequence) or isinstance(evidence, str | bytes):
            raise TypeError("judgement evidence must be a sequence of frame ids")
        failure_mode = value.get("failure_mode")
        return cls(
            verdict=str(value["verdict"]),  # type: ignore[arg-type]
            progress=float(value["progress"]),
            success_probability=float(value["success_probability"]),
            confidence=float(value["confidence"]),
            failure_mode=None if failure_mode is None else str(failure_mode),
            evidence=tuple(str(item) for item in evidence),
            rationale=str(value["rationale"]),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe judgement."""
        return {
            "verdict": self.verdict,
            "progress": self.progress,
            "success_probability": self.success_probability,
            "confidence": self.confidence,
            "failure_mode": self.failure_mode,
            "evidence": list(self.evidence),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class JudgeRun:
    """One backend call and its replay-relevant response provenance."""

    judgement: Judgement
    backend: str
    model: str
    request_sha256: str
    response_sha256: str
    response: Mapping[str, Any]

    def manifest(self) -> dict[str, Any]:
        """Return a JSON-safe backend run."""
        return {
            "judgement": self.judgement.as_dict(),
            "backend": self.backend,
            "model": self.model,
            "request_sha256": self.request_sha256,
            "response_sha256": self.response_sha256,
            "response": jsonable(self.response),
        }


@dataclass(frozen=True)
class AggregateJudgement:
    """Conservative aggregate across one or more judge samples."""

    verdict: Verdict
    progress: float
    success_probability: float
    confidence: float
    abstained: bool
    disagreement: float
    failure_modes: tuple[str, ...]
    rationale: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe aggregate."""
        return {
            "verdict": self.verdict,
            "progress": self.progress,
            "success_probability": self.success_probability,
            "confidence": self.confidence,
            "abstained": self.abstained,
            "disagreement": self.disagreement,
            "failure_modes": list(self.failure_modes),
            "rationale": self.rationale,
        }
