"""Process-aware verifier aggregation and Inspect Robots scorer integration."""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Literal

from inspect_robots.scene import Target
from inspect_robots.scorer import Score
from inspect_robots_verifier.artifact import ArtifactWriter
from inspect_robots_verifier.backend import JudgeBackend, OpenAICompatibleJudge
from inspect_robots_verifier.evidence import EvidenceSampler, SamplingConfig
from inspect_robots_verifier.models import (
    AggregateJudgement,
    Judgement,
    JudgeRun,
    Verdict,
)
from inspect_robots_verifier.prompt import build_request

if TYPE_CHECKING:
    from inspect_robots.rollout import TrialRecord

OutputMode = Literal["verdict", "progress", "success"]


class ProcessAwareVerifier:
    """Conservative visual scorer with explicit abstention and audit artifacts."""

    name = "process_vlm"

    def __init__(
        self,
        backend: JudgeBackend,
        *,
        artifact_root: str | Path = "artifacts/verifier",
        sampler: EvidenceSampler | None = None,
        samples: int = 1,
        min_confidence: float = 0.65,
        max_disagreement: float = 0.25,
        success_threshold: float = 0.8,
        output: OutputMode = "verdict",
    ):
        if samples < 1:
            raise ValueError("samples must be at least 1")
        for name, value in (
            ("min_confidence", min_confidence),
            ("max_disagreement", max_disagreement),
            ("success_threshold", success_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if output not in ("verdict", "progress", "success"):
            raise ValueError(f"unsupported output mode {output!r}")
        self.backend = backend
        self.sampler = sampler or EvidenceSampler()
        self.writer = ArtifactWriter(artifact_root)
        self.samples = samples
        self.min_confidence = min_confidence
        self.max_disagreement = max_disagreement
        self.success_threshold = success_threshold
        self.output = output

    def __call__(self, record: TrialRecord, target: Target | None) -> Score:
        """Score one trajectory and persist the exact evidence and responses."""
        evidence = self.sampler.sample(record, target)
        request_count = self.samples if evidence.frames else 1
        requests = tuple(build_request(evidence, index) for index in range(request_count))
        runs = tuple(self.backend.judge(request) for request in requests) if evidence.frames else ()
        aggregate = self._aggregate(runs, evidence_ids={f.frame_id for f in evidence.frames})
        artifact = self.writer.write(requests, runs, aggregate)
        metadata = {
            "aggregate": aggregate.as_dict(),
            "artifact": artifact.as_dict(),
            "backend": self.backend.backend_name,
            "model": self.backend.model,
            "samples": len(runs),
        }
        record.metadata.setdefault("verifier_artifacts", []).append(artifact.as_dict())
        return Score(
            value=self._score_value(aggregate),
            explanation=aggregate.rationale,
            metadata=metadata,
        )

    def _aggregate(
        self,
        runs: Sequence[JudgeRun],
        *,
        evidence_ids: set[str],
    ) -> AggregateJudgement:
        if not runs:
            return _abstention("No camera evidence was recorded for visual verification.")
        judgements = [run.judgement for run in runs]
        unknown_evidence = sorted(
            {
                frame_id
                for judgement in judgements
                for frame_id in judgement.evidence
                if frame_id not in evidence_ids
            }
        )
        probabilities = [judgement.success_probability for judgement in judgements]
        disagreement = max(probabilities) - min(probabilities)
        confidence = float(median(judgement.confidence for judgement in judgements))
        progress = float(median(judgement.progress for judgement in judgements))
        success_probability = float(median(probabilities))
        failure_modes = tuple(
            sorted({mode for judgement in judgements if (mode := judgement.failure_mode)})
        )
        reasons: list[str] = []
        if unknown_evidence:
            reasons.append(f"judge cited unknown evidence ids: {unknown_evidence}")
        if any(judgement.verdict == "unscorable" for judgement in judgements):
            reasons.append("at least one judge sample found the evidence unscorable")
        if confidence < self.min_confidence:
            reasons.append(f"median confidence {confidence:.3f} is below {self.min_confidence:.3f}")
        if disagreement > self.max_disagreement:
            reasons.append(
                f"success-probability spread {disagreement:.3f} exceeds {self.max_disagreement:.3f}"
            )

        votes = Counter(judgement.verdict for judgement in judgements)
        ordered = votes.most_common()
        tied = len(ordered) > 1 and ordered[0][1] == ordered[1][1]
        if tied:
            reasons.append("judge samples do not have a unique verdict majority")
        candidate: Verdict = ordered[0][0]
        if candidate == "success" and success_probability < self.success_threshold:
            reasons.append(
                f"success probability {success_probability:.3f} is below "
                f"{self.success_threshold:.3f}"
            )
        if reasons:
            return AggregateJudgement(
                verdict="unscorable",
                progress=progress,
                success_probability=success_probability,
                confidence=confidence,
                abstained=True,
                disagreement=disagreement,
                failure_modes=failure_modes,
                rationale="Abstained: " + "; ".join(reasons) + ".",
            )
        return AggregateJudgement(
            verdict=candidate,
            progress=progress,
            success_probability=success_probability,
            confidence=confidence,
            abstained=False,
            disagreement=disagreement,
            failure_modes=failure_modes,
            rationale=_aggregate_rationale(judgements),
        )

    def _score_value(self, aggregate: AggregateJudgement) -> bool | float | str:
        if self.output == "verdict":
            return aggregate.verdict
        if self.output == "progress":
            return aggregate.progress
        return aggregate.verdict == "success" and not aggregate.abstained


def process_vlm_scorer(
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str = "OPENROUTER_API_KEY",
    artifact_root: str = "artifacts/verifier",
    frames_per_camera: int = 5,
    max_cameras: int = 4,
    samples: int = 1,
    min_confidence: float = 0.65,
    max_disagreement: float = 0.25,
    success_threshold: float = 0.8,
    output: OutputMode = "verdict",
) -> ProcessAwareVerifier:
    """Registry factory for an OpenAI-compatible process-aware verifier."""
    resolved_model = model or os.environ.get("INSPECT_ROBOTS_VERIFIER_MODEL")
    if not resolved_model:
        raise ValueError(
            "no verifier model configured; pass model=... or set INSPECT_ROBOTS_VERIFIER_MODEL"
        )
    resolved_base = (
        base_url
        or os.environ.get("INSPECT_ROBOTS_VERIFIER_BASE_URL")
        or "https://openrouter.ai/api/v1"
    )
    backend = OpenAICompatibleJudge(
        base_url=resolved_base,
        api_key=os.environ.get(api_key_env, ""),
        model=resolved_model,
    )
    return ProcessAwareVerifier(
        backend,
        artifact_root=artifact_root,
        sampler=EvidenceSampler(
            SamplingConfig(
                frames_per_camera=frames_per_camera,
                max_cameras=max_cameras,
            )
        ),
        samples=samples,
        min_confidence=min_confidence,
        max_disagreement=max_disagreement,
        success_threshold=success_threshold,
        output=output,
    )


def _abstention(rationale: str) -> AggregateJudgement:
    return AggregateJudgement(
        verdict="unscorable",
        progress=0.0,
        success_probability=0.0,
        confidence=0.0,
        abstained=True,
        disagreement=0.0,
        failure_modes=(),
        rationale=rationale,
    )


def _aggregate_rationale(judgements: Sequence[Judgement]) -> str:
    unique = list(dict.fromkeys(judgement.rationale.strip() for judgement in judgements))
    return " | ".join(unique)
