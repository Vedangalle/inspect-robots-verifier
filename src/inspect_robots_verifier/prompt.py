"""Task-conditioned, process-aware judge prompts."""

from __future__ import annotations

from inspect_robots_verifier._json import canonical_json
from inspect_robots_verifier.models import EvidenceBundle, JudgeRequest

SYSTEM_PROMPT = """You are a conservative robot-trajectory verifier.
Judge only what the supplied visual evidence establishes. Compare terminal state
against the initial state, inspect intermediate progress, and distinguish task
completion from approach, contact, or an unstable near-miss. Never infer hidden
state. If the views are missing, contradictory, occluded, or insufficient, return
unscorable. False positives are more costly than abstention.

Return exactly one JSON object with:
- verdict: success | partial | failure | unscorable
- progress: number in [0,1]
- success_probability: number in [0,1]
- confidence: number in [0,1]
- failure_mode: short string or null
- evidence: array of supplied frame_id strings
- rationale: concise observable evidence, not private chain-of-thought
"""


def build_request(evidence: EvidenceBundle, sample_index: int = 0) -> JudgeRequest:
    """Build a stable judge request from one evidence bundle."""
    frame_index = [
        {
            "frame_id": frame.frame_id,
            "camera": frame.camera,
            "step": frame.step,
            "phase": frame.phase,
            "source": frame.source,
            "sha256": frame.sha256,
        }
        for frame in evidence.frames
    ]
    user_prompt = "\n".join(
        [
            f"Instruction: {evidence.instruction or '[not recorded]'}",
            f"Target: {canonical_json(evidence.target)}",
            "Evaluation rubric:",
            "1. Is the requested goal state visibly satisfied?",
            "2. Do initial-to-terminal changes support actual task progress?",
            "3. Do process frames reveal a failed grasp, drop, collision, reversal, or near-miss?",
            "4. Is the terminal state stable and unambiguous in the available views?",
            f"Evidence index: {canonical_json(frame_index)}",
            "The images follow in the same order as the evidence index.",
        ]
    )
    return JudgeRequest(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        evidence=evidence,
        sample_index=sample_index,
    )
