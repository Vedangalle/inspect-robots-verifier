"""Operator-label benchmark metrics and selective-risk curves."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inspect_robots_verifier._json import canonical_json


@dataclass(frozen=True)
class BenchmarkCase:
    """One operator-labeled verifier prediction."""

    case_id: str
    ground_truth_success: bool
    verdict: str
    success_probability: float
    confidence: float
    slice: str = "all"

    def __post_init__(self) -> None:
        if self.verdict not in ("success", "partial", "failure", "unscorable"):
            raise ValueError(f"unsupported verdict {self.verdict!r}")
        for name, value in (
            ("success_probability", self.success_probability),
            ("confidence", self.confidence),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and within [0, 1]")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> BenchmarkCase:
        """Validate one decoded JSONL row."""
        label = value["ground_truth_success"]
        if not isinstance(label, bool):
            raise TypeError("ground_truth_success must be a JSON boolean")
        return cls(
            case_id=str(value["case_id"]),
            ground_truth_success=label,
            verdict=str(value["verdict"]),
            success_probability=float(value["success_probability"]),
            confidence=float(value["confidence"]),
            slice=str(value.get("slice", "all")),
        )


def evaluate(
    cases: Sequence[BenchmarkCase],
    *,
    min_confidence: float = 0.0,
    calibration_bins: int = 10,
) -> dict[str, Any]:
    """Compute conservative success-detection metrics on covered cases."""
    if not cases:
        raise ValueError("benchmark requires at least one case")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be within [0, 1]")
    if calibration_bins < 1:
        raise ValueError("calibration_bins must be at least 1")

    covered = [
        case for case in cases if case.verdict != "unscorable" and case.confidence >= min_confidence
    ]
    tp = sum(case.verdict == "success" and case.ground_truth_success for case in covered)
    fp = sum(case.verdict == "success" and not case.ground_truth_success for case in covered)
    tn = sum(case.verdict != "success" and not case.ground_truth_success for case in covered)
    fn = sum(case.verdict != "success" and case.ground_truth_success for case in covered)
    positives = tp + fn
    negatives = tn + fp
    tpr = _ratio(tp, positives)
    tnr = _ratio(tn, negatives)
    balanced_accuracy = None if tpr is None or tnr is None else (tpr + tnr) / 2
    errors = fp + fn
    return {
        "cases": len(cases),
        "covered": len(covered),
        "coverage": len(covered) / len(cases),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "false_positive_rate": _ratio(fp, negatives),
        "false_negative_rate": _ratio(fn, positives),
        "balanced_accuracy": balanced_accuracy,
        "selective_risk": _ratio(errors, len(covered)),
        "brier_score": (
            None
            if not covered
            else sum(
                (case.success_probability - float(case.ground_truth_success)) ** 2
                for case in covered
            )
            / len(covered)
        ),
        "expected_calibration_error": _ece(covered, calibration_bins),
        "min_confidence": min_confidence,
    }


def selective_curve(
    cases: Sequence[BenchmarkCase],
    thresholds: Sequence[float] = (0.0, 0.5, 0.65, 0.8, 0.9),
) -> list[dict[str, Any]]:
    """Evaluate risk and coverage across confidence thresholds."""
    return [evaluate(cases, min_confidence=threshold) for threshold in thresholds]


def evaluate_slices(
    cases: Sequence[BenchmarkCase],
    *,
    min_confidence: float = 0.0,
) -> dict[str, dict[str, Any]]:
    """Report the same metrics independently for every named slice."""
    names = sorted({case.slice for case in cases})
    return {
        name: evaluate(
            [case for case in cases if case.slice == name],
            min_confidence=min_confidence,
        )
        for name in names
    }


def read_cases(path: str | Path) -> list[BenchmarkCase]:
    """Read non-empty JSON objects from a JSONL file."""
    cases: list[BenchmarkCase] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from None
            if not isinstance(value, Mapping):
                raise ValueError(f"line {line_number} must contain one JSON object")
            cases.append(BenchmarkCase.from_mapping(value))
    if not cases:
        raise ValueError("benchmark file contains no cases")
    return cases


def main(argv: Sequence[str] | None = None) -> int:
    """Run benchmark metrics over an operator-labeled JSONL file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="operator-labeled prediction JSONL")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--output", help="optional output JSON path")
    args = parser.parse_args(argv)
    cases = read_cases(args.input)
    report = {
        "overall": evaluate(cases, min_confidence=args.min_confidence),
        "slices": evaluate_slices(cases, min_confidence=args.min_confidence),
        "selective_curve": selective_curve(cases),
    }
    rendered = canonical_json(report) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _ece(cases: Sequence[BenchmarkCase], bins: int) -> float | None:
    if not cases:
        return None
    total = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            case
            for case in cases
            if lower <= case.success_probability < upper
            or (index == bins - 1 and case.success_probability == 1.0)
        ]
        if not members:
            continue
        confidence = sum(case.success_probability for case in members) / len(members)
        accuracy = sum(case.ground_truth_success for case in members) / len(members)
        total += (len(members) / len(cases)) * abs(accuracy - confidence)
    return total


if __name__ == "__main__":
    raise SystemExit(main())
