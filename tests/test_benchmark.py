from __future__ import annotations

import json
from pathlib import Path

import pytest

from inspect_robots_verifier.benchmark import (
    BenchmarkCase,
    evaluate,
    evaluate_slices,
    main,
    read_cases,
    selective_curve,
)


def cases() -> list[BenchmarkCase]:
    return [
        BenchmarkCase("tp", True, "success", 0.9, 0.9, "nominal"),
        BenchmarkCase("fp", False, "success", 0.8, 0.8, "near_miss"),
        BenchmarkCase("tn", False, "failure", 0.1, 0.9, "nominal"),
        BenchmarkCase("fn", True, "partial", 0.4, 0.7, "near_miss"),
        BenchmarkCase("abstain", False, "unscorable", 0.5, 0.2, "occlusion"),
    ]


def test_evaluate_reports_risk_coverage_calibration() -> None:
    report = evaluate(cases(), calibration_bins=5)
    assert report["cases"] == 5
    assert report["covered"] == 4
    assert report["coverage"] == pytest.approx(0.8)
    assert report["confusion"] == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
    assert report["false_positive_rate"] == pytest.approx(0.5)
    assert report["false_negative_rate"] == pytest.approx(0.5)
    assert report["balanced_accuracy"] == pytest.approx(0.5)
    assert report["selective_risk"] == pytest.approx(0.5)
    assert report["brier_score"] == pytest.approx((0.01 + 0.64 + 0.01 + 0.36) / 4)
    assert report["expected_calibration_error"] == pytest.approx(0.35)


def test_confidence_threshold_and_empty_coverage() -> None:
    report = evaluate(cases(), min_confidence=0.85)
    assert report["covered"] == 2
    assert report["confusion"] == {"tp": 1, "fp": 0, "tn": 1, "fn": 0}
    empty = evaluate(cases(), min_confidence=1.0)
    assert empty["covered"] == 0
    assert empty["selective_risk"] is None
    assert empty["brier_score"] is None
    assert empty["expected_calibration_error"] is None


def test_selective_curve_and_slices_are_deterministic() -> None:
    curve = selective_curve(cases(), thresholds=(0.0, 0.9))
    assert [item["coverage"] for item in curve] == [0.8, 0.4]
    slices = evaluate_slices(cases())
    assert list(slices) == ["near_miss", "nominal", "occlusion"]
    assert slices["occlusion"]["coverage"] == 0.0


def test_case_and_metric_validation() -> None:
    assert (
        BenchmarkCase.from_mapping(
            {
                "case_id": 1,
                "ground_truth_success": True,
                "verdict": "success",
                "success_probability": 1,
                "confidence": 1,
            }
        ).slice
        == "all"
    )
    with pytest.raises(TypeError, match="boolean"):
        BenchmarkCase.from_mapping(
            {
                "case_id": "bad",
                "ground_truth_success": 1,
                "verdict": "success",
                "success_probability": 1,
                "confidence": 1,
            }
        )
    with pytest.raises(ValueError, match="unsupported"):
        BenchmarkCase("bad", True, "maybe", 0.5, 0.5)
    with pytest.raises(ValueError, match="success_probability"):
        BenchmarkCase("bad", True, "success", float("nan"), 0.5)
    with pytest.raises(ValueError, match="at least one"):
        evaluate([])
    with pytest.raises(ValueError, match="min_confidence"):
        evaluate(cases(), min_confidence=2)
    with pytest.raises(ValueError, match="calibration_bins"):
        evaluate(cases(), calibration_bins=0)


def test_read_cases_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    input_path = tmp_path / "cases.jsonl"
    input_path.write_text(
        "\n"
        + "\n".join(
            json.dumps(
                {
                    "case_id": case.case_id,
                    "ground_truth_success": case.ground_truth_success,
                    "verdict": case.verdict,
                    "success_probability": case.success_probability,
                    "confidence": case.confidence,
                    "slice": case.slice,
                }
            )
            for case in cases()
        ),
        encoding="utf-8",
    )
    assert len(read_cases(input_path)) == 5
    assert main([str(input_path), "--min-confidence", "0.8"]) == 0
    stdout = json.loads(capsys.readouterr().out)
    assert stdout["overall"]["min_confidence"] == 0.8

    output = tmp_path / "report.json"
    assert main([str(input_path), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["overall"]["cases"] == 5


def test_read_cases_rejects_invalid_files(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no cases"):
        read_cases(empty)
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text("{\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        read_cases(invalid)
    array = tmp_path / "array.jsonl"
    array.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        read_cases(array)
