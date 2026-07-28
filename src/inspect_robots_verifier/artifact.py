"""Replay-grade verifier artifact persistence."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from inspect_robots_verifier._json import canonical_json, digest_json
from inspect_robots_verifier.models import (
    AggregateJudgement,
    JudgeRequest,
    JudgeRun,
)

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class ArtifactRef:
    """Identity and location of one persisted verification artifact."""

    path: str
    sha256: str
    evidence_sha256: str

    def as_dict(self) -> dict[str, str]:
        """Return a JSON-safe reference."""
        return {
            "path": self.path,
            "sha256": self.sha256,
            "evidence_sha256": self.evidence_sha256,
        }


class ArtifactWriter:
    """Persist exact selected arrays plus canonical request/response metadata."""

    schema_version = 1
    algorithm_version = "process-aware-v1"

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def write(
        self,
        requests: tuple[JudgeRequest, ...],
        runs: tuple[JudgeRun, ...],
        aggregate: AggregateJudgement,
    ) -> ArtifactRef:
        """Write one self-contained directory atomically at the manifest level."""
        if not requests:
            raise ValueError("at least one request is required")
        evidence = requests[0].evidence
        request_token = requests[0].sha256[:12]
        scene = _safe(evidence.scene_id)
        directory = self.root / f"{scene}-e{evidence.epoch}-{request_token}"
        frame_directory = directory / "frames"
        frame_directory.mkdir(parents=True, exist_ok=True)

        frame_manifests: list[dict[str, Any]] = []
        for index, frame in enumerate(evidence.frames):
            relative = Path("frames") / f"{index:03d}-{frame.sha256[:12]}.npy"
            final_path = directory / relative
            temporary = final_path.with_suffix(".npy.tmp")
            with temporary.open("wb") as handle:
                np.save(handle, frame.image, allow_pickle=False)
            os.replace(temporary, final_path)
            item = frame.manifest()
            item["file"] = str(relative)
            frame_manifests.append(item)

        body = {
            "schema_version": self.schema_version,
            "algorithm_version": self.algorithm_version,
            "evidence": {
                **evidence.manifest(),
                "frames": frame_manifests,
                "sha256": evidence.sha256,
            },
            "requests": [request.manifest() for request in requests],
            "runs": [run.manifest() for run in runs],
            "aggregate": aggregate.as_dict(),
        }
        artifact_sha256 = digest_json(body)
        manifest = {**body, "artifact_sha256": artifact_sha256}
        path = directory / "verification.json"
        temporary_manifest = directory / ".verification.json.tmp"
        temporary_manifest.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
        os.replace(temporary_manifest, path)
        return ArtifactRef(
            path=str(path),
            sha256=artifact_sha256,
            evidence_sha256=evidence.sha256,
        )


def _safe(value: str) -> str:
    normalized = _SAFE.sub("-", value).strip("-")
    return normalized or "scene"
