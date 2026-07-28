# Design

## Objective

Produce a visual robot-task judgement that is:

1. more informative than a final-frame binary classifier;
2. conservative under missing or conflicting evidence;
3. attributable to exact recorded frames;
4. reproducible at the request and response boundary; and
5. compatible with the existing Inspect Robots `Scorer` protocol.

The verifier is an evaluation instrument, not a robot controller. It never
changes actions or interacts with hardware.

## Data flow

```text
TrialRecord + Target
        |
        v
deterministic EvidenceSampler
        |
        +-- initial frame per camera
        +-- temporal quantile frames
        +-- terminal post-action frame when available
        |
        v
task-conditioned JudgeRequest
        |
        v
one or more JudgeBackend calls
        |
        v
validated structured Judgement values
        |
        v
conservative aggregation and abstention
        |
        +-- Inspect Robots Score
        +-- artifact reference in TrialRecord.metadata
        +-- lossless evidence + verification.json
```

## Evidence invariants

- Camera names are sorted before selection.
- A fixed `TrialRecord`, `Target`, and `SamplingConfig` produces the same frames
  and bundle digest.
- Sampling budgets apply independently per selected camera.
- Earliest and latest available frames are retained when a camera has at least
  two candidates.
- Intermediate indices are deterministic temporal quantiles.
- Image identity includes dtype and shape as well as contiguous pixel bytes.
- The model receives explicit frame identifiers and must cite only those
  identifiers.
- `.npy` artifact files preserve exact arrays without lossy image compression.

The sampler checks both inline observations and rollout-owned `FrameRef` values.
It also considers the final `StepResult.observation`, which captures a post-action
state that may not appear as the next pre-action observation when a trial ends.

## Judgement schema

Every backend response must contain:

- `verdict`: `success`, `partial`, `failure`, or `unscorable`;
- `progress`: observed task progress in `[0, 1]`;
- `success_probability`: model-assigned completion probability in `[0, 1]`;
- `confidence`: confidence that the available evidence supports the judgement;
- `failure_mode`: concise optional label;
- `evidence`: supplied frame identifiers;
- `rationale`: concise observable basis for the verdict.

The rationale is an audit explanation, not a request for hidden chain-of-thought.

## Aggregation

For repeated judgements, the verifier computes medians for progress,
success probability, and confidence. Disagreement is the range of success
probabilities.

The result is forced to `unscorable` when any of these conditions holds:

- a cited frame identifier was not supplied;
- any sample says the evidence is unscorable;
- median confidence is below `min_confidence`;
- probability range exceeds `max_disagreement`;
- verdict votes do not have a unique majority;
- the majority says success but median success probability is below
  `success_threshold`.

These rules intentionally optimize against false positive success labels. They
are engineering safeguards, not statistical calibration. Thresholds must be
selected on a held-out operator-labeled set before a production benchmark.

## Audit artifact

The writer emits a versioned manifest with:

- algorithm and schema versions;
- scene, epoch, seed, instruction, and target;
- exact frame metadata, hashes, and relative files;
- system and user prompts;
- sample indices and request hashes;
- backend and model identifiers;
- exact decoded API responses and response hashes;
- aggregate verdict and uncertainty diagnostics;
- a digest of the complete manifest content.

API keys are supplied only in HTTP authorization headers and are never written to
the artifact.

## Integration boundary

The package is out-of-tree and registers `process_vlm` through
`inspect_robots.scorers`. This keeps network and model concerns out of the
NumPy-only core.

Two small upstream changes would materially improve the design:

1. preserve scorer explanations and metadata in the persisted evaluation log;
2. persist the terminal post-action camera observation through `FrameStore`.

Until then, the scorer persists its own artifact and appends the artifact
reference to `TrialRecord.metadata` before the framework collects trial metadata.

## Threats to validity

- A model can be consistently wrong across all repeated samples.
- Temporal quantiles can miss a brief but decisive event.
- Cameras may share a blind spot or be temporally unsynchronized.
- A natural-language target may be underspecified.
- Confidence values produced by a model are not calibrated probabilities.
- A judge evaluated on data from its own training distribution may overstate
  generalization.

The planned benchmark therefore includes human adjudication, near-miss negatives,
camera ablations, distribution-shift slices, and selective-risk curves.
