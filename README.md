# Inspect Robots Verifier

Auditable, process-aware visual verification for
[Inspect Robots](https://github.com/robocurve/inspect-robots) trajectories.

The verifier asks a vision-language model to evaluate a task from a deterministic
sequence of initial, intermediate, and terminal frames. It returns a structured
verdict, abstains when the evidence or model response is unreliable, and writes
the exact evidence, prompts, responses, hashes, and aggregation decision to a
replay artifact.

This is a working research prototype. Its software behavior is tested; its
robot-task accuracy is **not yet benchmarked**, so this repository does not claim
state-of-the-art empirical performance.

## Why this exists

Final-frame success detection fails on important physical-AI cases:

- a robot touches the correct object but never establishes a grasp;
- a grasp succeeds and the object is dropped before the final frame;
- one camera shows apparent completion while another shows a near-miss;
- the goal state is occluded or outside the recorded view;
- a model produces a confident answer without citing supplied evidence.

Inspect Robots already records trajectories and reserves a `VLMScorer` interface.
This package turns that interface into an evidence protocol rather than a
single-image classifier.

## What is implemented

- deterministic temporal quantile sampling per camera;
- initial-state reference, process frames, and terminal evidence;
- sorted multi-camera selection with explicit budgets;
- task-conditioned rubric built from the instruction and `Target`;
- OpenAI-compatible vision backend with structured JSON output;
- backend-independent replay mode for exact offline tests;
- repeated-judgement aggregation with conservative abstention;
- rejection of hallucinated frame identifiers;
- categorical verdict, continuous progress, or verified-success outputs;
- operator-label benchmark metrics and selective risk-versus-coverage curves;
- self-contained `.npy` evidence and canonical JSON audit artifacts;
- SHA-256 identities for every frame, evidence bundle, request, response, and
  complete verification artifact;
- Inspect Robots scorer entry point named `process_vlm`.

## Quick start

Install the package next to a current Inspect Robots installation:

```bash
python -m pip install -e ".[dev]"
```

Configure any vision model served through an OpenAI-compatible Chat Completions
endpoint:

```bash
export INSPECT_ROBOTS_VERIFIER_MODEL="provider/model"
export OPENROUTER_API_KEY="..."
```

Then construct the scorer in Python:

```python
from inspect_robots_verifier import process_vlm_scorer

scorer = process_vlm_scorer(
    samples=3,
    frames_per_camera=5,
    max_cameras=3,
    output="verdict",
)
```

Because the package publishes an `inspect_robots.scorers` entry point, an
installed package is also discoverable as `process_vlm` through the framework
registry.

For a network-free end-to-end run:

```bash
python examples/replay_demo.py
```

The demo constructs an Inspect Robots `TrialRecord`, runs three recorded
judgements through the full scorer, and writes the same artifact layout used by
a live model.

## Output contract

The default score is categorical:

- `success`: visible evidence supports completion above the configured
  probability threshold;
- `partial`: observable progress without established completion;
- `failure`: observable non-completion or failure;
- `unscorable`: insufficient evidence, low confidence, disagreement, invalid
  evidence citations, or no unique majority.

`output="progress"` returns median progress in `[0, 1]`.
`output="success"` returns a boolean and maps every abstention to `False`; the
metadata still records `abstained=true`, so callers must not interpret that value
as an observed failure.

Each artifact directory contains:

```text
artifacts/verifier/<scene>-e<epoch>-<request>/
├── frames/
│   ├── 000-<sha256>.npy
│   └── ...
└── verification.json
```

`verification.json` records the logical request without duplicated inline pixels,
the exact backend response, all component hashes, the aggregate decision, and
relative paths to lossless frame arrays.

## Validation

Run the local quality gates:

```bash
ruff check .
ruff format --check .
mypy
coverage run -m pytest
coverage report
```

The current suite covers deterministic sampling, sidecar `FrameRef` loading,
request construction, PNG encoding, API failures, JSON validation, replay,
uncertainty gates, output modes, benchmark metrics, and artifact round trips.

An operator-labeled benchmark file uses one JSON object per line:

```json
{"case_id":"run-001","ground_truth_success":true,"verdict":"success","success_probability":0.93,"confidence":0.88,"slice":"nominal"}
```

Generate overall, per-slice, and selective-risk metrics with:

```bash
inspect-robots-verifier-benchmark labels.jsonl --output report.json
```

## Research plan

The verifier protocol is informed by recent findings that robot success
evaluation benefits from temporal evidence, hard negative and near-miss examples,
and process-aware diagnostics:

- [RoboReward](https://arxiv.org/abs/2601.00675)
- [RoboProcessBench](https://arxiv.org/abs/2606.13040)
- [RobotArena Infinity](https://openreview.net/forum?id=OutljIofvS)
- [Vision-Language Models for Robot Success Detection](https://ojs.aaai.org/index.php/AAAI/article/view/30552)

The next empirical milestone is a preregistered comparison on operator-labeled
Inspect Robots logs:

1. final frame only;
2. initial plus final frames;
3. temporal single-view evidence;
4. temporal multi-view evidence;
5. temporal multi-view evidence with repeated judgements and abstention.

Primary metrics should be false-positive rate, selective risk versus coverage,
balanced accuracy, calibration error, and agreement with independent operators.
Near-miss, occlusion, and distribution-shift slices must be reported separately.

## Known limitations

- No operator-labeled benchmark results exist yet.
- Repeated calls to one model are correlated and are not independent statistical
  samples; disagreement is a safety heuristic, not a confidence interval.
- The generic backend relies on prompt-constrained JSON. Provider-native strict
  schemas should be added where available.
- The current Inspect Robots score aggregation path does not preserve
  `Score.explanation` and `Score.metadata` in its compact `EvalLog`. The plugin
  therefore writes a durable artifact and stores its reference in
  `TrialRecord.metadata`.
- A terminal observation held in memory can be scored immediately, but current
  frame-sidecar behavior does not guarantee that the terminal post-action image
  is available for later offline re-scoring. That upstream provenance gap should
  be fixed before calling the workflow fully replayable from an EvalLog alone.

See [DESIGN.md](DESIGN.md) for invariants, aggregation rules, and integration
boundaries.
