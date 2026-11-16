# cuheng-ple — PLE agent harness

Entry for the Kaggle competition **Google — Agent Harness on
Physical-Engineering's Last Exam** (deadline 2027-01-05). The harness solves
professional engineering tasks (structural, civil, electrical, mechanical)
inside task containers over the PLE serve protocol.

## Architecture

Based on the official `plan-verify` sample (google-research/ple), extended
over time. Three roles in separate conversations:

1. **Planner** — explores the task files, writes a structured plan per
   deliverable (path, format, engineering method, verification method).
2. **Worker** — executes the plan, produces the deliverables.
3. **Verifier** — fresh conversation; inspects deliverables against the task
   instruction and the plan, replies PASS or a list of issues; issues feed
   back into the worker for fix rounds.

Design priorities, driven by the competition scoring
(`s = 0.4p + 0.6·G·O′`, where one broken hard constraint sets G=0):

1. Always leave deliverables at the declared paths (missing = 0).
2. Never break a hard (safety/feasibility) constraint.
3. Verify before finishing; fix rounds beat raw speed.
4. Optimize the design objective last; watch cost (Pareto ranking).

## Files

| File | Purpose |
|---|---|
| `serve.py` | The harness: WebSocket server + planner/worker/verifier loop |
| `taskbridge.py` | PLE serve-protocol implementation (from the official kit) |
| `submission.yaml` | Install/serve commands + declared model |
| `requirements.txt` | Pinned Python dependencies |

## Model configuration

`submission.yaml` currently declares a **local-dev** model
(`openai-compatible/gpt-6-1-sol` via Coarena) so the dev set can be run and
`artifacts/` generated locally, as the competition instructions allow.
Before the final commit it is switched to the Stage 2 target
(`google-cloud/claude-opus-5-5`) and `model_base_url` is removed.
The API key is supplied via the `MODEL_API_KEY` environment variable /
`--env-file`; it is never committed.

## Local testing without Docker

`serve.py` can run standalone on a host (no egress proxy, no containers):

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
MODEL_API_KEY=<key> OPENAI_BASE_URL=https://coarena.ai/api/v1 \
  MODEL_NAME=gpt-6-1-sol PORT=8765 .venv/bin/python serve.py
```

`tools/fake_evaluator.py` (outside this repo during development) speaks the
evaluator side of the protocol against a local sandbox directory for smoke
tests. Official runs (`kit check`, `eval/run.py`) require a Docker machine
and happen there, not on the dev box.
