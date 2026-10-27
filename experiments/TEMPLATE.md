# <ID> — <one-line claim, e.g. "Seed ckpt under-predicts Jaipur foreground by ~5x">

| | |
|---|---|
| **Status** | planned / running / ✅ done / ❌ abandoned |
| **Date** | YYYY-MM-DD |
| **Commit** | `<sha>` on `<branch>` — the code that produced this |
| **Supersedes** | *(experiment ID, or —)* |

## Question

What is being asked, in one or two sentences.

**Why it matters:** which assumption this replaces, or which decision it gates. If the answer
is "nothing changes either way", do not run it. Name the planning doc section it feeds
(e.g. `MASTER_CONTEXT §3.1`, run-table row R7).

**Prediction (write BEFORE running):** what you expect, and roughly what number. Recording
this is what separates a result from a rationalisation — a surprise is only visible if the
expectation was written down first.

## Setup

| | |
|---|---|
| Data | dataset, split, how many samples, where on disk |
| Model / ckpt | arch, encoder, checkpoint path |
| Hardware | GPU(s), or CPU-only |
| Container | `nvcr.io/nvidia/pytorch:24.05-py3` via `run_docker.sh` |
| Runtime | wall-clock |

```bash
# the exact command, copy-pasteable
./run_docker.sh <gpu> "python -u <script> <args>"
```

## Results

Raw output: `<path to json/csv>`

| metric | value |
|---|---|
| | |

## Interpretation

What the numbers mean. Be willing to write "this did not work" — a negative result that kills
a planned direction is worth more than a marginal positive one, and the thesis needs the
reasoning either way.

## Decision

- [ ] What happens next as a direct consequence.
- [ ] What this unblocks, or kills.

## Threats to validity

The reasons this number might be wrong or narrower than it looks. Label-semantics mismatches,
sampling bias, degenerate sweeps, train/test leakage, single-seed noise. **Anything here that
would embarrass you in a viva goes in this section, not in a footnote.**

## Reproduce

Anything a fresh clone needs beyond the command above: data that must be restored first,
env vars, expected runtime, known failure modes.
