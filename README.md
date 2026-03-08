# autoresearch

Colab-friendly notes, notebooks, and helper patches for experimenting with `karpathy/autoresearch`.

This repository is focused on making the upstream project easier to explore from Google Colab, especially for quick validation runs, debugging, and lower-memory setups.

## Why this repo exists

The upstream project is exciting, but running it smoothly in Colab can require a few practical adjustments:

- Colab GPUs vary widely across free and paid tiers
- memory limits can make default settings too aggressive
- some environment assumptions may not hold in notebook sessions
- fast iteration is easier with staged notebooks than with one long setup/debug loop

This repo collects a lightweight workflow for addressing those issues.

## Current goals

- make environment setup reproducible in Colab
- provide a small-step workflow for testing before full training
- document known Colab-specific friction points
- keep changes minimal and easy to understand

## Repository contents

- `docs/colab-findings.md`
- `requirements-colab.txt`
- `notebooks/01_environment_setup.ipynb`
- `notebooks/02_prepare_data.ipynb`
- `notebooks/03_train_smoke_test.ipynb`
- `patches/train_colab.py`
- `patches/prepare_colab.py`
- `.gitignore`

## Recommended workflow

### 1. Start with environment setup
Run `notebooks/01_environment_setup.ipynb`.

Use this to confirm:
- Python packages install cleanly
- GPU is visible
- PyTorch CUDA is working
- flash attention dependencies are in a reasonable state

### 2. Validate data preparation on a small run
Run `notebooks/02_prepare_data.ipynb`.

Use a small shard count first. The goal is to confirm the pipeline works before spending time on larger preprocessing jobs.

### 3. Launch a smoke-test training run
Run `notebooks/03_train_smoke_test.ipynb`.

This should be treated as a sanity check, not a full benchmark. Confirm that:
- training starts
- imports resolve
- attention code paths work
- memory usage is acceptable
- dtype selection is compatible with the current GPU

### 4. Scale up only after validation
Once the smoke test succeeds, increase workload gradually:
- larger shard counts
- larger batch sizes
- longer runs
- stronger GPU runtimes if available

## Key Colab findings so far

### Missing `kernels.py`
The upstream training setup appears to reference a custom kernel wrapper that is not present in this workflow.

Practical Colab approach:
- replace that dependency with standard `flash-attn` usage where possible
- validate imports early before attempting long runs

### Precision can depend on GPU type
Different Colab GPUs behave differently.

General guidance:
- prefer `bfloat16` on newer GPUs such as A100 or L4
- use `float16` fallback on T4 if `bfloat16` causes issues

### Default batch sizes may be too large
What works in a stronger environment may be too large for Colab VRAM limits.

Practical advice:
- begin with a small `DEVICE_BATCH_SIZE`
- only scale after confirming stable memory usage

## What this repo is not

This repo is not trying to be:
- a fork that fully replaces the upstream project
- a definitive benchmark setup
- a production training framework

Instead, it is a practical Colab companion for experimentation and debugging.

## Suggested next improvements

Potential future additions:
- `notebooks/04_full_training_notes.ipynb`
- clearer patch examples against upstream files
- GPU-specific tuning notes for T4 vs L4 vs A100
- a more explicit end-to-end Colab walkthrough

## Usage philosophy

The intended approach is:

1. get the environment working
2. verify preparation on a tiny run
3. verify training on a tiny run
4. only then invest in bigger experiments

That sequence reduces wasted Colab time and makes failures easier to diagnose.

## Upstream reference

This repo is meant to support experimentation around:

- `karpathy/autoresearch`

If you use this scaffold, compare behavior against upstream regularly so that local Colab workarounds do not drift too far from the original codebase.