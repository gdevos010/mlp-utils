ABOUTME: Actionable checklist for integrating and hardening Tversky neural modules.
ABOUTME: Follows test-first, incremental steps; each item wires into repo without orphans.

# Tversky Neural Network (TNN) Checklist

Reference: [Tversky Neural Networks: Psychologically Plausible Deep Learning with Differentiable Tversky Similarity](https://arxiv.org/html/2506.11035v1)

## Repo integration targets (confirm before starting)
- [x] Confirm current branch is `feat/tnn-parity-tversky`
- [x] Activate env: `source /home/gdevos/structural_break/mlp-utils/.venv/bin/activate`
- [x] Code location: `src/mlp_utils/layers/tversky.py`
- [x] Public exports: `src/mlp_utils/layers/__init__.py`
- [x] Tests: `tests/test_tversky.py`
- [x] Docs: `README.md` (Layers section)
- [x] Examples: `experiments/tversky_xor.py`, `experiments/train_mnist_tversky.py`

## Global invariants (apply to every change)
- [ ] Add or extend a test first
- [ ] Implement the minimal code for the test
- [ ] Update public exports if needed
- [ ] Update README and/or example scripts to reference new code (behind a flag if optional)
- [ ] Run `pytest -q`; keep commits small and isolated

---

## Phase 0 — Baseline acceptance
- [x] Run all tests: `pytest -q`
- [x] Capture baseline coverage: `pytest --cov=mlp_utils --cov-report=term-missing`
- [x] Verify public API in README aligns with `src/mlp_utils/layers/__init__.py`
- [x] Commit: "tnn: record baseline tests and coverage"

Acceptance:
- [x] Tests green
- [x] Baseline metrics recorded

---

## Phase 1 — API stabilization and docs
- [ ] Docstrings: ensure complete parameter/shape docs for `tversky.py` public APIs
- [ ] README: add Tversky quickstart, tips (α/β asymmetry, θ stability, transforms, τ), examples links
- [ ] Add small API reference table for TNN modules
- [ ] Run tests: `pytest -q`
- [ ] Commit: "tnn: docstrings and README quickstart/reference"

Acceptance:
- [ ] Docs readable; links valid; no behavior change; tests green

---

## Phase 2 — Testing expansion and numerical robustness
- [ ] Property tests (Hypothesis): bounds [0,1], asymmetry when α≠β, monotonicity in shared mass, broadcast shapes
- [ ] Stress tests: extreme magnitudes, sparse patterns, degenerate pairs (zeros, disjoint)
- [ ] Precision/dtypes: float32/float64 parity; large D; product reduction underflow guard
- [ ] TorchCompile/AMP smoke: `TverskyProjection`, `TverskyFeatureSharing`
- [ ] Run tests: `pytest -q`
- [ ] Commit: "tnn: robustness and property-based tests"

Acceptance:
- [ ] New tests pass; coverage improves

---

## Phase 3 — Interpretability and visualization
- [ ] Add helper `explain_similarity(input, prototype)` that wraps `tversky_attributions` and returns named components with sums
- [ ] Tests: shape and sum consistency vs `tversky_similarity`
- [ ] Add optional plotting utils (matplotlib) in `experiments/utils_plot.py` (used only by examples)
- [ ] Wire XOR example with `--plot` flag to generate I/A/B bars
- [ ] Run tests: `pytest -q`
- [ ] Commit: "tnn: explain helper and optional plotting"

Acceptance:
- [ ] Helper tested; example produces figures when `--plot` is used

---

## Phase 4 — Initialization, seeding, and diversity regularization
- [ ] Add `seed_prototypes_from_loader` (dataset-driven seeding by class; supports stage1 memberships when feature-sharing)
- [ ] Tests: deterministic with fixed seed; weight shapes and assignments correct
- [ ] Add `prototype_diversity_loss(weights, margin)` to penalize collapse
- [ ] Tests: loss decreases with increased separation; zero on orthogonal prototypes (within tolerance)
- [ ] Wire flags into MNIST example: `--seed-prototypes`, `--seed-samples-per-class`, `--diversity-margin`
- [ ] Run tests: `pytest -q`
- [ ] Commit: "tnn: seeding helper and diversity regularizer + example wiring"

Acceptance:
- [ ] Helpers tested; MNIST flags exercise new code; tests green

---

## Phase 5 — Training QoL (schedules and learnables)
- [ ] Add τ schedule utilities (linear/cosine anneal) with unit tests for endpoints/monotonicity
- [ ] Ensure learnable α/β are exposed per-prototype with optional α+β=1 normalization (tests: positivity, normalization)
- [ ] Ensure learnable θ and match-threshold pathways are exposed and tested (positivity and effect on similarities)
- [ ] Wire flags into XOR/MNIST (e.g., `--learnable-alpha`, `--learnable-beta`, `--alpha-beta-normalize`, `--learnable-theta`, `--learnable-match-threshold`, `--tau-schedule`)
- [ ] Run tests: `pytest -q`
- [ ] Commit: "tnn: schedules and learnable params + example flags"

Acceptance:
- [ ] Learnables and schedules tested; examples toggle features

---

## Phase 6 — Heads and pooling variants
- [ ] Implement attention pooling over prototypes (tiny scorer MLP); tests for shapes/masking
- [ ] Add `TverskyClassifierHead` wrapper (projection + pooling) with clean API
- [ ] Replace direct pooling in MNIST with head wrapper (CLI switch `--prototype-pool {mean,max,attn}`)
- [ ] Run tests: `pytest -q`
- [ ] Commit: "tnn: attention pooling and classifier head"

Acceptance:
- [ ] Head tested; MNIST script switchable among pooling modes

---

## Phase 7 — Calibration and losses
- [ ] Add `calibrate_temperature(logits, labels)` utility (held-out set); tests: NLL decreases or stays same
- [ ] Add optional margin-style loss and/or label smoothing; smoke tests
- [ ] Wire optional calibration/loss into examples behind flags
- [ ] Run tests: `pytest -q`
- [ ] Commit: "tnn: calibration utility and optional loss variants"

Acceptance:
- [ ] Calibration tested; examples show toggle effects

---

## Phase 8 — Monitoring and logging
- [ ] Add logging helper to aggregate I/A/B, α/β histograms, prototype norms/overlaps per epoch
- [ ] Tests: helper returns expected keys/shapes; handles feature-sharing and single-stage
- [ ] Wire logging into XOR/MNIST with `--log-tnn-metrics` and optional TensorBoard
- [ ] Run tests: `pytest -q`
- [ ] Commit: "tnn: training-time metric logging"

Acceptance:
- [ ] Logs visible; unit tests pass

---

## Phase 9 — CI, packaging, and examples polish
- [ ] Add CI workflow (GitHub Actions): lint + tests (CPU), small datasets cache; no GPU required
- [ ] Optional: docs build check
- [ ] README quickstart: end-to-end snippet using installed package path; verify from clean checkout
- [ ] Run tests: `pytest -q`
- [ ] Commit: "tnn: CI and examples polish"

Acceptance:
- [ ] CI green; examples runnable; docs linked

---

## Final wiring checklist (no orphaned code)
- [ ] Every new function/class exported in `src/mlp_utils/layers/__init__.py` if public
- [ ] Each new function/class has at least one unit test
- [ ] README and/or examples reference new functionality (behind flags if optional)
- [ ] Examples default to simple fast settings; advanced features are opt-in
- [ ] Unified pass of `pytest -q` is green before merge
