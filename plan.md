ABOUTME: High-level blueprint to integrate and harden Tversky neural modules in `mlp-utils`.
ABOUTME: Multi-iteration plan with test-first, incremental steps that always wire into the repo.

## Tversky Neural Network Integration Blueprint

Source reference: [Tversky Neural Networks: Psychologically Plausible Deep Learning with Differentiable Tversky Similarity](https://arxiv.org/html/2506.11035v1)

### Assumptions and current repo state
- Core implementations already exist in `src/mlp_utils/layers/tversky.py`: `tversky_similarity`, `pairwise_tversky`, `tversky_attributions`, `TverskyProjection`, `TverskyFeatureSharing`, plus config classes.
- Public exports are wired in `src/mlp_utils/layers/__init__.py`.
- Tests exist in `tests/test_tversky.py` (unit, gradcheck, JIT, XOR training), and examples in `experiments/tversky_xor.py` and `experiments/train_mnist_tversky.py`.

Goal: polish, expand tests, add docs, add interpretability tooling, add recipes, and ensure robust CI—without big jumps. Every step is incremental, test-first, and ends with integration.

Guiding principles:
- Smallest reasonable changes, strong test coverage, no orphaned code, consistent naming, and clear integration at each step.
- Prefer data-driven initialization and interpretability consistent with the paper’s thrust.

### Repo integration targets
- you: are on a `feat/tnn-parity-tversky` git branch
- Code: `src/mlp_utils/layers/tversky.py`
- Exports: `src/mlp_utils/layers/__init__.py`
- Tests: `tests/test_tversky.py`
- Docs: `README.md` additions under Layers
- Optional example: `experiments/tversky_xor.py`
- python env: source /home/gdevos/structural_break/mlp-utils/.venv/bin/activate
- all packages are already installed


---

## Phase 0 — Baseline acceptance (verify and freeze scope)
1) Run all tests; capture baseline timings and coverage.
2) Verify public API surface documented in README and `__all__`/exports.

Acceptance: green tests, baseline metrics recorded.

---

## Phase 1 — API stabilization and docs
1) Formalize docstrings for all public functions/classes in `tversky.py` (parameters, shapes, invariants, examples).
2) Add a dedicated README section for Tversky modules: quickstart, tips, gotchas (α/β asymmetry, θ stability, transforms, smoothing τ).
3) Add a minimal API reference table to README and cross-link example scripts.

Acceptance: docs built/readable locally; links valid; no code behavior changes; tests stay green.

---

## Phase 2 — Testing expansion and numerical robustness
1) Property-based tests (Hypothesis) for monotonicity, bounds, asymmetry, broadcasting shapes.
2) Stress tests: extreme magnitudes, sparse patterns, degenerate cases (all zeros, disjoint support).
3) Precision tests across dtypes, larger dims, and product reduction underflow guards.
4) TorchCompile/AMP smoke tests for `TverskyProjection` and `TverskyFeatureSharing`.

Acceptance: new tests in `tests/test_tversky.py` (or split files) pass; coverage improves.

---

## Phase 3 — Interpretability and visualization
1) Add a light-weight attribution helper that wraps `tversky_attributions` into a dict with named components and sums.
2) Add simple plotting utilities (matplotlib) to visualize per-feature I/A/B bars and heatmaps.
3) Add a small example notebook (or script) showcasing attribution on XOR and MNIST prototypes.

Acceptance: example runs produce figures; unit tests validate helper’s invariants (sum consistency, shapes).

---

## Phase 4 — Initialization, seeding, and diversity regularization
1) Add dataset-driven prototype seeding helpers (class-conditional mean/medoid) and wire into MNIST example.
2) Add optional prototype diversity regularizer (pairwise separation) behind a small utility function.
3) Extend examples to toggle seeding + diversity and show metrics impact.

Acceptance: helpers unit-tested; example flags exercise new code; existing tests remain green.

---

## Phase 5 — Training quality-of-life (QoL)
1) Smoothing τ schedule utilities (e.g., cosine/linear anneal); tests ensure schedule endpoints.
2) α/β parameterization: per-prototype learnable with optional α+β normalization; tests verify positivity and normalization.
3) Learnable θ and match-threshold pathways (already partially present) with tests for positivity and effect.

Acceptance: unit tests for learnables and schedules; integration in XOR/MNIST scripts via flags.

---

## Phase 6 — Heads and pooling variants
1) Prototype pooling: keep mean/max and add attention pooling with a tiny scorer MLP; tests for shape and masking.
2) Classification head wrappers that combine projection + pooling with clean API; wire into MNIST demo.

Acceptance: head passes tests; demo script switchable between pooling modes.

---

## Phase 7 — Calibration and losses
1) Temperature and bias calibration utilities; tests for calibration monotonicity and NLL impact.
2) Margin-style losses or label smoothing variants; smoke tests.

Acceptance: calibration helpers unit-tested; demos show toggle effects.

---

## Phase 8 — Monitoring and logging
1) Simple hooks to log I/A/B aggregates, α/β histograms, prototype norms/overlaps during training.
2) Add logging to example scripts (prints and optional TensorBoard).

Acceptance: logs visible; unit tests cover hook shape and expected keys.

---

## Phase 9 — CI, packaging, and examples polish
1) Add CI workflow: lint, tests (CPU), minimal datasets cache.
2) Add optional docs build check.
3) Finalize examples; README quickstart end-to-end snippet using installed package path.

Acceptance: CI green; examples runnable; docs linked.

---

# Iterative breakdown into small, incremental chunks

Each chunk yields a shippable state and wires into tests/examples. Chunks are ordered; each depends only on prior completed chunks.

### Chunk 0: Baseline verify (Phase 0)
- Step 0.1: Run `pytest -q`; record pass/fail and durations.
- Step 0.2: Record coverage (e.g., `pytest --cov=mlp_utils`).
- Step 0.3: Create branch `feat/tversky-hardening`; commit baseline.

### Chunk 1: API docs pass (Phase 1)
- Step 1.1: Add/extend docstrings in `tversky.py` (no logic change).
- Step 1.2: README: add Tversky section with quickstart and tips; link to examples.
- Step 1.3: `pytest -q` to ensure no regressions.

### Chunk 2: Robustness tests (Phase 2)
- Step 2.1: Add property-based tests for similarity bounds and asymmetry.
- Step 2.2: Add stress cases: large/small scales, disjoint supports.
- Step 2.3: Add dtype/shape broadcast matrix tests.
- Step 2.4: Add torch.compile + AMP smoke tests.
- Step 2.5: `pytest -q`; ensure coverage increase.

### Chunk 3: Interpretability helpers (Phase 3)
- Step 3.1: Add `tversky_explain.py` with `explain_similarity(input, proto)` returning named components.
- Step 3.2: Add plotting helper in `experiments/utils_plot.py` (matplotlib), used only in examples.
- Step 3.3: New tests for `explain_similarity` sums/shape invariants.
- Step 3.4: Wire plotting in XOR script behind a flag; `pytest -q`.

### Chunk 4: Seeding + diversity (Phase 4)
- Step 4.1: Add `seed_prototypes_from_loader` utility (uses backbone features or stage1 memberships).
- Step 4.2: Add `prototype_diversity_loss(weights, margin)`.
- Step 4.3: Unit tests for seeding (shape, deterministic with fixed seed) and diversity (penalizes overlaps).
- Step 4.4: Wire into MNIST script via CLI flags; `pytest -q`.

### Chunk 5: QoL: schedules and learnables (Phase 5)
- Step 5.1: Add `anneal_tau(step)` schedulers; unit tests for endpoints/monotonicity.
- Step 5.2: Ensure learnable α/β/θ/match-threshold pathways exposed via layer flags (already present) and tested for positivity/normalization.
- Step 5.3: Wire flags into XOR/MNIST scripts; `pytest -q`.

### Chunk 6: Pooling and heads (Phase 6)
- Step 6.1: Add attention pooling module for prototypes; unit tests for shapes.
- Step 6.2: Add a small `TverskyClassifierHead` wrapper combining projection + pooling.
- Step 6.3: Use head wrapper in MNIST script; `pytest -q`.

### Chunk 7: Calibration and losses (Phase 7)
- Step 7.1: `calibrate_temperature` utility; unit tests (NLL decreases on validation split after calibration step).
- Step 7.2: Optional margin loss; smoke tests; wire optional use in examples.
- Step 7.3: `pytest -q`.

### Chunk 8: Monitoring/logging (Phase 8)
- Step 8.1: Add a small logging helper that computes/returns I/A/B aggregates and α/β stats each epoch.
- Step 8.2: Wire logging into XOR and MNIST scripts; add CLI toggle.
- Step 8.3: Unit tests for helper output keys/types; `pytest -q`.

### Chunk 9: CI and packaging polish (Phase 9)
- Step 9.1: Add GitHub Actions workflow: lint + tests (CPU, no data download failures).
- Step 9.2: Add minimal docs check (optional).
- Step 9.3: README quickstart review; ensure examples run from clean checkout; `pytest -q`.

---

# Micro-steps (another round of refinement)

For each chunk, the following micro-steps ensure right-sized work units with immediate integration:

1) Write (or extend) one test that captures the desired behavior.
2) Implement the minimal code to satisfy that test.
3) Export new public symbols in `src/mlp_utils/layers/__init__.py` if needed.
4) Update README or example scripts to reference the new code behind a flag.
5) Run `pytest -q`; fix any regressions; keep commits small and isolated.

Examples per chunk:
- Chunk 2 → Micro: add one property test (bounds), run; then another (asymmetry), run; then broadcast test, run; then compile smoke, run.
- Chunk 3 → Micro: add `explain_similarity` with tests; only after green, add plotting helper used by XOR example (guard with `--plot`).
- Chunk 4 → Micro: add seeding util + tests; wire in MNIST; then add diversity loss + tests; wire flag; keep each change separately shippable.
- Chunk 5 → Micro: add tau scheduler + tests; expose α/β normalization flag in `TverskyProjection` (tests already present can be extended); wire flags to examples.
- Chunk 6 → Micro: implement attention pooling; test; integrate into a small head wrapper; replace direct pooling in MNIST via CLI switch.
- Chunk 7 → Micro: implement temperature calibration helper; unit test using a frozen logits tensor; then add margin loss; smoke test.
- Chunk 8 → Micro: logging helper returns dict; unit test for keys; wire prints in examples; optional TensorBoard logging later.
- Chunk 9 → Micro: CI yaml for tests; push; ensure pass; then add coverage gate.

---

## Final wiring checklist (no orphaned code)
- Public exports updated when new utilities are added.
- Every new function/class has at least one unit test and is referenced in either README or an example script.
- Example scripts have flags for all optional behaviors; defaults remain simple and fast.
- CI covers unit tests; examples are exercised manually and documented.

---


Each stretch item should follow the same micro-step pattern: add tests/utilities first, wire to a minimal example, document, and keep changes small.
