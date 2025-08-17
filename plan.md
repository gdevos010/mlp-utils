## Blueprint: Add Tversky Neural Network Components to mlp_utils

Reference: Tversky Neural Networks: Psychologically Plausible Deep Learning with Differentiable Tversky Similarity [arXiv:2506.11035](https://arxiv.org/abs/2506.11035)

### Goals
- **Provide a differentiable Tversky similarity** API suitable for tensors, batched inputs, and prototype banks.
- **Introduce a `TverskyProjection` layer** as a drop-in alternative to linear projections for MLP-style blocks.
- **Integrate safely** into the existing package structure (`src/mlp_utils/layers`) with complete unit tests and docs.
- **Keep incremental scope**: start minimal and correct, then add features (learnable α/β, adapters, examples, benchmarks).

### Non-goals (for now)
- Custom CUDA kernels or specialized acceleration.
- Full reproduction of the paper’s end-to-end pipelines (NABirds, PTB). We'll provide hooks/examples only.

### Repo integration targets
- you: are on a tnn git branch
- Code: `src/mlp_utils/layers/tversky.py`
- Exports: `src/mlp_utils/layers/__init__.py`
- Tests: `tests/test_tversky.py`
- Docs: `README.md` additions under Layers
- Optional example: `experiments/tversky_xor.py`
- python env: source /home/gdevos/structural_break/mlp-utils/.venv/bin/activate
- all packages are already installed
---

## Architecture and API design

### Differentiable Tversky similarity (tensor-friendly)
- Function: `tversky_similarity(input, prototype, alpha=0.5, beta=0.5, eps=1e-6, input_transform=None, nonnegative=True, smoothing_tau=None)`
  - Inputs: `input` and `prototype` with shape `[..., D]` (broadcastable on leading dims).
  - Optional input transform applied before set-ops proxy:
    - `input_transform=None` (default, no transform), or one of `"relu"`, `"clamp01"`, `"sigmoid"`, or a callable. If provided, this supersedes `nonnegative`.
    - If only `nonnegative=True`, clamp to nonnegative via ReLU.
  - Set-ops proxies (hard by default):
    - Intersection: `sum(min(input, prototype), dim=-1)`
    - A\B: `sum(relu(input - prototype), dim=-1)`
    - B\A: `sum(relu(prototype - input), dim=-1)`
  - Optional smoothing via `smoothing_tau` (>0):
    - If set, use temperature-smoothed proxies: elementwise soft-min for intersection (e.g., `softmin(a,b;tau)` via log-sum-exp) and `softplus(· / tau)` for differences. As `tau → 0`, these approach the hard ops.
  - Similarity: `(I + eps) / (I + alpha * A_only + beta * B_only + eps)`

- Batched and multi-prototype variant: `pairwise_tversky(input, prototypes, ...)`
  - `input`: `[B, D]`, `prototypes`: `[K, D]` → returns `[B, K]` similarities.

### Layer: `TverskyProjection`
- Purpose: compute similarities to a bank of learned prototypes as a projection.
- Signature:
  - `TverskyProjection(input_dim, output_dim, alpha=0.5, beta=0.5, eps=1e-6, bias=False, input_transform=None, nonnegative=True, smoothing_tau=None, learnable_alpha=False, learnable_beta=False, alpha_beta_normalize=False, temperature=None)`
- Parameters:
  - `weight`: `[output_dim, input_dim]` — each row is a prototype.
  - `bias`: optional `[output_dim]` (default disabled to preserve [0,1] similarity semantics).
  - Optional `alpha`, `beta` as buffers or `nn.Parameter` if learnable.
    - If `learnable_*` is True, internally parameterize with unconstrained variables and map via `softplus` to ensure nonnegativity.
    - If `alpha_beta_normalize=True`, renormalize learned `alpha` and `beta` to sum to 1 for additional stability.
  - Optional `input_transform` and `smoothing_tau` forwarded to similarity computation.
  - Optional `temperature` (scalar float) to scale outputs if used downstream.
- Forward:
  - Given `x: [*, input_dim]`, return `y: [*, output_dim]` via `pairwise_tversky(x, weight)` plus optional `bias` and `temperature` scaling.
- Notes:
  - Range semantics: without `bias` and with default temperature, outputs lie in `(0, 1]` under nonnegative/bounded inputs. Enabling `bias` or non-unit `temperature` produces an affine/scale-transformed similarity.
  - Deterministic, no randomness.
  - TorchScript/`torch.compile` friendly operations only.

### Optional adapter: `TverskyAdapter`
- Lightweight classifier/regressor head that wraps a frozen backbone output with `TverskyProjection`.
- Out of critical path; add after core layer stabilizes.

### Optional visualization utility
- Provide an attribution helper to expose interpretability signals highlighted by the paper:
  - Function: `tversky_attributions(input, prototype, alpha, beta, input_transform=None, smoothing_tau=None)`
    - Returns per-feature contributions for intersection (`I_components = min(·)` or soft-min) and distinctive parts (`A_only_components`, `B_only_components`).
    - Enables visualizing which input features support or detract from prototype similarity.

### Numerics and stability
- Ensure nonnegativity or boundedness before set proxies via `input_transform` or `nonnegative`.
- Optional smoothing with small `smoothing_tau` improves optimization smoothness; `tau → 0` approximates hard ops.
- Learnable `alpha`/`beta` are constrained positive via `softplus`; optional normalization to sum to 1.
- Small `eps` in numerator/denominator.
- Unit tests cover gradient existence (autograd) and boundedness (0..1) when `bias=False`.

---

## Testing strategy
- `tests/test_tversky.py`
  - Unit: `tversky_similarity`
    - Identity/self-similarity: `S(x,x) ≈ 1` for random `x ≥ 0`; include edge case `x = 0` giving exactly `1` under current formula.
    - Asymmetry: with `alpha ≠ beta`, verify `S(x,y) ≠ S(y,x)` for a constructed non-symmetric pair.
    - Monotonicity: for fixed `y`, increasing shared mass in `x` raises `S(x,y)`; create pairs with controlled overlap to check ordering.
    - Range/boundedness: for nonnegative inputs and default settings (`bias=False`), `S(x,y) ∈ (0,1]` including with `smoothing_tau` enabled.
    - Input transforms: verify that `input_transform="relu"` matches pre-ReLUed inputs; check `"clamp01"` keeps values in `[0,1]`; confirm callable support.
    - Smoothing extremes: compare hard ops to `smoothing_tau=1e-3` (close within tolerance); verify numerical stability at larger `tau` values.
    - Numerical stability: no NaN/Inf for zero vectors, sparse/disjoint vectors, and large-magnitude vectors; `eps` prevents division-by-zero.
    - Gradients: autograd produces finite gradients; use `torch.autograd.gradcheck` in double precision on small random positive inputs (avoid exact ties to prevent cusp issues).
    - Dtype/device: CPU-only, test `float32` and `float64`.
    - Error handling: invalid `smoothing_tau ≤ 0` or unknown `input_transform` raises a clear `ValueError`.
  - Unit: `pairwise_tversky`
    - Shapes: `[B,D]×[K,D]→[B,K]` and broadcasting with leading dims: `[B,T,D]×[K,D]→[B,T,K]`.
    - Parity with loop: matches Python for-loop over prototypes within tolerance.
    - Gradients: nonzero grads w.r.t both inputs and prototypes; verify finite values.
    - Dtype/device: CPU `float32/float64` parity.
  - Module: `TverskyProjection`
    - Parameterization: `weight` shape `[out,in]`, optional `bias` shape `[out]`; default `bias=False`.
    - Learnable α/β: when enabled, internal params map via softplus to `α,β > 0`; with `alpha_beta_normalize=True`, `α+β ≈ 1`.
    - Forward: output shape preserves batch dims; with default settings and nonnegative/bounded inputs, outputs lie in `(0,1]`.
    - Temperature: enabling `temperature=t` scales outputs; verify proportionality numerically.
    - Training step: simple synthetic task (e.g., mapping few prototypes) decreases loss over a few optimizer steps with fixed seed.
    - Serialization: `state_dict` save/load roundtrips parameters and preserves outputs.
    - Scripting/compilation: guard-tested `torch.jit.script` or `torch.jit.trace` forward pass succeeds; optionally test `torch.compile` if available.
  - Utility: `tversky_attributions`
    - Shapes and consistency: per-feature components sum to aggregate intersection and distinctive parts used by `tversky_similarity`.
    - Nonnegativity: contributions for intersection/distinctive parts are nonnegative under nonnegative inputs.
  - Integration
    - Tiny block: `nn.Sequential(TverskyProjection, nn.Softmax)` forward/backward on random data works; gradients flow to `weight`.
    - XOR demo: 2D XOR with `output_dim=2` achieves high accuracy (>90%) in a few steps; deterministic via fixed seeds and small batch.
  - Test hygiene
    - Parametrization via `pytest.mark.parametrize` for dtypes, transforms, and smoothing `tau` values.
    - Clear tolerances (`rtol`, `atol`) documented; keep runtime under CI budget by marking heavier tests (e.g., gradcheck, XOR train) as `slow` and skipping by default in CI.

---

## Iteration plan — Round 1 (coarse milestones)

1) Scaffolding
- Add `tversky.py` with module docstring and empty class/function stubs.
- Export from `layers/__init__.py`.
- Add `tests/test_tversky.py` with skipped tests (`pytest.mark.skip`) placeholders.
- CI passes (no runtime code yet, only skips).

2) Core similarity function
- Implement `tversky_similarity` for single pair and validate numerics.
- Add unit tests for identity, range, asymmetry.

3) Batched/multi-prototype API
- Implement `pairwise_tversky` with broadcasting and vectorization.
- Add tests for shapes, batching, and equivalence to looping.

4) `TverskyProjection` layer
- Implement learnable prototypes and forward using `pairwise_tversky`.
- Add tests for parameter shapes, forward pass, and gradient flow.

5) Minimal integration
- Add a tiny demo block (local to tests) using `TverskyProjection` to ensure compatibility with existing layers.
- Ensure test suite green.

6) Docs and README
- Update `README.md` Layers section with `TverskyProjection` usage snippet and brief explanation with link to the paper.

7) Example script (optional, guarded)
- Add `experiments/tversky_xor.py` (small toy) with quick train loop.
- Do not reference in README as required step; keep as optional example.

8) Extended features (optional)
- Add `learnable_alpha`, `learnable_beta`, `temperature` with unit tests.
- Add `TverskyAdapter` with a minimal test for shape and training.

---

## Iteration plan — Round 2 (break each milestone into smaller, safe chunks)

1) Scaffolding
- 1.1 Create `src/mlp_utils/layers/tversky.py` with module docstring, `__all__`.
- 1.2 Add no-op stubs: `tversky_similarity`, `pairwise_tversky`, `TverskyProjection` raising `NotImplementedError`.
- 1.3 Export symbols in `src/mlp_utils/layers/__init__.py` (import and add to `__all__`).
- 1.4 Create `tests/test_tversky.py` with `pytest.skip` on top to avoid failures.

2) Core similarity
- 2.1 Implement `tversky_similarity` for shape `[..., D]` inputs.
- 2.2 Add tests: identity (x==y), range in (0,1], basic asymmetry when `alpha!=beta`.
- 2.3 Remove `pytest.skip` guard for these tests.

3) Batched/multi-prototype
- 3.1 Implement `pairwise_tversky(x:[B,D], P:[K,D]) -> [B,K]` (and support `[*,D]`/`[K,D]`).
- 3.2 Add tests for shapes, broadcasting, and parity with Python loop over K.

4) Projection layer
- 4.1 Implement `TverskyProjection.__init__` (params: weight, bias; store alpha/beta/eps flags).
- 4.2 Implement `forward(x)` using `pairwise_tversky` and bias.
- 4.3 Tests: module builds, forward returns `[*, output_dim]`, gradients w.r.t. `weight` exist.

5) Minimal integration
- 5.1 Add a test-local mini block (e.g., `nn.Sequential(TverskyProjection, nn.Softmax)`) and run forward/backward.
- 5.2 Ensure CPU-only path passes in CI.

6) Docs
- 6.1 Add `README.md` section: short synopsis, API, quick code snippet.
- 6.2 Link to [arXiv:2506.11035](https://arxiv.org/abs/2506.11035).

7) Example (optional)
- 7.1 Add `experiments/tversky_xor.py`.
- 7.2 Ensure script runs in < 5s on CPU (small steps/epochs) and is not imported elsewhere.

8) Extended features (optional)
- 8.1 Add `learnable_alpha`, `learnable_beta` toggles; update tests to cover parameter registration and training.
- 8.2 Add optional `temperature` scaling; test numeric effect.
- 8.3 Add `TverskyAdapter` (simple wrapper); test forward shape and training on synthetic data.

---

## Iteration plan — Round 3 (micro-steps with acceptance criteria)

Step A: File and export skeletons
- A.1 Create `tversky.py` with module docstring and `__all__`.
  - Acceptance: file exists, imports without side effects.
- A.2 Update `layers/__init__.py` to import and expose symbols.
  - Acceptance: `from mlp_utils.layers import TverskyProjection` succeeds.
- A.3 Add `tests/test_tversky.py` with top-level `pytest.skip("scaffold")` and a placeholder class.
  - Acceptance: test discovery passes without failures.

Step B: Implement `tversky_similarity`
- B.1 Implement function with `relu` and `minimum` proxy ops; add `eps` and `nonnegative` flag.
  - Acceptance: unit tests for identity and range pass.
- B.2 Add asymmetry test by setting `alpha != beta`.
  - Acceptance: `S(x,y) != S(y,x)` for a non-symmetric pair.

Step C: Implement `pairwise_tversky`
- C.1 Vectorized computation over prototypes; avoid Python loops.
  - Acceptance: matches looped version within tolerance, shapes correct.
- C.2 Add dtype/device tests (cpu, float32/float64) as feasible in CI.
  - Acceptance: tests green.

Step D: Implement `TverskyProjection`
- D.1 Add parameters: `weight` `[out,in]`, optional `bias`.
  - Acceptance: `.parameters()` lists `weight` (and `bias` when enabled).
- D.2 Implement forward using `pairwise_tversky` respecting batch dims.
  - Acceptance: gradient flows to `weight` under a simple loss.

Step E: Integration sanity
- E.1 Write a test-only tiny model combining `TverskyProjection` with `nn.Linear`/`nn.Softmax` to ensure composability.
  - Acceptance: forward/backward pass without error; loss decreases over a few steps on toy data.

Step F: Docs and usage
- F.1 Add README snippet demonstrating construction and forward pass.
  - Acceptance: snippet imports succeed locally (doctest-like minimal check in CI optional).

Step G: Example script (optional)
- G.1 Implement `experiments/tversky_xor.py` (2D XOR, tiny MLP with `TverskyProjection`).
  - Acceptance: script runs quickly and prints final loss/accuracy.

Step H: Extended features (optional)
- H.1 Add `learnable_alpha`, `learnable_beta` `nn.Parameter` toggles.
  - Acceptance: parameters present and updated by optimizer in a minimal training test.
- H.2 Add `temperature` scaling (post-similarity multiplier).
  - Acceptance: numeric effect covered by unit test.
- H.3 `TverskyAdapter` module as a simple head.
  - Acceptance: forward shape + a tiny training step on synthetic labels.

---

## Best practices and guardrails
- Keep functions pure and side-effect free; modules deterministic.
- Maintain full type hints and clear docstrings with equations and references.
- Respect existing formatting and module organization.
- Tests first where feasible, or add immediately after code with small, verifiable assertions.
- Avoid premature optimization; prefer readability and correctness.

---

## Final wiring (no orphaned code)
- Ensure `tversky.py` is fully exported via `layers/__init__.py` and importable from `mlp_utils.layers`.
- `tests/test_tversky.py` covers all public APIs (`tversky_similarity`, `pairwise_tversky`, `TverskyProjection`).
- `README.md` includes a concise usage example and paper citation link.
- Optional `experiments/tversky_xor.py` is runnable but not imported by the package.
- Entire test suite green on CPU-only CI.
