## Tversky NN Integration TODO Checklist

Reference: Tversky Neural Networks: Psychologically Plausible Deep Learning with Differentiable Tversky Similarity [arXiv:2506.11035](https://arxiv.org/abs/2506.11035)

Use this as a living checklist while implementing `tversky.py`, tests, docs, and examples.

---

### 1) Project scaffolding
- [x] Create `src/mlp_utils/layers/tversky.py` with module docstring, `__all__`, and imports only.
- [x] Add stubs for `tversky_similarity`, `pairwise_tversky`, `TverskyProjection` that raise `NotImplementedError`.
- [x] Update `src/mlp_utils/layers/__init__.py` to import and expose symbols.
- [x] Create `tests/test_tversky.py` with `pytest.skip("scaffold")` to avoid failures at start.
- [x] Sanity: `pytest -q` runs with 0 failures (skipped ok).

Acceptance:
- [ ] `from mlp_utils.layers import TverskyProjection` imports without side effects.

---

### 2) Implement `tversky_similarity`
- [x] Signature: `tversky_similarity(input, prototype, alpha=0.5, beta=0.5, eps=1e-6, input_transform=None, nonnegative=True, smoothing_tau=None)`.
- [x] Implement input transform: support `None`, `"relu"`, `"clamp01"`, `"sigmoid"`, or callable; if set, it supersedes `nonnegative`.
- [x] Hard set-proxy ops by default using `torch.minimum` and `F.relu`.
- [x] Optional smoothing when `smoothing_tau>0`: soft-min for intersection (log-sum-exp) and `softplus(·/tau)` for differences.
- [x] Validate and error on invalid `smoothing_tau ≤ 0`.
- [x] Numerical stability: add `eps` to numerator and denominator; avoid NaN/Inf.
- [x] Full type hints and a docstring including the similarity formula and semantics.

Tests (unit):
- [x] Identity/self-similarity: `S(x,x) ≈ 1` for random `x ≥ 0`; `x=0` edge-case → `1`.
- [x] Asymmetry: `alpha ≠ beta` yields `S(x,y) ≠ S(y,x)` on a non-symmetric pair.
- [ ] Monotonicity: increased overlap raises similarity; verify ordering on controlled vectors.
- [x] Range/boundedness: for nonnegative inputs and defaults, `S ∈ (0,1]` (also with smoothing).
- [x] Input transforms: `relu` parity, `clamp01` range enforcement, `sigmoid` transform; callable support.
- [ ] Smoothing extremes: `tau→0` approximates hard ops within tolerance; stable for moderate `tau`.
- [ ] Numerical stability: zero vectors, disjoint vectors, large values → finite outputs.
- [ ] Gradients: `autograd.gradcheck` (double precision, small random positive inputs) passes.
- [x] Dtype/device: CPU `float32`/`float64` equivalence within tolerance.
- [x] Error handling: invalid `smoothing_tau` and unknown `input_transform` raise `ValueError`.

---

### 3) Implement `pairwise_tversky`
- [ ] Vectorized computation over prototypes: `[B,D]×[K,D]→[B,K]`.
- [ ] Broadcasting support for leading dims: `[*,D]×[K,D]→[*,K]`.
- [ ] Forward parameters for `input_transform`, `nonnegative`, `smoothing_tau`, `alpha`, `beta`, `eps`.

Tests (unit):
- [ ] Shapes for `[B,D]×[K,D]` and `[B,T,D]×[K,D]` produce `[B,K]` and `[B,T,K]`.
- [ ] Parity with Python loop over K within numeric tolerance.
- [ ] Gradients nonzero and finite w.r.t. inputs and prototypes.
- [ ] Dtype/device: CPU `float32/float64` parity.

---

### 4) Implement `TverskyProjection` layer
- [ ] Constructor: `TverskyProjection(input_dim, output_dim, alpha=0.5, beta=0.5, eps=1e-6, bias=False, input_transform=None, nonnegative=True, smoothing_tau=None, learnable_alpha=False, learnable_beta=False, alpha_beta_normalize=False, temperature=None)`.
- [ ] Parameters: `weight` shape `[out,in]`; optional `bias` shape `[out]` (default off to preserve `(0,1]`).
- [ ] α/β handling: buffers by default; if learnable, register unconstrained params mapped via softplus to positive α/β.
- [ ] Optional `alpha_beta_normalize=True`: renormalize α and β so `α+β=1`.
- [ ] Forward: compute `pairwise_tversky(x, weight, ...)`, add optional bias, apply optional temperature scaling; preserve leading batch dims.
- [ ] Register buffers and dtypes/devices properly; ensure `.to(device)` moves α/β.
- [ ] Range semantics note in docstring: bias/temperature break strict `(0,1]` interpretation.

Tests (module):
- [ ] Parameter shapes and registration (`weight`, `bias` when enabled).
- [ ] Forward shape preserved; values in `(0,1]` for defaults and nonnegative inputs.
- [ ] Learnable α/β positivity (softplus) and optional normalization (`≈1` sum).
- [ ] Training step reduces a simple loss; gradient flows to `weight` and learnable α/β if enabled.
- [ ] `state_dict` save/load roundtrip yields identical outputs.
- [ ] JIT/compile: `torch.jit.script` or `trace` forward works; optionally `torch.compile` if available.

---

### 5) Implement `tversky_attributions` utility
- [ ] Function returns per-feature contributions for intersection and distinctive parts consistent with `tversky_similarity`.
- [ ] Works with both hard and smoothed proxies; respects `input_transform`.
- [ ] Clear docstring explaining outputs and intended visualization.

Tests (utility):
- [ ] Shapes match input feature dimension.
- [ ] Nonnegativity of contributions under nonnegative inputs.
- [ ] Sum of components equals aggregate terms used in similarity calculation (within tolerance for smoothing).

---

### 6) Integration and demos
- [ ] Mini model: `nn.Sequential(TverskyProjection, nn.Softmax)` runs forward/backward on random tensors; gradients propagate.
- [ ] XOR example: `experiments/tversky_xor.py` with 2D inputs and `output_dim=2`, fixed seeds, quick CPU run (<5s), prints loss/accuracy.

Tests (integration):
- [ ] Tiny training loop reduces loss and achieves high accuracy (>90%) deterministically.

---

### 7) Documentation
- [ ] Update `README.md` Layers section with `TverskyProjection` synopsis, construction snippet, and example forward pass.
- [ ] Add notes on similarity range semantics, bias/temperature effects, and interpretability.
- [ ] Link to paper: `[arXiv:2506.11035](https://arxiv.org/abs/2506.11035)`.
- [ ] Ensure docstrings include parameter meaning and equations where helpful.

Acceptance:
- [ ] README snippet imports and runs in a local Python shell.

---

### 8) Test hygiene and CI
- [ ] Parametrize tests over dtypes (`float32`, `float64`), transforms, and select `smoothing_tau` values.
- [ ] Configure tolerances (`rtol`, `atol`) and avoid flaky comparisons.
- [ ] Mark heavier tests (gradcheck, XOR training) as `@pytest.mark.slow` and skip in default CI.
- [ ] Ensure `pytest -q` is green on CPU-only environment.

---

### 9) Final wiring
- [ ] Export all public APIs from `mlp_utils.layers` and verify they are importable.
- [ ] Confirm no orphaned code, no unused symbols, and clear `__all__` in `tversky.py`.
- [ ] Code readability: type hints, descriptive names, concise comments/docstrings.
- [ ] Run the full test suite locally; ensure zero failures.

Done when:
- [ ] All checkboxes above are completed, tests are green, and README/docs accurately reflect the implemented APIs.
