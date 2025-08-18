# ABOUTME: Small helper that wraps tversky_attributions and returns named components.
# ABOUTME: Provides sum checks and convenience structure for interpretability tooling.

"""Interpretability helper for Tversky similarity.

Provides a thin wrapper over `tversky_attributions` that returns a dictionary
with named components and validates that the reconstructed similarity matches
`tversky_similarity`.
"""

from __future__ import annotations

import torch

from .tversky import tversky_attributions, tversky_similarity


def explain_similarity(  # noqa: PLR0913
    input: torch.Tensor,  # noqa: A002
    prototype: torch.Tensor,
    *,
    alpha: float = 0.5,
    beta: float = 0.5,
    theta: float = 1e-7,
    input_transform: str | None = None,
    nonnegative: bool = True,
    smoothing_tau: float | None = None,
    difference_reduction: str = "subtractmatch",
    match_threshold: float = 0.0,
) -> dict[str, torch.Tensor]:  # noqa: PLR0913
    """Return per-feature components, their sums, and the resulting similarity.

    The returned dict contains:
    - I_components, A_components, B_components: featurewise tensors whose sums
      along the last dimension reproduce the aggregate terms.
    - I, A, B: sums of the above components.
    - similarity: reconstructed similarity value.
    """
    i_components, a_components, b_components = tversky_attributions(
        input,
        prototype,
        input_transform=input_transform,
        nonnegative=nonnegative,
        smoothing_tau=smoothing_tau,
        alpha=alpha,
        beta=beta,
        difference_reduction=difference_reduction,  # type: ignore[arg-type]
        match_threshold=match_threshold,
    )
    i_sum = i_components.sum(dim=-1)
    a_sum = a_components.sum(dim=-1)
    b_sum = b_components.sum(dim=-1)
    similarity = (i_sum + theta) / (i_sum + alpha * a_sum + beta * b_sum + theta)

    # Consistency w.r.t reference implementation
    similarity_ref = tversky_similarity(
        input,
        prototype,
        alpha=alpha,
        beta=beta,
        theta=theta,
        input_transform=input_transform,
        nonnegative=nonnegative,
        smoothing_tau=smoothing_tau,
        difference_reduction=difference_reduction,  # type: ignore[arg-type]
        match_threshold=match_threshold,
    )
    if not torch.allclose(similarity, similarity_ref, rtol=1e-5, atol=1e-6):
        raise RuntimeError("explain_similarity: mismatch vs tversky_similarity")

    return {
        "I_components": i_components,
        "A_components": a_components,
        "B_components": b_components,
        "I": i_sum,
        "A": a_sum,
        "B": b_sum,
        "similarity": similarity,
    }
