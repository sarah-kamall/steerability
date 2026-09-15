"""Pure tensor operations for prior-Fisher drift-budget updates."""

import math

import torch


def drift_budget_update(
    gradients: list[torch.Tensor],
    fisher: list[torch.Tensor],
    *,
    kappa: float,
    damping: float,
) -> tuple[list[torch.Tensor], dict[str, float]]:
    """Compute the diagonal prior-Fisher drift-budget update.

    The returned update minimizes the target loss's first-order approximation under
    `0.5 * delta.T @ (F + damping * I) @ delta <= kappa`. Inputs are not mutated.

    Args:
        gradients: Target-loss gradients, one tensor per selected parameter.
        fisher: Diagonal Fisher estimates matching `gradients`.
        kappa: Positive quadratic drift budget.
        damping: Positive diagonal damping value.

    Returns:
        The parameter-shaped updates and scalar diagnostics.

    Raises:
        ValueError: If the tensor lists or their shapes differ.
        FloatingPointError: If the preconditioned quadratic form is not finite.
    """
    if len(gradients) != len(fisher):
        raise ValueError("gradients and fisher must contain the same number of tensors.")
    if not gradients:
        raise ValueError("gradients and fisher must not be empty.")
    if any(gradient.shape != diagonal.shape for gradient, diagonal in zip(gradients, fisher)):
        raise ValueError("Every Fisher tensor must match its gradient tensor's shape.")
    for index, gradient in enumerate(gradients):
        if not torch.isfinite(gradient).all():
            raise FloatingPointError(f"Target gradient tensor {index} contains non-finite values.")
    for index, diagonal in enumerate(fisher):
        if not torch.isfinite(diagonal).all():
            raise FloatingPointError(f"Fisher tensor {index} contains non-finite values.")

    directions = [gradient.float() / (diagonal.float() + damping) for gradient, diagonal in zip(gradients, fisher)]
    q = sum(
        float((gradient.double() * direction.double()).sum().detach().cpu())
        for gradient, direction in zip(gradients, directions)
    )
    gradient_norm_sq = sum(float(gradient.double().square().sum().detach().cpu()) for gradient in gradients)

    if not math.isfinite(q):
        raise FloatingPointError("The Fisher-preconditioned quadratic form is not finite.")
    if q <= 1e-12:
        updates = [torch.zeros_like(gradient, dtype=torch.float32) for gradient in gradients]
        return updates, {
            "q": q,
            "gradient_norm": math.sqrt(max(gradient_norm_sq, 0.0)),
            "step_norm": 0.0,
            "predicted_drift": 0.0,
            "scale": 0.0,
        }

    scale = math.sqrt(2.0 * kappa / q)
    updates = [-scale * direction for direction in directions]
    step_norm_sq = sum(float(update.square().sum().detach().cpu()) for update in updates)
    predicted_drift = 0.5 * sum(
        float(((diagonal.float() + damping) * update.square()).sum().detach().cpu())
        for diagonal, update in zip(fisher, updates)
    )
    return updates, {
        "q": q,
        "gradient_norm": math.sqrt(max(gradient_norm_sq, 0.0)),
        "step_norm": math.sqrt(max(step_norm_sq, 0.0)),
        "predicted_drift": predicted_drift,
        "scale": scale,
    }
