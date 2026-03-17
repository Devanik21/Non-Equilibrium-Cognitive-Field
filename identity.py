"""
identity.py — Identity Curvature Functional H[L] and Rollback.
===============================================================
H[L] = (1/N) Σ_i ||L_i - L_i^0||²  +  κ · Var(L_i)

Two terms:
  1. Drift penalty   — rules shouldn't wander far from initialization
  2. Variance penalty — rules shouldn't all collapse to the same value

Gradient ∂H/∂L_i pulls rules back toward init while preserving diversity.
Rollback: if δH > threshold in one step, revert L with gradient correction.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class IdentityState:
    H_current:     float = 0.0
    H_previous:    float = 0.0
    rollback_count: int  = 0
    H_history:     list  = field(default_factory=list)


class IdentityCurvature:
    """
    Computes H[L] and enforces rollback when identity curvature spikes.

    H[L] = 0 at initialization.
    H[L] → ∞ in both collapse modes:
      - Chaotic drift:     each L_i wanders → drift term grows
      - Catatonic collapse: all L_i identical → variance term shrinks,
                            but drift term grows (they all drift together)

    Viable regime: H bounded in a system-dependent range (typically [0.1, 5]).
    """

    def __init__(self, cfg):
        self.kappa     = cfg.kappa
        self.threshold = cfg.rollback_threshold
        self.eta       = cfg.rollback_eta
        self.L_init:   Optional[np.ndarray] = None
        self.state     = IdentityState()

    def initialize(self, L: np.ndarray):
        """
        Record initial rule state. Called once at experiment start.
        L shape: (N, 3) for [alpha, beta, gamma].
        """
        self.L_init = L.copy()

    def compute(self, L: np.ndarray) -> float:
        """
        H[L] = (1/N) Σ_i ||L_i - L_i^0||²  +  κ · mean_dim Var_i(L[:,d])

        Both terms are non-negative. H=0 iff L = L_init exactly AND all L_i identical.
        """
        if self.L_init is None:
            raise RuntimeError("IdentityCurvature.initialize() must be called first.")

        N = L.shape[0]
        # Term 1: mean squared drift from initialization
        drift = np.mean(np.sum((L - self.L_init) ** 2, axis=1))

        # Term 2: mean variance across rule dimensions
        # np.var(L, axis=0) gives variance per rule dimension, shape (3,)
        variance = self.kappa * float(np.mean(np.var(L, axis=0)))

        return float(drift + variance)

    def gradient(self, L: np.ndarray) -> np.ndarray:
        """
        ∂H/∂L_i = (2/N)(L_i - L_i^0) + κ · (2/N)(L_i - L̄)

        Shape: (N, 3) — gradient at each node for each rule dimension.
        """
        N = L.shape[0]

        # Gradient of drift term
        grad_drift = 2.0 * (L - self.L_init) / N

        # Gradient of variance term: ∂/∂L_ij Var_j = 2(L_ij - L̄_j)/N
        L_mean     = L.mean(axis=0, keepdims=True)   # (1, 3)
        grad_var   = self.kappa * 2.0 * (L - L_mean) / N

        return grad_drift + grad_var   # (N, 3)

    def check_and_rollback(
        self,
        L_current:  np.ndarray,
        L_previous: np.ndarray,
    ) -> Tuple[np.ndarray, bool]:
        """
        Compute δH. If δH > threshold, rollback to L_previous + gradient step.

        Rollback mechanism:
          L ← L_previous - η * ∇_L H[L_previous]

        This is a corrective step: the system rolls back AND takes a gradient
        step in the direction that reduces H — preventing the same spike twice.

        Returns: (L_after, did_rollback)
        """
        H_current  = self.compute(L_current)
        H_previous = self.compute(L_previous)
        delta_H    = H_current - H_previous

        self.state.H_current  = H_current
        self.state.H_previous = H_previous
        self.state.H_history.append(H_current)

        if delta_H > self.threshold:
            # Rollback with gradient correction
            grad = self.gradient(L_previous)
            L_rolled = L_previous - self.eta * grad
            self.state.rollback_count += 1
            return L_rolled, True

        return L_current, False
