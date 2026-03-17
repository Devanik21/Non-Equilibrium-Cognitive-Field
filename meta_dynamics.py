"""
meta_dynamics.py — Level-2/3: Learning Rule Evolution.
======================================================
CORE NOVEL CONTRIBUTION: L_i = (α_i, β_i, γ_i) evolves as a dynamic state.

BUG FIX v2 — Epistemic Contagion Singularity (Teacher's Fix #1):
  OLD (BROKEN): neighbor_weight = 1.0 / (eps + 1e-6)
  PROBLEM: If any single node hits eps ≈ 1e-5, its weight → 100,000.
           After normalization, contagion degenerates to a one-hot vector.
           The entire field violently snaps to one lucky node's rule.
           This is NOT field dynamics — it is a discrete switch.

  NEW (CORRECT): Boltzmann softmax (thermodynamic distribution)
      w_i = exp(-eps_i / κ) / Σ_j exp(-eps_j / κ)

  This gives:
    • Smooth, continuous weight gradients — true field diffusion
    • κ (temperature) controls contagion sharpness:
        κ → 0 : approaches one-hot (crisp selection)
        κ → ∞ : approaches uniform (no selection)
    • No singularity: even eps=0 gives finite weight exp(0)/Σ
    • Thermodynamically consistent with the non-equilibrium framing

Rule evolution equation:
  dL_i/dt = F(L_i, ε_i) - λ · ∇_{L_i} H[L]

where F implements Boltzmann-weighted epistemic contagion.
"""

import numpy as np
from config import NECFConfig
from identity import IdentityCurvature


class MetaDynamics:
    """
    Level-3 dynamics: learning rules evolve under Boltzmann-weighted
    epistemic contagion + identity curvature constraint + rollback.

    State: L (N, 3) — columns are [alpha, beta, gamma] per node.
    """

    def __init__(self, cfg: NECFConfig, rng: np.random.Generator):
        self.cfg       = cfg.meta_rule
        self.id_cfg    = cfg.identity
        self.N         = cfg.field.N
        self.dt        = cfg.field.dt
        self.rng       = rng
        self.identity  = IdentityCurvature(cfg.identity)

        # Boltzmann temperature: controls contagion sharpness
        # Default 0.1 — meaningful discrimination without singularity
        self.kappa_boltzmann = getattr(cfg.meta_rule, 'kappa_boltzmann', 0.1)

        # Initialize rule field with small symmetry-breaking noise
        self.L = np.column_stack([
            np.full(self.N, cfg.meta_rule.alpha_init),
            np.full(self.N, cfg.meta_rule.beta_init),
            np.full(self.N, cfg.meta_rule.gamma_init),
        ])  # shape (N, 3)

        self.L += rng.normal(0, 0.01, size=self.L.shape)
        self.L = np.clip(self.L, 1e-3, 3.0)

        # Initialize identity reference (called once, here)
        self.identity.initialize(self.L)

        # Previous L for rollback comparison
        self._L_previous = self.L.copy()

        # Diagnostic counters
        self.total_steps = 0

    # ─────────────────────────────────────────────────────────────────────────
    # FIX #1: Boltzmann Softmax Epistemic Contagion
    # ─────────────────────────────────────────────────────────────────────────

    def _boltzmann_weights(self, eps: np.ndarray) -> np.ndarray:
        """
        Compute Boltzmann-softmax neighbor influence weights.

        w_i = exp(-eps_i / κ) / Σ_j exp(-eps_j / κ)

        Properties:
          - w_i ∈ (0, 1) for all i — no singularities ever
          - Σ_i w_i = 1 — proper probability distribution
          - Low-error nodes get higher weight (thermodynamically preferred)
          - κ controls sharpness: small κ = selective, large κ = diffuse

        Uses log-sum-exp trick for numerical stability:
          log(Σ exp(x_i)) = max(x) + log(Σ exp(x_i - max(x)))
        """
        log_w = -eps / self.kappa_boltzmann        # shape (N,)
        log_w -= log_w.max()                       # log-sum-exp stability
        w = np.exp(log_w)
        w /= w.sum() + 1e-10                       # normalize
        return w                                   # shape (N,)

    def _epistemic_contagion(
        self,
        eps: np.ndarray,
        W: np.ndarray,
    ) -> np.ndarray:
        """
        Boltzmann-weighted rule contagion:
        Low-error nodes propagate their rules to high-error neighbors.

        For node i:
          L_target_i = Σ_j (W_ij * w_j * L_j) / (Σ_j W_ij * w_j)

          where w_j = Boltzmann weight of node j

          dL_i/dt += μ * (L_target_i - L_i) * eps_i

        Interpretation: high-error nodes are "receptive" (eps_i large),
        and they pull toward the Boltzmann-weighted mean of their
        low-error neighbors — a thermodynamically smooth field update.
        """
        # Boltzmann weights: shape (N,)
        w = self._boltzmann_weights(eps)

        # Weighted coupling matrix: W_ij * w_j — shape (N, N)
        W_weighted = W * w[np.newaxis, :]

        # Row-normalized target rules: shape (N, 3)
        row_sums = W_weighted.sum(axis=1, keepdims=True) + 1e-10
        L_target = (W_weighted @ self.L) / row_sums   # (N, 3)

        # Update rates per rule dimension
        mu = np.array([
            self.cfg.mu_alpha,
            self.cfg.mu_beta,
            self.cfg.mu_gamma,
        ])  # shape (3,)

        # Contagion: high-error nodes update more strongly toward target
        # eps[:, np.newaxis] makes error the "receptivity" of each node
        contagion = mu[np.newaxis, :] * (L_target - self.L) * eps[:, np.newaxis]
        return contagion   # shape (N, 3)

    # ─────────────────────────────────────────────────────────────────────────
    # Main Step
    # ─────────────────────────────────────────────────────────────────────────

    def step(
        self,
        eps: np.ndarray,
        W: np.ndarray,
        dt: float,
    ) -> tuple:
        """
        Advance rule field by one timestep.

        dL/dt = F_contagion(L, eps, W) - λ * ∇_L H[L]

        Returns:
            L_new (N, 3)   — updated rule field
            did_rollback   — True if identity rollback was triggered
        """
        self._L_previous = self.L.copy()

        # Term 1: Boltzmann epistemic contagion
        contagion = self._epistemic_contagion(eps, W)

        # Term 2: Identity curvature gradient (opposes drift + collapse)
        identity_grad = self.identity.gradient(self.L)

        # Combined update
        dL = (contagion - self.id_cfg.lambda_identity * identity_grad) * dt

        # Clip to valid parameter bounds
        L_candidate = self.L + dL
        L_candidate[:, 0] = np.clip(L_candidate[:, 0],
                                     self.cfg.alpha_bounds[0],
                                     self.cfg.alpha_bounds[1])
        L_candidate[:, 1] = np.clip(L_candidate[:, 1],
                                     self.cfg.beta_bounds[0],
                                     self.cfg.beta_bounds[1])
        L_candidate[:, 2] = np.clip(L_candidate[:, 2],
                                     self.cfg.gamma_bounds[0],
                                     self.cfg.gamma_bounds[1])

        # Rollback check: if identity curvature spikes, revert
        L_new, did_rollback = self.identity.check_and_rollback(
            L_candidate, self._L_previous
        )

        self.L = L_new
        self.total_steps += 1
        return self.L.copy(), did_rollback

    # ─────────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def H(self) -> float:
        """Current identity curvature H[L]."""
        return self.identity.compute(self.L)

    @property
    def L_current(self) -> np.ndarray:
        """Current rule field copy."""
        return self.L.copy()

    @property
    def rollback_count(self) -> int:
        return self.identity.state.rollback_count

    def boltzmann_weights_snapshot(self, eps: np.ndarray) -> np.ndarray:
        """Expose current Boltzmann weights for dashboard visualization."""
        return self._boltzmann_weights(eps)
