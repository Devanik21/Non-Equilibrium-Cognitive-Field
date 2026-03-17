"""
observer.py — All Measurable Observables for NECF.
===================================================
Computes: r(t), λ_max proxy, entropy, prediction error,
rule diversity, H[L], rollback frequency, regime classification.

BUG FIX v2 — False Chaos Trigger (Teacher's Fix #2):
  OLD (BROKEN): Lyapunov proxy computed from raw ||δθ(t)||

  PROBLEM: environment.py deliberately resets phases of spiked nodes:
      new_theta[spike_nodes] = rng.uniform(0, 2π)
    
    A spiked node's phase can jump from 0.1 to 5.9 radians in one step.
    That is a phase difference of ~5.8 — enormous.
    The raw L2 norm ||δθ|| explodes, gets logged, returns λ >> 1.
    Your rollback threshold fires instantly and CONTINUOUSLY.
    The system freezes in perpetual rollback. No dynamics at all.

  ROOT CAUSE: You are measuring EXTERNAL INJECTION as INTERNAL CHAOS.
    These are physically distinct. The Lyapunov exponent is defined
    for the system's OWN dynamics, not for the noise you feed it.

  NEW (CORRECT): Mask spiked nodes out of the Lyapunov computation.
    Only measure phase divergence on nodes that were NOT externally spiked.
    This correctly isolates the system's internal dynamical complexity.

  Implementation:
    - field.step() now returns spike_mask alongside eps
    - observer.record() accepts spike_mask parameter
    - Lyapunov proxy computed only on ~spike_mask nodes
    - If all nodes were spiked (edge case): return NaN, not a fake value

  Mathematical note:
    The masked Lyapunov proxy is:
      λ_proxy = (1/Δt) * log(||δθ[~spike_mask]|| / N_unmasked)
    
    This is still an approximation (a real λ_max requires the variational
    equations of the full system), but it is now an approximation of the
    RIGHT QUANTITY.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Snapshot:
    """Complete observable state at one time step."""
    t:                    int
    r:                    float   # Kuramoto order parameter
    mean_error:           float   # mean prediction error across field
    H:                    float   # identity curvature H[L]
    alpha_mean:           float   # mean α across nodes
    beta_mean:            float   # mean β across nodes
    gamma_mean:           float   # mean γ across nodes
    rule_diversity:       float   # mean Var(L_i) across rule dimensions
    rollback_count:       int     # cumulative identity rollbacks
    curiosity_directive:  str     # last CuriosityEngine directive
    lyapunov_proxy:       float   # MASKED internal Lyapunov estimate
    entropy:              float   # Shannon entropy of phase distribution
    n_spiked:             int     # number of externally spiked nodes this step
    n_valid_lyapunov:     int     # number of nodes used for λ estimate


@dataclass
class ObserverHistory:
    snapshots: List[Snapshot] = field(default_factory=list)

    def to_arrays(self) -> dict:
        """Convert snapshot list to numpy arrays for plotting."""
        if not self.snapshots:
            return {}
        scalar_fields = [
            't', 'r', 'mean_error', 'H',
            'alpha_mean', 'beta_mean', 'gamma_mean',
            'rule_diversity', 'rollback_count',
            'lyapunov_proxy', 'entropy', 'n_spiked', 'n_valid_lyapunov',
        ]
        return {
            **{k: np.array([getattr(s, k) for s in self.snapshots])
               for k in scalar_fields},
            'directive': [s.curiosity_directive for s in self.snapshots],
        }


class Observer:
    """
    Computes all observables. Accepts spike_mask to correctly isolate
    internal dynamics from external driving.
    """

    def __init__(self):
        self.history         = ObserverHistory()
        self._prev_theta:    Optional[np.ndarray] = None
        self._prev_spike_mask: Optional[np.ndarray] = None

    # ─────────────────────────────────────────────────────────────────────────
    # FIX #2: Masked Lyapunov Proxy
    # ─────────────────────────────────────────────────────────────────────────

    def compute_lyapunov_proxy(
        self,
        theta: np.ndarray,
        spike_mask: np.ndarray,
    ) -> tuple:
        """
        Approximate internal max Lyapunov exponent.

        ONLY computed on nodes that were NOT externally spiked this step
        AND were NOT spiked in the previous step (their deltas are clean).

        Args:
            theta:      current phases (N,)
            spike_mask: binary array (N,) — 1 = spiked this step

        Returns:
            (lyapunov_proxy, n_valid_nodes)
        """
        if self._prev_theta is None:
            # First call: just store state, return neutral value
            self._prev_theta      = theta.copy()
            self._prev_spike_mask = spike_mask.copy()
            return 0.0, int(theta.shape[0])

        # Valid nodes: not spiked NOW and not spiked in PREVIOUS step.
        # A node spiked at t-1 has a phase reset at t-1 → its δθ is
        # contaminated even at the current step.
        valid_mask = (~spike_mask.astype(bool)) & (~self._prev_spike_mask.astype(bool))
        n_valid = int(valid_mask.sum())

        if n_valid < 2:
            # Edge case: almost all nodes spiked — cannot estimate λ
            self._prev_theta      = theta.copy()
            self._prev_spike_mask = spike_mask.copy()
            return float('nan'), n_valid

        # Phase delta on valid nodes only (circular distance)
        delta = theta[valid_mask] - self._prev_theta[valid_mask]
        delta_circular = np.arctan2(np.sin(delta), np.cos(delta))

        norm = np.linalg.norm(delta_circular)

        # Update stored state
        self._prev_theta      = theta.copy()
        self._prev_spike_mask = spike_mask.copy()

        if norm < 1e-12:
            return -2.0, n_valid   # strongly converging

        # Lyapunov proxy: log of normalized phase divergence rate
        lya = float(np.log(norm / np.sqrt(n_valid) + 1e-10))
        return lya, n_valid

    # ─────────────────────────────────────────────────────────────────────────
    # Other Observables (unchanged from v1)
    # ─────────────────────────────────────────────────────────────────────────

    def compute_entropy(self, theta: np.ndarray, bins: int = 24) -> float:
        """
        Shannon entropy of phase distribution.
        H = -Σ p_k log(p_k)  over histogram bins of [0, 2π].
        High H → random/chaotic.  Low H → synchronized/collapsed.
        """
        hist, _ = np.histogram(
            theta % (2 * np.pi),
            bins=bins,
            range=(0.0, 2 * np.pi),
        )
        hist = hist.astype(float)
        total = hist.sum()
        if total < 1e-10:
            return 0.0
        p = hist / total
        # Avoid log(0)
        nz = p > 0
        H = float(-np.sum(p[nz] * np.log(p[nz])))
        return H

    def record(
        self,
        t:                   int,
        field_state,               # OscillatorField.state (FieldState dataclass)
        L:                   np.ndarray,
        H:                   float,
        rollback_count:      int,
        eps:                 np.ndarray,
        directive:           str,
        spike_mask:          np.ndarray,   # NEW: required for masked Lyapunov
    ) -> Snapshot:
        """
        Record one complete snapshot.

        spike_mask MUST be the actual spike mask from this timestep's
        Environment.step() call — not a placeholder.
        """
        lya, n_valid = self.compute_lyapunov_proxy(field_state.theta, spike_mask)

        snap = Snapshot(
            t=t,
            r=float(field_state.r),
            mean_error=float(np.mean(eps)),
            H=H,
            alpha_mean=float(np.mean(L[:, 0])),
            beta_mean=float(np.mean(L[:, 1])),
            gamma_mean=float(np.mean(L[:, 2])),
            rule_diversity=float(np.mean(np.var(L, axis=0))),
            rollback_count=rollback_count,
            curiosity_directive=directive,
            lyapunov_proxy=lya,
            entropy=self.compute_entropy(field_state.theta),
            n_spiked=int(spike_mask.sum()),
            n_valid_lyapunov=n_valid,
        )
        self.history.snapshots.append(snap)
        return snap

    # ─────────────────────────────────────────────────────────────────────────
    # Regime Classification
    # ─────────────────────────────────────────────────────────────────────────

    def classify_regime(self, snap: Snapshot) -> str:
        """
        Classify current dynamical regime based on MASKED Lyapunov
        and order parameter.

        Viable:    λ ∈ (-0.5, 0.8)  AND  r > 0.15  AND  H ∈ (0.05, 10)
        Chaotic:   λ > 0.8  (internal divergence, not spike noise)
        Catatonic: r < 0.05 (complete desynchronization)
        """
        lya = snap.lyapunov_proxy
        r   = snap.r
        H   = snap.H

        if np.isnan(lya):
            return "SPIKE_DOMINATED"     # almost all nodes spiked — transient
        if lya > 0.8:
            return "CHAOTIC"
        if r < 0.05:
            return "CATATONIC"
        if H > 10.0 or H < 0.01:
            return "IDENTITY_BROKEN"
        return "VIABLE"
