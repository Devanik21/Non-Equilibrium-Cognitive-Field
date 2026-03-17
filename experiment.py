"""
experiment.py — Full NECF Run Loop.
=====================================
Orchestrates all components in the correct order.
Returns observable history for analysis.

v2 Changes:
  - field.step() now returns (eps, spike_mask) — both are used
  - spike_mask is threaded to observer.record() for masked Lyapunov
  - No other logic changes — all fixes are internal to the modules
"""

import numpy as np
from typing import Generator, Optional

from config import NECFConfig, DEFAULT_CONFIG
from field import OscillatorField
from meta_dynamics import MetaDynamics
from environment import Environment
from curiosity import CuriosityEngine
from observer import Observer


class NECFExperiment:
    """
    Single NECF experiment instance.

    Usage:
        exp = NECFExperiment(cfg)

        # Streaming (for Streamlit live dashboard):
        for snap in exp.stream():
            update_dashboard(snap)

        # Batch (for offline analysis):
        history = exp.run_batch()
    """

    def __init__(self, cfg: NECFConfig = None):
        self.cfg       = cfg or DEFAULT_CONFIG
        self.rng       = np.random.default_rng(self.cfg.seed)

        self.field     = OscillatorField(self.cfg, self.rng)
        self.meta      = MetaDynamics(self.cfg, self.rng)
        self.env       = Environment(self.cfg, self.rng)
        self.curiosity = CuriosityEngine(self.cfg, self.rng)
        self.observer  = Observer()

        self.t         = 0
        self.done      = False
        self._current_directive = "NONE"

    def run_step(self) -> Optional[object]:
        """
        Advance simulation by exactly one step.
        Returns Snapshot or None if experiment is complete.

        Call order (must not be reordered):
          1. Environment signals        → ext perturbations this step
          2. Curiosity potential        → U(θ) for phase update
          3. Field step                 → updates φ, returns (eps, spike_mask)
          4. Curiosity update           → record error, detect plateau
          5. Curiosity directive        → modify L if plateauing
          6. Meta-dynamics step         → update L, check rollback
          7. Observer record            → store snapshot with spike_mask
        """
        if self.t >= self.cfg.field.T:
            self.done = True
            return None

        # ── 1. External driving signals ────────────────────────────────────
        env_signals = self.env.step(self.cfg.field.dt, self.cfg.field.N)

        # ── 2. Curiosity uncertainty potential U(θ) ────────────────────────
        U = self.curiosity.compute_uncertainty_potential(self.field.state.theta)

        # ── 3. Field step: Level-1 dynamics ────────────────────────────────
        # Returns eps (prediction error) AND spike_mask
        # spike_mask is critical — it goes to the Observer for masked λ
        eps, spike_mask = self.field.step(self.meta.L, env_signals, U)

        # ── 4. Curiosity engine: record error ─────────────────────────────
        self.curiosity.update(float(np.mean(eps)))

        # ── 5. Curiosity directive (fired only on plateau) ─────────────────
        self._current_directive = "NONE"
        if self.curiosity.is_plateauing:
            n_rollbacks = self.meta.rollback_count
            # Count recent high-error steps as "falsifications"
            n_falsified = sum(
                1 for s in self.observer.history.snapshots[-20:]
                if s.mean_error > 0.3
            )
            directive = self.curiosity.get_directive(n_falsified, n_rollbacks)
            self.meta.L = self.curiosity.apply_directive(
                directive, self.meta.L, self.rng
            )
            self._current_directive = directive

        # ── 6. Meta-dynamics: Level-3 rule evolution ──────────────────────
        L_new, did_rollback = self.meta.step(
            eps, self.field.state.W, self.cfg.field.dt
        )

        # ── 7. Observer: record with spike_mask for masked Lyapunov ────────
        snap = self.observer.record(
            t=self.t,
            field_state=self.field.state,
            L=L_new,
            H=self.meta.H,
            rollback_count=self.meta.rollback_count,
            eps=eps,
            directive=self._current_directive,
            spike_mask=spike_mask,          # ← Fix #2 threading
        )

        self.t += 1
        return snap

    def stream(self) -> Generator:
        """Yield snapshots one at a time. Use this for Streamlit."""
        while not self.done:
            snap = self.run_step()
            if snap is not None:
                yield snap

    def run_batch(self):
        """Run to completion. Returns ObserverHistory."""
        while not self.done:
            self.run_step()
        return self.observer.history

    # ─────────────────────────────────────────────────────────────────────────
    # Ablation helpers (for the paper's baseline comparisons)
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def level1_only(cls, cfg: NECFConfig = None) -> "NECFExperiment":
        """
        Level-1 ablation: fixed rules, no meta-dynamics.
        L is frozen at initialization for the entire run.
        Used as baseline vs NECF in paper.
        """
        exp = cls(cfg)
        # Override meta step to be a no-op
        exp._ablation_fixed_rules = True
        original_step = exp.meta.step

        def frozen_step(eps, W, dt):
            return exp.meta.L.copy(), False

        exp.meta.step = frozen_step
        return exp

    @classmethod
    def level2_adaptive_coupling(cls, cfg: NECFConfig = None) -> "NECFExperiment":
        """
        Level-2 ablation: coupling K_ij evolves, but same rule for all nodes.
        Equivalent to global adaptive Kuramoto — existing prior art.
        """
        exp = cls(cfg)
        # Override: beta is updated globally (not per-node L_i)
        original_step = exp.meta.step

        def global_beta_step(eps, W, dt):
            # Only beta (column 1) adapts; alpha and gamma frozen
            mean_eps = float(np.mean(eps))
            exp.meta.L[:, 1] += 0.01 * (0.5 - mean_eps) * dt
            exp.meta.L[:, 1] = np.clip(exp.meta.L[:, 1], 0.01, 3.0)
            return exp.meta.L.copy(), False

        exp.meta.step = global_beta_step
        return exp
