"""
curiosity.py — Curiosity Engine.
Monitors prediction error, detects plateaus, issues directives.
"""
import numpy as np
from collections import deque
from dataclasses import dataclass, field


@dataclass
class CuriosityState:
    directive:          str  = "NONE"
    plateau_detected:   bool = False
    intervention_count: int  = 0
    error_history:      list = field(default_factory=list)


class CuriosityEngine:
    def __init__(self, cfg, rng):
        self.cfg    = cfg.curiosity
        self.rng    = rng
        self.window = deque(maxlen=cfg.curiosity.plateau_window)
        self.state  = CuriosityState()

    def compute_uncertainty_potential(self, theta: np.ndarray) -> np.ndarray:
        """
        U(θ_i) ≈ -log p(θ_i | distribution) via circular KDE.
        Nodes in sparse phase regions have high uncertainty.
        """
        N = len(theta)
        theta_w = theta % (2 * np.pi)
        bw = max(1.06 * np.std(theta_w) * N ** (-0.2), 0.05)

        density = np.zeros(N)
        for i in range(N):
            diffs    = np.sin(theta_w - theta_w[i])
            density[i] = np.mean(np.exp(-0.5 * (diffs / bw) ** 2)) + 1e-8

        return -np.log(density)

    def update(self, error: float) -> None:
        self.window.append(error)
        self.state.error_history.append(error)

    @property
    def is_plateauing(self) -> bool:
        if len(self.window) < self.cfg.plateau_window:
            return False
        return (max(self.window) - min(self.window)) < self.cfg.plateau_epsilon

    def get_directive(self, n_falsified: int, n_rollbacks: int) -> str:
        if n_rollbacks > 5:
            return "REDUCE_META_RATE"
        if n_falsified > 3:
            return "REFRAME_COUPLING"
        if self.state.intervention_count % 2 == 0:
            return "BOOST_CURIOSITY"
        return "DIVERSIFY_RULES"

    def apply_directive(self, directive: str, L: np.ndarray, rng) -> np.ndarray:
        L_new = L.copy()
        if directive == "BOOST_CURIOSITY":
            L_new[:, 2] = np.clip(L_new[:, 2] * (1 + self.cfg.spike_strength), 0.001, 0.5)
        elif directive == "DIVERSIFY_RULES":
            L_new += rng.normal(0, 0.05, size=L.shape)
            L_new  = np.clip(L_new, 0.001, 3.0)
        elif directive == "REFRAME_COUPLING":
            N    = L.shape[0]
            mask = rng.uniform(size=N) < 0.2
            L_new[mask, 1] = rng.uniform(0.2, 1.5, size=mask.sum())
        elif directive == "REDUCE_META_RATE":
            L_new *= 0.95
        if directive != "NONE":
            self.state.intervention_count += 1
            self.state.directive        = directive
            self.state.plateau_detected = True
        return L_new
