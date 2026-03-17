"""
field.py — Level-1 Oscillator Field.
=====================================
Core state: A(N,) amplitudes, θ(N,) phases.
Evolution: generalized Kuramoto + curiosity gradient + external driving.

BUG FIX v2 — Numba Optimization (Teacher's Fix #3):
  OLD (SLOW): NumPy broadcasting for O(N²) phase difference matrix
      theta_diff = theta[np.newaxis, :] - theta[:, np.newaxis]  # (N, N) alloc
      sync_pull  = np.sum(W * A * np.sin(theta_diff), axis=1) / N

  PROBLEM: At N=64, dt=0.01, T=2000:
      - 2000 allocations of 64×64 float64 matrix = 2000 × 32KB = 64MB churn
      - Each allocation triggers GC pressure
      - At N=128: 2000 × 128KB = 256MB — severe Streamlit lag
      - NumPy sin() on (N,N) is not cache-friendly for large N

  FIX: @numba.njit compiled inner loop
      - Compiles to native machine code on first call (JIT)
      - Cache-friendly sequential access pattern
      - No intermediate (N,N) matrix — computes sync_pull directly
      - 10-50× faster for N≥64; critical for N=128 or N=256

  IMPORTANT: Numba JIT has a one-time compile cost (~1-3 seconds on
  first call). We pre-warm it during __init__ to avoid stalling
  the Streamlit dashboard on the first simulation step.

  FALLBACK: If numba is not installed, falls back to NumPy broadcasting
  with a warning. Dashboard still works, just slower for large N.

  Also: field.step() now RETURNS spike_mask alongside eps,
  so observer.py can use it for masked Lyapunov computation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

# ─── Numba Optional Import ────────────────────────────────────────────────────
try:
    import numba
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("[field.py] WARNING: numba not installed. Falling back to NumPy broadcasting.")
    print("           Install with: pip install numba>=0.58.0")
    print("           Dashboard will be slower for N >= 64.")


# ─── Numba-Compiled Kuramoto Kernel ──────────────────────────────────────────

if NUMBA_AVAILABLE:

    @njit(parallel=True, cache=True, fastmath=True)
    def _kuramoto_sync_pull_numba(
        theta: np.ndarray,   # (N,)
        A:     np.ndarray,   # (N,)
        W:     np.ndarray,   # (N, N)
        beta:  np.ndarray,   # (N,) — local coupling strength from L
    ) -> np.ndarray:
        """
        Compute Kuramoto synchrony pull for all nodes.

        sync_pull[i] = β_i * (1/N) * Σ_j W[i,j] * A[j] * sin(θ[j] - θ[i])

        Compiled to native code via @njit:
          - No (N,N) intermediate matrix allocated
          - Parallel outer loop over i (prange)
          - fastmath=True: SIMD vectorization of sin() inner loop
          - cache=True: compiled binary cached to disk after first run

        Returns: sync_pull (N,)
        """
        N = theta.shape[0]
        sync_pull = np.zeros(N)
        for i in prange(N):
            s = 0.0
            for j in range(N):
                s += W[i, j] * A[j] * np.sin(theta[j] - theta[i])
            sync_pull[i] = beta[i] * s / N
        return sync_pull

else:

    def _kuramoto_sync_pull_numba(theta, A, W, beta):
        """NumPy fallback when Numba is unavailable."""
        theta_diff = theta[np.newaxis, :] - theta[:, np.newaxis]   # (N, N)
        sync_pull  = np.sum(W * A[np.newaxis, :] * np.sin(theta_diff), axis=1) / len(theta)
        return beta * sync_pull


# ─── Field State Dataclass ────────────────────────────────────────────────────

@dataclass
class FieldState:
    """Complete snapshot of oscillator field at time t."""
    A:     np.ndarray      # amplitudes  (N,)  ∈ (0, 1]
    theta: np.ndarray      # phases      (N,)  ∈ [0, 2π)
    omega: np.ndarray      # natural frequencies (N,) — fixed throughout
    W:     np.ndarray      # coupling matrix     (N, N) — symmetric, zero diagonal
    r:     float  = 0.0    # Kuramoto order parameter magnitude
    psi:   float  = 0.0    # mean phase ψ
    t:     int    = 0      # current step index


# ─── Oscillator Field ─────────────────────────────────────────────────────────

class OscillatorField:
    """
    Coupled oscillator field — the Level-1 dynamical substrate.

    Departures from standard Kuramoto:
      1. Local coupling strength β_i drawn from rule field L (not global K)
      2. Local curiosity weight γ_i from L drives exploration gradient
      3. Amplitude A_i evolves under prediction error (not fixed at 1)
      4. External driving (Lorenz + periodic + spikes) breaks equilibrium
      5. Returns spike_mask alongside eps for masked Lyapunov computation
    """

    def __init__(self, cfg, rng: np.random.Generator):
        self.cfg  = cfg
        self.N    = cfg.field.N
        self.dt   = cfg.field.dt
        self.rng  = rng

        # Fixed natural frequencies — drawn once, never change
        self.omega = rng.normal(
            cfg.oscillator.omega_mean,
            cfg.oscillator.omega_std,
            size=self.N,
        )

        # Initial amplitudes
        A_init = np.full(self.N, cfg.oscillator.A_init)

        # Initial phases
        if cfg.oscillator.theta_init == "random":
            theta_init = rng.uniform(0.0, 2 * np.pi, size=self.N)
        elif cfg.oscillator.theta_init == "uniform":
            theta_init = np.linspace(0.0, 2 * np.pi, self.N, endpoint=False)
        elif cfg.oscillator.theta_init == "clustered":
            theta_init = rng.normal(np.pi, 0.5, size=self.N) % (2 * np.pi)
        else:
            theta_init = rng.uniform(0.0, 2 * np.pi, size=self.N)

        # Symmetric coupling matrix with zero diagonal
        W = rng.uniform(0.5, 1.5, size=(self.N, self.N))
        np.fill_diagonal(W, 0.0)
        W = (W + W.T) / 2.0

        self.state = FieldState(
            A=A_init, theta=theta_init, omega=self.omega, W=W,
        )

        # Pre-warm Numba JIT on first call with dummy data to avoid
        # stalling the dashboard on the first real simulation step
        if NUMBA_AVAILABLE:
            self._prewarm_numba()

    def _prewarm_numba(self):
        """
        Trigger JIT compilation before the dashboard starts.
        Called once in __init__. Takes ~1-3s; happens during setup.
        """
        dummy_theta = np.zeros(self.N)
        dummy_A     = np.ones(self.N)
        dummy_W     = np.eye(self.N)
        dummy_beta  = np.ones(self.N)
        _kuramoto_sync_pull_numba(dummy_theta, dummy_A, dummy_W, dummy_beta)

    # ─────────────────────────────────────────────────────────────────────────

    def compute_order_parameter(self) -> Tuple[float, float]:
        """
        Kuramoto complex order parameter:
          z = r * exp(iψ) = (1/N) * Σ_i A_i * exp(iθ_i)

        r ∈ [0, 1]:  0 = incoherent, 1 = fully synchronized
        ψ ∈ [0, 2π): mean phase angle
        """
        z   = np.mean(self.state.A * np.exp(1j * self.state.theta))
        r   = float(np.abs(z))
        psi = float(np.angle(z)) % (2 * np.pi)
        return r, psi

    def compute_prediction_error(
        self,
        target_phases: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Local prediction error ε_i for each node.

        Without target: deviation from mean field ψ (circular distance).
        With target:    circular distance to provided target phases.

        Returns array (N,) in [0, 1].
        """
        if target_phases is not None:
            diff = self.state.theta - target_phases
        else:
            diff = self.state.theta - self.state.psi
        # Circular distance normalized to [0, 1]
        return np.sin(diff / 2.0) ** 2

    def step(
        self,
        L:                   np.ndarray,      # (N, 3)
        env_signals:         dict,
        curiosity_potential: np.ndarray,       # (N,)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advance field by one step dt.

        Returns:
            eps        (N,)  — local prediction error BEFORE phase update
            spike_mask (N,)  — binary: 1 = this node was externally spiked
                               MUST be passed to observer.record()
        """
        alpha = L[:, 0]   # error sensitivity
        beta  = L[:, 1]   # coupling strength
        gamma = L[:, 2]   # curiosity weight

        # ── Compute prediction error BEFORE update (for meta-dynamics) ───────
        r, psi           = self.compute_order_parameter()
        self.state.r     = r
        self.state.psi   = psi
        eps              = self.compute_prediction_error()

        # ── Phase update ──────────────────────────────────────────────────────

        # Term 1: Numba-compiled Kuramoto sync pull
        sync_pull = _kuramoto_sync_pull_numba(
            self.state.theta,
            self.state.A,
            self.state.W,
            beta,               # local β_i from rule field
        )

        # Term 2: Curiosity gradient (γ_i drives nodes toward uncertain regions)
        curiosity_pull = gamma * curiosity_potential

        # Term 3: Lorenz phase kick (external, chaotic)
        lorenz_kick = env_signals["lorenz_phase_kick"]

        # Integrate phase
        d_theta = (
            self.omega
            + sync_pull
            + curiosity_pull
            + lorenz_kick
        ) * self.dt

        new_theta = (self.state.theta + d_theta) % (2 * np.pi)

        # ── Spikes: external Poisson phase reset ──────────────────────────────
        # spike_mask is returned to caller — DO NOT use these nodes for λ_max
        spike_mask = env_signals["spike_mask"].astype(bool)
        if spike_mask.any():
            new_theta[spike_mask] = self.rng.uniform(
                0.0, 2 * np.pi, size=spike_mask.sum()
            )

        # ── Amplitude update ──────────────────────────────────────────────────
        # dA/dt = -α_i * ε_i * A_i + σ * η_i + periodic modulation
        noise  = self.rng.normal(0.0, self.cfg.field.noise_sigma, size=self.N)
        d_A = (
            -alpha * eps * self.state.A
            + noise
            + env_signals["periodic_amp_mod"]
        ) * self.dt

        new_A = np.clip(self.state.A + d_A, 0.01, 1.0)

        # ── Commit state ──────────────────────────────────────────────────────
        self.state.theta = new_theta
        self.state.A     = new_A
        self.state.t    += 1

        # Recompute order parameter with updated phases
        r, psi       = self.compute_order_parameter()
        self.state.r = r
        self.state.psi = psi

        # Return spike_mask as float array for storage
        return eps, spike_mask.astype(float)
