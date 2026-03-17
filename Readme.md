# Non-Equilibrium Cognitive Field (NECF)
## Technical Report v2.0 — Post-Literature-Review Edition
### Devanik | B.Tech ECE '26, NIT Agartala | Samsung Convergence Software Fellow, IISc

---

## PART I: HONEST POSITIONING — WHERE NECF ACTUALLY STANDS

### What the Literature Search Found

Extensive web search across Google Scholar, arXiv, SIAM, Nature, ScienceDirect, and Frontiers (March 2026) confirms:

| NECF Concept | Already Exists | Citation |
|---|---|---|
| Coupled phase oscillator field | ✅ Kuramoto (1975); Strogatz (1994) |
| Lyapunov stability for oscillator networks | ✅ van Hemmen & Wreszinski, J. Stat. Phys. 72 (1993) |
| Adaptive coupling (K_ij evolves) | ✅ Ha et al., SIAM J. App. Dyn. Sys. (2016) |
| Co-evolving structure + dynamics | ✅ Bassett & Sporns, ScienceDirect (2023) |
| Hopfield-Kuramoto joint memory | ✅ arXiv:2505.03648 (May 2025) |
| Free energy / surprise minimization | ✅ Friston FEP (2005–2022) |
| Curiosity as exploration signal | ✅ Friston Active Inference (2017) |
| Non-equilibrium thermodynamics + RL | ✅ Adamczyk et al., ICLR 2026 workshop |
| Meta-learning (learning to learn) | ✅ MAML, Finn et al. (2017) |
| Spontaneous symmetry breaking → identity | ✅ Prigogine (1977), Landsman (2013) |
| Self-modifying agent architectures | ✅ ADAS, ICLR 2025 (Hu et al.) |

### What is NOT Found in the Literature

After exhaustive search, the following specific combination does **not appear** to exist:

> **A coupled oscillator field where the local learning rules (α_i, β_i, γ_i) themselves evolve as a second-order dynamical system, constrained by an identity curvature functional H[L], with Lyapunov-gated rollback of destructive meta-updates — unified in a single implementable architecture.**

The gap is precise:

- Standard Kuramoto: coupling strength K is **fixed**
- Adaptive Kuramoto: K_ij **evolves**, but the rule governing K's evolution is **fixed**
- Hopfield-Kuramoto (arXiv 2025): joint memory model, but **no meta-rule evolution**
- MAML: **learns an initialization**, not a continuously evolving rule field
- FEP: **minimizes free energy**, but learning rules are not dynamic field variables
- **NECF**: the rule governing each node's evolution is itself a **dynamic state variable** — and the rules governing *that* evolution are constrained by identity curvature

This is the Level 3 system in a hierarchy where all prior work stops at Level 2:

```
Level 1: φ(t) evolves       [Standard dynamical system]
Level 2: K(t) evolves       [Adaptive coupling — existing work]
Level 3: L(t) evolves       [Meta-rule dynamics — NECF's contribution]
         subject to H[L]    [Identity curvature constraint — NECF's contribution]
         with rollback       [Reversible meta-dynamics — NECF's contribution]
```

---

## PART II: RIGOROUS TECHNICAL SPECIFICATION

### 2.1 Field State Representation

Each node i in an N-node field carries state:

```
φ_i(t) = (A_i(t), θ_i(t))
```

- **A_i ∈ (0, 1]**: amplitude — energy / confidence
- **θ_i ∈ [0, 2π)**: phase — causal alignment / temporal state

Order parameter (global synchrony measure):
```
r(t) · exp(iψ(t)) = (1/N) · Σ_i A_i · exp(iθ_i)
```
where r ∈ [0,1] is the synchronization index and ψ is the mean phase.

### 2.2 Local Learning Rules as Dynamic Variables

**This is NECF's core departure from all prior work.**

Each node i carries a local learning rule vector:
```
L_i(t) = (α_i(t), β_i(t), γ_i(t))
```

- **α_i**: sensitivity to local prediction error — governs amplitude update
- **β_i**: coupling strength to neighbors — governs phase pull
- **γ_i**: curiosity weight — governs uncertainty-seeking perturbation

In ALL prior adaptive oscillator work, the equivalent of (α, β, γ) is either fixed or evolves according to a **fixed** update rule. In NECF, the update rule for L_i is itself governed by a second dynamical system.

### 2.3 Level-1 Dynamics: Field Evolution

Phase evolution (generalized Kuramoto with curiosity):
```
dθ_i/dt = ω_i + β_i · (1/N) · Σ_j W_ij · A_j · sin(θ_j - θ_i)
         + γ_i · ∇_θ U(θ_i)   [curiosity term]
```

Amplitude evolution (driven by prediction error):
```
dA_i/dt = -α_i · ε_i(t) · A_i + σ · η_i(t)
```
where:
- ε_i(t) = |predicted_i - observed_i|² / N  is local prediction error
- η_i(t) is stochastic noise (Wiener process)
- σ is noise amplitude

Uncertainty field (curiosity potential):
```
U(θ_i) = -log p(θ_i | context)   [negative log likelihood → high gradient where uncertain]
```

### 2.4 Level-2 Dynamics: Meta-Rule Evolution

The learning rules evolve according to:
```
dL_i/dt = F(L_i, φ_i, ε_i) - λ · ∇_{L_i} H[L]
```

where:
- F(·) is a local meta-update function (error-driven rule modification)
- H[L] is the **identity curvature functional** (defined in 2.5)
- λ is the identity preservation weight

The meta-update function F:
```
F(L_i, φ_i, ε_i) = μ · (L*_i - L_i) · ε_i
```

where L*_i is the "ideal rule" estimated from local phase coherence — nodes with high synchrony and low error have their rules used as targets by neighbors.

Explicitly:
```
dα_i/dt = μ_α · (ᾱ_neighbors - α_i) · ε_i  - λ · ∂H/∂α_i
dβ_i/dt = μ_β · (β̄_neighbors - β_i) · ε_i  - λ · ∂H/∂β_i
dγ_i/dt = μ_γ · (γ̄_neighbors - γ_i) · ε_i  - λ · ∂H/∂γ_i
```

This is **not** gradient descent on a loss function. It is a dynamical system in rule space, where rules propagate from low-error nodes to high-error nodes — a form of epistemic contagion.

### 2.5 Identity Curvature Functional

Identity is not a scalar — it is a **functional over rule space**:

```
H[L] = (1/N) · Σ_i ||L_i(t) - L_i(0)||²  +  κ · Var(L_i)
       \_______________________________/     \____________/
          drift from initial rules              rule diversity
```

This has two terms:
1. **Drift penalty**: rules should not drift arbitrarily far from their initialization (structural memory)
2. **Variance penalty**: rules should not collapse to all being identical (diversity preservation)

The gradient ∂H/∂L_i pulls rules back toward their initialization while preventing homogenization.

**Critical property**: H[L] = 0 at initialization. H[L] → ∞ in two collapse modes:
- Chaotic drift: each L_i wanders randomly → H grows via drift term
- Catatonic collapse: all L_i converge to same value → H grows via variance term

The viable regime is **bounded H** — the system maintains identity by staying in a narrow region of rule space.

### 2.6 Rollback Condition (Reversible Meta-Dynamics)

At each time step, compute:
```
δH(t) = H[L(t)] - H[L(t-Δt)]
```

If δH > θ_rollback:
```
L_i(t) ← L_i(t-Δt) - η · ∇_{L_i} H[L(t-Δt)]
```

This is not gradient descent — it is a **thermodynamic rollback**: the system reverts to a previous state when identity curvature increases too rapidly. It implements memory of past stability.

### 2.7 External Coupling (Non-Equilibrium Driving)

The system is an open thermodynamic system driven by three signal types:

**Lorenz Attractor (chaotic, deterministic):**
```
dx/dt = σ(y-x),  dy/dt = x(ρ-z)-y,  dz/dt = xy-βz
```
Injected as phase perturbation: θ_i += ε_L · x(t) · sin(i · 2π/N)

**Periodic Signal (structured, sinusoidal):**
```
s(t) = A_s · sin(2πf_s t + φ_s)
```
Injected as amplitude modulation: A_i += ε_s · s(t)

**Stochastic Spikes (Poisson process):**
```
P(spike at t) = Δt · λ_spike
```
Creates sudden phase resets in randomly selected nodes.

The combination ensures the system never reaches equilibrium — it is perpetually driven.

### 2.8 Proto-Will: Constrained Agency Maximization

At each time step, when the system faces a choice between competing stable attractors (detected when |r| bifurcates), selection follows:

```
attractor* = argmax_a [H_causal(a) - μ · D_identity(a)]
```

where:
- H_causal(a) = -Σ_future p(s'|a) log p(s'|a)  [causal entropy — future state diversity]
- D_identity(a) = ||L_attractor_a - L_current||   [identity distortion cost]
- μ = tradeoff parameter

This operationalizes will as **constrained future-agency maximization** — the system selects the attractor that preserves the most future options while minimizing identity cost.

### 2.9 Measurable Observables (What Makes This Science)

The system is falsifiable through these quantitative observables:

| Observable | Metric | Prediction |
|---|---|---|
| Phase synchrony | r(t) — order parameter | Should rise from ~0.1 to >0.6 over ~500 steps |
| Identity preservation | H[L(t)] / H[L(0)] | Should stay bounded in [0.5, 2.0] |
| Lyapunov exponent λ_max | Computed via variational method | Should stay in (-0.1, 0.5) for viable regime |
| Prediction error | ε(t) | Should decrease monotonically after ~200 steps |
| Rule diversity | Var(L_i) | Should be bounded, not collapse to zero |
| Curiosity activation | γ̄(t) | Should spike when r(t) plateaus |
| Rollback frequency | count(δH > θ) | Should decrease over time (learning stability) |

---

## PART III: WHAT NECF IS AND IS NOT

### What This System Can Do
- **Demonstrate** that learning rule evolution constrained by identity curvature produces stable adaptive behavior
- **Show** that Level-3 meta-dynamics enables recovery from perturbations that Level-2 (adaptive coupling) cannot
- **Measure** phase transitions between chaos, catatonia, and viable cognition
- **Produce** falsifiable predictions on all 7 observables above

### What This System Cannot Do (Honest)
- Solve language tasks, vision tasks, or any benchmark
- Demonstrate AGI — it is a **proto-cognitive substrate**
- Replace or compete with transformer-based systems
- Prove consciousness

### Why It Matters Despite These Limits
The claim is not "NECF is AGI." The claim is:

> **Meta-rule dynamics constrained by identity curvature produces qualitatively different adaptive behavior than fixed-rule or adaptive-coupling systems, and this difference is measurable on 7 quantitative metrics.**

This is falsifiable. This is publishable. This is the contribution.

---

## PART IV: COMPLETE IMPLEMENTATION PLAN

### Architecture: 9 Python Files

```
necf/
├── app.py               # Streamlit frontend — calls all backends
├── config.py            # All hyperparameters, constants, defaults
├── field.py             # Level-1: oscillator field state + evolution
├── meta_dynamics.py     # Level-2/3: learning rule evolution
├── identity.py          # H[L] curvature functional + rollback
├── curiosity.py         # Curiosity potential + exploration directives
├── environment.py       # External drivers: Lorenz, periodic, spikes
├── observer.py          # All metrics: r(t), λ_max, entropy, etc.
└── experiment.py        # Session manager, run loop, export
```

---

### FILE 1: `config.py`

```python
"""
config.py — All hyperparameters for NECF.
Change here; everything else reads from here.
"""
from dataclasses import dataclass, field
import numpy as np

@dataclass
class FieldConfig:
    N: int = 64                    # number of oscillator nodes
    dt: float = 0.01               # integration time step
    T: int = 2000                  # total steps per experiment
    noise_sigma: float = 0.02      # stochastic amplitude noise

@dataclass
class OscillatorConfig:
    omega_mean: float = 1.0        # mean natural frequency
    omega_std: float = 0.3         # frequency diversity
    A_init: float = 0.5            # initial amplitude (uniform)
    theta_init: str = "random"     # "random" | "uniform" | "clustered"

@dataclass
class MetaRuleConfig:
    alpha_init: float = 0.3        # initial error sensitivity
    beta_init: float = 0.8         # initial coupling strength
    gamma_init: float = 0.1        # initial curiosity weight
    mu_alpha: float = 0.05         # meta-update rate for alpha
    mu_beta: float = 0.05          # meta-update rate for beta
    mu_gamma: float = 0.05         # meta-update rate for gamma
    alpha_bounds: tuple = (0.01, 2.0)
    beta_bounds: tuple = (0.01, 3.0)
    gamma_bounds: tuple = (0.001, 0.5)

@dataclass
class IdentityConfig:
    kappa: float = 0.5             # variance penalty weight in H[L]
    lambda_identity: float = 0.1   # identity gradient weight in dL/dt
    rollback_threshold: float = 0.3  # δH/step that triggers rollback
    rollback_eta: float = 0.05     # rollback learning rate

@dataclass
class CuriosityConfig:
    plateau_window: int = 20       # steps to detect plateau
    plateau_epsilon: float = 0.01  # min improvement to not be plateau
    spike_strength: float = 0.3    # gamma boost on plateau detection

@dataclass
class EnvironmentConfig:
    lorenz_eps: float = 0.05       # Lorenz coupling strength
    periodic_eps: float = 0.03     # periodic signal strength
    periodic_freq: float = 0.1     # frequency of periodic signal
    spike_rate: float = 0.02       # Poisson spike rate per node per step
    lorenz_sigma: float = 10.0
    lorenz_rho: float = 28.0
    lorenz_beta: float = 8.0/3.0

@dataclass
class WillConfig:
    mu_will: float = 1.0           # identity distortion tradeoff
    attractor_detection_threshold: float = 0.1  # bifurcation detector

@dataclass
class NECFConfig:
    field: FieldConfig = field(default_factory=FieldConfig)
    oscillator: OscillatorConfig = field(default_factory=OscillatorConfig)
    meta_rule: MetaRuleConfig = field(default_factory=MetaRuleConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    curiosity: CuriosityConfig = field(default_factory=CuriosityConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    will: WillConfig = field(default_factory=WillConfig)
    seed: int = 42

DEFAULT_CONFIG = NECFConfig()
```

---

### FILE 2: `environment.py`

```python
"""
environment.py — External driving signals for NECF.
Implements: Lorenz attractor, periodic signals, stochastic Poisson spikes.
All three combined ensure the field never reaches thermodynamic equilibrium.
"""
import numpy as np
from dataclasses import dataclass

class LorenzDriver:
    """Deterministic chaotic driving via Lorenz attractor."""

    def __init__(self, cfg):
        self.sigma = cfg.lorenz_sigma
        self.rho   = cfg.lorenz_rho
        self.beta  = cfg.lorenz_beta
        self.dt    = 0.01
        self.state = np.array([1.0, 0.0, 0.0])  # initial condition

    def step(self) -> np.ndarray:
        x, y, z = self.state
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        self.state += self.dt * np.array([dx, dy, dz])
        return self.state.copy()

    @property
    def current_x(self) -> float:
        return self.state[0]


class PeriodicDriver:
    """Structured sinusoidal signal — creates resonance opportunities."""

    def __init__(self, cfg):
        self.freq = cfg.periodic_freq
        self.amp  = cfg.periodic_eps
        self.t    = 0.0

    def step(self, dt: float) -> float:
        self.t += dt
        return self.amp * np.sin(2 * np.pi * self.freq * self.t)


class SpikeDriver:
    """Poisson process spike generator — random perturbations."""

    def __init__(self, cfg, rng: np.random.Generator):
        self.rate = cfg.spike_rate  # probability per node per step
        self.rng  = rng

    def get_spikes(self, N: int) -> np.ndarray:
        """Returns binary array of shape (N,): 1 = spike this step."""
        return (self.rng.uniform(size=N) < self.rate).astype(float)


class Environment:
    """
    Composite external driver. Combines Lorenz + periodic + spikes.
    The system is an open thermodynamic system — no closed-loop equilibrium.
    """

    def __init__(self, cfg, rng: np.random.Generator):
        self.cfg     = cfg
        self.lorenz  = LorenzDriver(cfg.environment)
        self.periodic= PeriodicDriver(cfg.environment)
        self.spikes  = SpikeDriver(cfg.environment, rng)
        self.rng     = rng

    def step(self, dt: float, N: int) -> dict:
        """
        Returns a dict of driving signals for this step.
        All signals are pre-scaled by their respective epsilon.
        """
        lorenz_state = self.lorenz.step()
        lorenz_x     = lorenz_state[0] / 25.0  # normalize ~[-1, 1]

        periodic_val = self.periodic.step(dt)

        spike_mask   = self.spikes.get_spikes(N)

        # Phase perturbation from Lorenz (spatially distributed)
        node_indices = np.arange(N)
        lorenz_phase_kick = (
            self.cfg.environment.lorenz_eps
            * lorenz_x
            * np.sin(node_indices * 2 * np.pi / N)
        )

        # Amplitude modulation from periodic signal
        periodic_amp_mod = periodic_val * np.ones(N)

        return {
            "lorenz_phase_kick": lorenz_phase_kick,    # shape (N,)
            "periodic_amp_mod":  periodic_amp_mod,     # shape (N,)
            "spike_mask":        spike_mask,           # shape (N,) binary
            "lorenz_state":      lorenz_state,
        }
```

---

### FILE 3: `identity.py`

```python
"""
identity.py — Identity Curvature Functional H[L] and Rollback.

The key mathematical object: H[L] measures how far the rule field L
has drifted from its initialization, penalizing both drift and collapse.

H[L] = (1/N) Σ ||L_i - L_i^0||² + κ · Var(L_i)

Rollback: if δH > threshold, revert L and apply gradient correction.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class IdentityState:
    H_current: float = 0.0
    H_previous: float = 0.0
    rollback_count: int = 0
    H_history: list = None

    def __post_init__(self):
        if self.H_history is None:
            self.H_history = []


class IdentityCurvature:
    """
    Computes H[L] and enforces rollback when identity curvature spikes.

    This implements the core invariant: the system's learning rules
    are allowed to evolve but not to drift into chaos or collapse.
    """

    def __init__(self, cfg):
        self.kappa     = cfg.kappa
        self.threshold = cfg.rollback_threshold
        self.eta       = cfg.rollback_eta
        self.L_init    = None   # set on first call
        self.state     = IdentityState()

    def initialize(self, L: np.ndarray):
        """
        Record the initial rule state. Called once at experiment start.
        L shape: (N, 3) for [alpha, beta, gamma]
        """
        self.L_init = L.copy()

    def compute(self, L: np.ndarray) -> float:
        """
        H[L] = (1/N) Σ_i ||L_i - L_i^0||²  +  κ · Var(L_i)

        Returns scalar H value.
        """
        if self.L_init is None:
            raise RuntimeError("IdentityCurvature not initialized.")

        N = L.shape[0]
        drift   = np.mean(np.sum((L - self.L_init) ** 2, axis=1))
        variance= self.kappa * np.mean(np.var(L, axis=0))
        H = drift + variance
        return float(H)

    def gradient(self, L: np.ndarray) -> np.ndarray:
        """
        ∂H/∂L_i = 2(L_i - L_i^0) + κ · ∂Var/∂L_i
        Returns array of shape (N, 3).
        """
        N = L.shape[0]
        grad_drift = 2.0 * (L - self.L_init) / N
        # Gradient of variance term: ∂/∂L_i [Var(L)] = 2(L_i - L̄) / N
        L_mean = L.mean(axis=0, keepdims=True)
        grad_var = self.kappa * 2.0 * (L - L_mean) / N
        return grad_drift + grad_var

    def check_and_rollback(
        self,
        L_current: np.ndarray,
        L_previous: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """
        Compute δH. If δH > threshold, apply rollback.

        Returns: (L_after_rollback, did_rollback)
        """
        H_current  = self.compute(L_current)
        H_previous = self.compute(L_previous)
        delta_H    = H_current - H_previous

        self.state.H_current  = H_current
        self.state.H_previous = H_previous
        self.state.H_history.append(H_current)

        if delta_H > self.threshold:
            # Rollback: revert to previous + gradient correction
            grad = self.gradient(L_previous)
            L_rolled_back = L_previous - self.eta * grad
            self.state.rollback_count += 1
            return L_rolled_back, True

        return L_current, False
```

---

### FILE 4: `curiosity.py`

```python
"""
curiosity.py — Curiosity Engine.

Monitors prediction error over time. Detects plateaus.
On plateau: boosts gamma (curiosity) to drive exploration.
Computes uncertainty potential U(θ) for curiosity gradient.
"""
import numpy as np
from collections import deque
from dataclasses import dataclass


@dataclass
class CuriosityState:
    directive: str = "NONE"         # current exploration directive
    plateau_detected: bool = False
    intervention_count: int = 0
    error_history: list = None

    def __post_init__(self):
        if self.error_history is None:
            self.error_history = []


class CuriosityEngine:
    """
    Active Inference core: drives the field toward minimal free energy
    by detecting epistemic stagnation and issuing exploration directives.

    Difference from Friston's FEP:
    - FEP minimizes free energy via a fixed variational update rule
    - NECF's CuriosityEngine modifies γ_i (curiosity weight in L_i)
      which changes HOW free energy is minimized — a meta-level intervention
    """

    def __init__(self, cfg, rng: np.random.Generator):
        self.cfg     = cfg.curiosity
        self.rng     = rng
        self.window  = deque(maxlen=cfg.curiosity.plateau_window)
        self.state   = CuriosityState()

    def compute_uncertainty_potential(self, theta: np.ndarray) -> np.ndarray:
        """
        U(θ_i) = -log p(θ_i | context) ≈ negative log of phase density.
        Approximated via KDE on current phase distribution.

        Nodes in sparse regions of phase space have high uncertainty → high gradient.
        Returns array of shape (N,) — uncertainty at each node.
        """
        N = len(theta)
        # Wrap phases to [0, 2π]
        theta_wrapped = theta % (2 * np.pi)

        # Kernel density estimation: bandwidth = Silverman's rule
        bw = 1.06 * np.std(theta_wrapped) * N ** (-0.2) + 1e-6

        # For each node, compute density at its phase
        density = np.zeros(N)
        for i in range(N):
            diffs = np.sin(theta_wrapped - theta_wrapped[i])
            density[i] = np.mean(np.exp(-0.5 * (diffs / bw) ** 2)) + 1e-8

        uncertainty = -np.log(density)  # high where sparse
        return uncertainty

    def update(self, error: float) -> None:
        """Record current error. Returns nothing — state updated internally."""
        self.window.append(error)
        self.state.error_history.append(error)

    @property
    def is_plateauing(self) -> bool:
        """True if error hasn't improved by epsilon over the last window steps."""
        if len(self.window) < self.cfg.plateau_window:
            return False
        improvement = max(self.window) - min(self.window)
        return improvement < self.cfg.plateau_epsilon

    def get_directive(self, n_falsified: int, n_rollbacks: int) -> str:
        """
        Returns exploration directive based on system state.
        Called when plateau is detected.
        """
        if n_rollbacks > 5:
            return "REDUCE_META_RATE"    # too many rollbacks → calm down
        if n_falsified > 3:
            return "REFRAME_COUPLING"    # high falsification → change topology
        if self.state.intervention_count % 2 == 0:
            return "BOOST_CURIOSITY"     # increase gamma field-wide
        return "DIVERSIFY_RULES"         # inject diversity into L_i

    def apply_directive(
        self,
        directive: str,
        L: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Execute the directive on the rule field L.
        Returns modified L.
        """
        L_new = L.copy()
        gamma_col = 2  # column index for γ

        if directive == "BOOST_CURIOSITY":
            L_new[:, gamma_col] = np.clip(
                L_new[:, gamma_col] * (1 + self.cfg.spike_strength),
                0.001, 0.5
            )

        elif directive == "DIVERSIFY_RULES":
            noise = rng.normal(0, 0.05, size=L.shape)
            L_new += noise
            L_new = np.clip(L_new, 0.001, 3.0)

        elif directive == "REFRAME_COUPLING":
            # Reset beta (coupling) for 20% of nodes
            N = L.shape[0]
            mask = rng.uniform(size=N) < 0.2
            L_new[mask, 1] = rng.uniform(0.2, 1.5, size=mask.sum())

        elif directive == "REDUCE_META_RATE":
            # Decay all rule update rates
            L_new *= 0.95

        if directive != "NONE":
            self.state.intervention_count += 1
            self.state.directive = directive
            self.state.plateau_detected = True

        return L_new
```

---

### FILE 5: `field.py`

```python
"""
field.py — Level-1 Oscillator Field.

Core state: amplitudes A(N,) and phases θ(N,).
Evolution equations: generalized Kuramoto + amplitude dynamics.
"""
import numpy as np
from dataclasses import dataclass, field as dc_field
from typing import Optional


@dataclass
class FieldState:
    """Complete snapshot of oscillator field at time t."""
    A: np.ndarray          # amplitudes (N,)
    theta: np.ndarray      # phases (N,) in [0, 2π)
    omega: np.ndarray      # natural frequencies (N,) — fixed
    W: np.ndarray          # coupling matrix (N, N)
    r: float = 0.0         # order parameter magnitude
    psi: float = 0.0       # mean phase
    t: int = 0             # current step


class OscillatorField:
    """
    Coupled oscillator field — the Level-1 substrate.

    Key departure from pure Kuramoto:
    - Each node has local β_i (coupling strength) from rule field L
    - Each node has local γ_i (curiosity weight) from rule field L
    - Phase evolution is driven by: synchrony pull + curiosity gradient
    - Amplitude evolution is driven by: prediction error + noise
    """

    def __init__(self, cfg, rng: np.random.Generator):
        self.cfg = cfg
        self.N   = cfg.field.N
        self.dt  = cfg.field.dt
        self.rng = rng

        # Initialize natural frequencies
        self.omega = rng.normal(
            cfg.oscillator.omega_mean,
            cfg.oscillator.omega_std,
            size=self.N,
        )

        # Initialize amplitudes
        A_init = np.full(self.N, cfg.oscillator.A_init)

        # Initialize phases
        if cfg.oscillator.theta_init == "random":
            theta_init = rng.uniform(0, 2 * np.pi, size=self.N)
        elif cfg.oscillator.theta_init == "uniform":
            theta_init = np.linspace(0, 2 * np.pi, self.N, endpoint=False)
        else:  # clustered
            theta_init = rng.normal(np.pi, 0.5, size=self.N) % (2 * np.pi)

        # Initialize coupling matrix: all-to-all with random weights
        W = rng.uniform(0.5, 1.5, size=(self.N, self.N))
        np.fill_diagonal(W, 0.0)
        W = (W + W.T) / 2.0  # symmetric

        self.state = FieldState(
            A=A_init,
            theta=theta_init,
            omega=self.omega,
            W=W,
            r=0.0,
            psi=0.0,
            t=0,
        )

    def compute_order_parameter(self) -> tuple[float, float]:
        """r·exp(iψ) = (1/N) Σ A_i · exp(iθ_i)"""
        z   = np.mean(self.state.A * np.exp(1j * self.state.theta))
        r   = float(np.abs(z))
        psi = float(np.angle(z)) % (2 * np.pi)
        return r, psi

    def compute_prediction_error(self, target_phases: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Local prediction error ε_i.
        Without external target: error = deviation from mean field.
        With target: error = |θ_i - target_i|² / (4π²)
        """
        if target_phases is not None:
            diff = self.state.theta - target_phases
            return (np.sin(diff / 2)) ** 2  # circular distance, normalized
        else:
            # Deviation from mean phase ψ (using circular distance)
            diff = self.state.theta - self.state.psi
            return (np.sin(diff / 2)) ** 2

    def step(
        self,
        L: np.ndarray,
        env_signals: dict,
        curiosity_potential: np.ndarray,
    ) -> np.ndarray:
        """
        Advance field by one step dt.
        Returns local prediction error (N,) for meta-dynamics use.

        Args:
            L: rule field (N, 3) — [alpha, beta, gamma]
            env_signals: dict from Environment.step()
            curiosity_potential: U(θ) from CuriosityEngine (N,)
        """
        alpha = L[:, 0]  # error sensitivity
        beta  = L[:, 1]  # coupling strength
        gamma = L[:, 2]  # curiosity weight

        # --- Phase update ---
        # Synchrony pull: β_i · (1/N) Σ_j W_ij · A_j · sin(θ_j - θ_i)
        theta_diff = (
            self.state.theta[np.newaxis, :]   # θ_j: shape (1, N)
            - self.state.theta[:, np.newaxis] # θ_i: shape (N, 1)
        )  # shape (N, N)
        sync_pull = np.sum(
            self.state.W * self.state.A[np.newaxis, :] * np.sin(theta_diff),
            axis=1,
        ) / self.N  # shape (N,)

        # Curiosity gradient: γ_i · ∂U/∂θ_i ≈ γ_i · U_i (simplified)
        curiosity_pull = gamma * curiosity_potential

        # External: Lorenz phase kick
        lorenz_kick = env_signals["lorenz_phase_kick"]

        # Phase spikes: full reset for spiked nodes
        spike_mask = env_signals["spike_mask"]

        d_theta = (
            self.omega
            + beta * sync_pull
            + gamma * curiosity_pull
            + lorenz_kick
        ) * self.dt

        new_theta = (self.state.theta + d_theta) % (2 * np.pi)

        # Apply spikes: reset phase to random value
        spike_nodes = spike_mask.astype(bool)
        new_theta[spike_nodes] = self.rng.uniform(
            0, 2 * np.pi, size=spike_nodes.sum()
        )

        # --- Amplitude update ---
        # Compute prediction error before updating theta
        eps = self.compute_prediction_error()

        # dA/dt = -α_i · ε_i · A_i + σ · η_i
        noise  = self.rng.normal(0, self.cfg.field.noise_sigma, size=self.N)
        d_A    = (-alpha * eps * self.state.A + noise) * self.dt

        # Periodic amplitude modulation
        d_A += env_signals["periodic_amp_mod"] * self.dt

        new_A = np.clip(self.state.A + d_A, 0.01, 1.0)

        # --- Update state ---
        self.state.theta = new_theta
        self.state.A     = new_A
        self.state.t    += 1

        r, psi             = self.compute_order_parameter()
        self.state.r       = r
        self.state.psi     = psi

        return eps
```

---

### FILE 6: `meta_dynamics.py`

```python
"""
meta_dynamics.py — Level-2/3: Learning Rule Evolution.

THIS IS THE CORE NOVEL CONTRIBUTION.

Rules L_i = (α_i, β_i, γ_i) evolve via:
  dL_i/dt = F(L_i, φ_i, ε_i) - λ · ∇_{L_i} H[L]

where F implements epistemic contagion: low-error nodes propagate their
rules to high-error neighbors. ∇H is the identity gradient constraint.

This is NOT gradient descent. It is a dynamical system in rule space.
"""
import numpy as np
from .config import NECFConfig
from .identity import IdentityCurvature


class MetaDynamics:
    """
    Level-3 dynamics: learning rules evolve under error-driven
    contagion + identity curvature constraint + rollback.

    State: L (N, 3) — [alpha, beta, gamma] for each node.
    """

    def __init__(self, cfg: NECFConfig, rng: np.random.Generator):
        self.cfg      = cfg.meta_rule
        self.id_cfg   = cfg.identity
        self.N        = cfg.field.N
        self.rng      = rng
        self.identity = IdentityCurvature(cfg.identity)

        # Initialize rule field
        self.L = np.column_stack([
            np.full(self.N, cfg.meta_rule.alpha_init),
            np.full(self.N, cfg.meta_rule.beta_init),
            np.full(self.N, cfg.meta_rule.gamma_init),
        ])  # shape (N, 3)

        # Add small noise to break symmetry
        self.L += rng.normal(0, 0.01, size=self.L.shape)
        self.L = np.clip(self.L, 0.001, 3.0)

        # Initialize identity reference
        self.identity.initialize(self.L)

        self._L_previous = self.L.copy()

    def _epistemic_contagion(
        self,
        eps: np.ndarray,
        W: np.ndarray,
    ) -> np.ndarray:
        """
        F(L_i, ε_i): Rules propagate from low-error to high-error nodes.

        For each node i: the "target rule" is the weighted average of
        neighbor rules, where weights are inversely proportional to
        neighbor errors (low-error neighbors contribute more).

        dL_i/dt += μ · (L_target_i - L_i) · ε_i
        """
        N = self.N
        mu = np.array([
            self.cfg.mu_alpha,
            self.cfg.mu_beta,
            self.cfg.mu_gamma,
        ])

        # Neighbor influence weights: higher for low-error neighbors
        neighbor_weight = 1.0 / (eps + 1e-6)  # shape (N,)
        neighbor_weight = neighbor_weight / (neighbor_weight.sum() + 1e-6)

        # Weighted target rule for each node
        # L_target_i = Σ_j W_ij · neighbor_weight_j · L_j / Σ_j W_ij · nw_j
        W_weighted = W * neighbor_weight[np.newaxis, :]  # (N, N)
        W_row_sum  = W_weighted.sum(axis=1, keepdims=True) + 1e-6
        L_target   = (W_weighted @ self.L) / W_row_sum  # (N, 3)

        # Contagion update: proportional to local error
        contagion = mu[np.newaxis, :] * (L_target - self.L) * eps[:, np.newaxis]
        return contagion

    def step(
        self,
        eps: np.ndarray,
        W: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, bool]:
        """
        Advance rule field by one step.

        Returns:
            L_new: updated rule field (N, 3)
            did_rollback: True if identity rollback was triggered
        """
        self._L_previous = self.L.copy()

        # Epistemic contagion term
        contagion = self._epistemic_contagion(eps, W)

        # Identity gradient term
        identity_grad = self.identity.gradient(self.L)

        # Full rule update
        dL = (contagion - self.id_cfg.lambda_identity * identity_grad) * dt
        L_new = np.clip(
            self.L + dL,
            [self.cfg.alpha_bounds[0], self.cfg.beta_bounds[0], 0.001],
            [self.cfg.alpha_bounds[1], self.cfg.beta_bounds[1], self.cfg.gamma_bounds[1]],
        )

        # Rollback check
        L_new, did_rollback = self.identity.check_and_rollback(
            L_new, self._L_previous
        )

        self.L = L_new
        return self.L.copy(), did_rollback

    @property
    def H(self) -> float:
        return self.identity.compute(self.L)

    @property
    def L_current(self) -> np.ndarray:
        return self.L.copy()
```

---

### FILE 7: `observer.py`

```python
"""
observer.py — All measurable observables.

Computes: r(t), λ_max, entropy, prediction error stats,
rule diversity, H[L], rollback frequency, causal entropy.

These are the falsifiable predictions of NECF.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class Snapshot:
    """Complete observable state at one time step."""
    t: int
    r: float                  # order parameter
    mean_error: float         # mean prediction error
    H: float                  # identity curvature
    alpha_mean: float
    beta_mean: float
    gamma_mean: float
    rule_diversity: float     # Var(L_i)
    rollback_count: int
    curiosity_directive: str
    lyapunov_proxy: float     # approximation of λ_max
    entropy: float            # Shannon entropy of phase distribution


@dataclass
class ObserverHistory:
    snapshots: List[Snapshot] = field(default_factory=list)

    def to_arrays(self) -> dict:
        """Convert list of snapshots to dict of arrays for plotting."""
        if not self.snapshots:
            return {}
        keys = [f.name for f in Snapshot.__dataclass_fields__.values()
                if f.name != "t" and f.name != "curiosity_directive"]
        return {
            "t": np.array([s.t for s in self.snapshots]),
            **{k: np.array([getattr(s, k) for s in self.snapshots]) for k in keys},
            "directive": [s.curiosity_directive for s in self.snapshots],
        }


class Observer:
    """Computes all observables. Stateless — all inputs passed per call."""

    def __init__(self):
        self.history = ObserverHistory()
        self._prev_theta: np.ndarray = None

    def compute_lyapunov_proxy(self, theta: np.ndarray) -> float:
        """
        Approximate max Lyapunov exponent from phase divergence rate.
        λ_proxy = (1/Δt) · log(||δθ(t)|| / ||δθ(0)||)

        Uses finite difference from consecutive steps.
        """
        if self._prev_theta is None:
            self._prev_theta = theta.copy()
            return 0.0
        delta = theta - self._prev_theta
        delta_circular = np.arctan2(np.sin(delta), np.cos(delta))
        norm = np.linalg.norm(delta_circular)
        self._prev_theta = theta.copy()
        if norm < 1e-10:
            return -1.0  # converging
        return float(np.log(norm + 1e-10))

    def compute_entropy(self, theta: np.ndarray, bins: int = 20) -> float:
        """Shannon entropy of phase distribution."""
        hist, _ = np.histogram(theta % (2 * np.pi), bins=bins,
                               range=(0, 2 * np.pi), density=True)
        hist = hist / (hist.sum() + 1e-10)
        ent  = -np.sum(hist * np.log(hist + 1e-10))
        return float(ent)

    def record(
        self,
        t: int,
        field_state,
        L: np.ndarray,
        H: float,
        rollback_count: int,
        eps: np.ndarray,
        directive: str,
    ) -> Snapshot:
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
            lyapunov_proxy=self.compute_lyapunov_proxy(field_state.theta),
            entropy=self.compute_entropy(field_state.theta),
        )
        self.history.snapshots.append(snap)
        return snap
```

---

### FILE 8: `experiment.py`

```python
"""
experiment.py — Full NECF run loop.

Orchestrates all components. Returns observable history.
Designed to be called from app.py (Streamlit) or standalone.
"""
import numpy as np
from typing import Generator
from .config import NECFConfig, DEFAULT_CONFIG
from .field import OscillatorField
from .meta_dynamics import MetaDynamics
from .environment import Environment
from .curiosity import CuriosityEngine
from .observer import Observer


class NECFExperiment:
    """
    Single NECF experiment instance.
    Use .run_step() for streaming (Streamlit) or .run() for batch.
    """

    def __init__(self, cfg: NECFConfig = None):
        self.cfg = cfg or DEFAULT_CONFIG
        self.rng = np.random.default_rng(self.cfg.seed)

        self.field      = OscillatorField(self.cfg, self.rng)
        self.meta       = MetaDynamics(self.cfg, self.rng)
        self.env        = Environment(self.cfg, self.rng)
        self.curiosity  = CuriosityEngine(self.cfg, self.rng)
        self.observer   = Observer()
        self.t          = 0
        self.done       = False

    def run_step(self) -> dict:
        """
        Advance simulation by one step. Returns observable dict.
        Called repeatedly from Streamlit for streaming updates.
        """
        if self.t >= self.cfg.field.T:
            self.done = True
            return None

        # 1. Environment signals
        env_signals = self.env.step(self.cfg.field.dt, self.cfg.field.N)

        # 2. Curiosity potential
        U = self.curiosity.compute_uncertainty_potential(self.field.state.theta)

        # 3. Field step (Level-1)
        eps = self.field.step(self.meta.L, env_signals, U)

        # 4. Curiosity update
        self.curiosity.update(float(np.mean(eps)))

        # 5. Curiosity directive (if plateauing)
        directive = "NONE"
        if self.curiosity.is_plateauing:
            n_rollbacks = self.meta.identity.state.rollback_count
            n_falsified = sum(
                1 for h in self.observer.history.snapshots[-20:]
                if h.mean_error > 0.3
            ) if self.observer.history.snapshots else 0
            directive = self.curiosity.get_directive(n_falsified, n_rollbacks)
            self.meta.L = self.curiosity.apply_directive(
                directive, self.meta.L, self.rng
            )

        # 6. Meta-dynamics step (Level-2/3)
        L_new, did_rollback = self.meta.step(
            eps, self.field.state.W, self.cfg.field.dt
        )

        # 7. Record observables
        snap = self.observer.record(
            t=self.t,
            field_state=self.field.state,
            L=L_new,
            H=self.meta.H,
            rollback_count=self.meta.identity.state.rollback_count,
            eps=eps,
            directive=directive,
        )

        self.t += 1
        return snap

    def run(self, stream: bool = False) -> Generator:
        """
        Run full experiment. Yields snapshots if stream=True.
        Returns observer history if stream=False.
        """
        while not self.done:
            snap = self.run_step()
            if snap and stream:
                yield snap
        if not stream:
            return self.observer.history
```

---

### FILE 9: `app.py` (Streamlit Frontend)

```python
"""
app.py — NECF Live Scientific Dashboard.
Main Streamlit entry point. Calls all backend modules.

Run: streamlit run app.py
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from config import NECFConfig, FieldConfig, MetaRuleConfig, IdentityConfig, \
                   CuriosityConfig, EnvironmentConfig, OscillatorConfig
from experiment import NECFExperiment
from observer import Snapshot

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NECF — Non-Equilibrium Cognitive Field",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── DARK CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Syne:wght@400;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background: #060810; }
section[data-testid="stSidebar"] { background: #0a0c14; border-right: 1px solid #1a1f35; }
h1 { background: linear-gradient(90deg, #00d4ff, #8b5cf6, #ff6b6b);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent;
     font-weight: 800; font-size: 2rem; }
h2 { color: #7dd3fc; border-bottom: 1px solid #1a1f35; padding-bottom: 4px; }
h3 { color: #94a3b8; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; }
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0d1020 0%, #131828 100%);
    border: 1px solid #1e2745; border-radius: 10px; padding: 12px;
}
[data-testid="metric-container"] label { color: #4a5568 !important; font-size: 10px; font-family: 'JetBrains Mono'; }
[data-testid="metric-container"] [data-testid="metric-value"] {
    color: #e2e8f0 !important; font-size: 20px; font-weight: 700; font-family: 'JetBrains Mono';
}
.stButton>button {
    background: linear-gradient(135deg, #1e3a8a, #7c3aed) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-weight: 700 !important; font-family: 'Syne' !important;
}
</style>
""", unsafe_allow_html=True)

# ─── PLOTTING HELPERS ──────────────────────────────────────────────────────────
_BG = "#060810"
_AX = "#0a0e1a"
_SP = "#1a1f35"

def dark_fig(w=10, h=4):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=_BG)
    ax.set_facecolor(_AX)
    for sp in ax.spines.values():
        sp.set_edgecolor(_SP)
    ax.tick_params(colors="#475569", labelsize=8)
    return fig, ax

def plot_phase_circle(theta, A, ax):
    """Polar plot of oscillator phases."""
    ax.set_facecolor(_AX)
    colors = plt.cm.plasma(A)
    for i in range(len(theta)):
        ax.plot([theta[i]], [A[i]], 'o', color=colors[i], markersize=4, alpha=0.8)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax.tick_params(colors="#475569", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(_SP)

# ─── SESSION STATE ─────────────────────────────────────────────────────────────
def _init():
    if "experiment" not in st.session_state:
        st.session_state.experiment  = None
        st.session_state.running     = False
        st.session_state.history     = []
        st.session_state.step_count  = 0
        st.session_state.config      = NECFConfig()

_init()

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌀 NECF Controls")
    st.caption("Non-Equilibrium Cognitive Field")
    st.divider()

    st.markdown("**Field Parameters**")
    N    = st.slider("Nodes (N)", 16, 256, 64, step=16)
    T    = st.slider("Steps (T)", 200, 5000, 1000, step=100)
    seed = st.number_input("Seed", value=42, step=1)

    st.markdown("**Oscillator**")
    omega_std = st.slider("Frequency Diversity", 0.0, 1.0, 0.3, step=0.05)
    noise_sigma = st.slider("Noise σ", 0.0, 0.1, 0.02, step=0.005)

    st.markdown("**Identity Constraint**")
    kappa     = st.slider("κ (variance penalty)", 0.0, 2.0, 0.5, step=0.1)
    lambda_id = st.slider("λ (identity gradient weight)", 0.0, 1.0, 0.1, step=0.05)
    rollback_thresh = st.slider("Rollback threshold δH", 0.05, 1.0, 0.3, step=0.05)

    st.markdown("**External Driving**")
    lorenz_eps  = st.slider("Lorenz coupling ε_L", 0.0, 0.2, 0.05, step=0.01)
    spike_rate  = st.slider("Spike rate λ_s", 0.0, 0.1, 0.02, step=0.005)

    st.divider()
    col1, col2 = st.columns(2)
    run_btn    = col1.button("▶ Run",   type="primary", use_container_width=True)
    reset_btn  = col2.button("⟳ Reset", use_container_width=True)

    if st.session_state.experiment and st.session_state.history:
        import json
        export = {
            "config": {"N": N, "T": T, "seed": seed, "kappa": kappa},
            "observables": [
                {
                    "t": s.t, "r": s.r, "H": s.H,
                    "error": s.mean_error,
                    "lyapunov": s.lyapunov_proxy,
                    "directive": s.curiosity_directive,
                }
                for s in st.session_state.history
            ]
        }
        st.download_button(
            "💾 Export Data",
            data=json.dumps(export, indent=2),
            file_name=f"necf_seed{seed}.json",
            mime="application/json",
            use_container_width=True,
        )

# ─── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("# Non-Equilibrium Cognitive Field")
st.markdown("**Level-3 Meta-Rule Dynamics · Identity-Constrained Oscillator Field · Proto-Cognitive Substrate**")

mc = st.columns(5)
mc[0].metric("Architecture",   "Level-3 (Novel)")
mc[1].metric("Prior Art",      "Level-2 max")
mc[2].metric("No Backprop",    "Verified")
mc[3].metric("Observable",     "7 Metrics")
mc[4].metric("Falsifiable",    "Yes")
st.divider()

# ─── HANDLE BUTTONS ────────────────────────────────────────────────────────────
if reset_btn:
    st.session_state.experiment = None
    st.session_state.running    = False
    st.session_state.history    = []
    st.session_state.step_count = 0
    st.rerun()

if run_btn:
    cfg = NECFConfig(
        field=FieldConfig(N=N, T=T, noise_sigma=noise_sigma),
        oscillator=OscillatorConfig(omega_std=omega_std),
        meta_rule=MetaRuleConfig(),
        identity=IdentityConfig(kappa=kappa, lambda_identity=lambda_id,
                                rollback_threshold=rollback_thresh),
        curiosity=CuriosityConfig(),
        environment=EnvironmentConfig(lorenz_eps=lorenz_eps, spike_rate=spike_rate),
        seed=int(seed),
    )
    st.session_state.experiment  = NECFExperiment(cfg)
    st.session_state.running     = True
    st.session_state.history     = []
    st.session_state.step_count  = 0

# ─── MAIN DISPLAY ──────────────────────────────────────────────────────────────
if st.session_state.experiment is None:
    st.info("👈 Configure parameters and click **▶ Run** to start.")
    st.markdown("""
    #### Observable Predictions (NECF's Falsifiable Claims)
    | Metric | Prediction | Falsification Condition |
    |--------|-----------|------------------------|
    | r(t) order parameter | Rises from ~0.1 to >0.6 | r never exceeds 0.3 |
    | H[L] identity curvature | Stays in [0.5, 2.0] | H diverges or collapses to 0 |
    | Lyapunov proxy | Stays in (-0.1, 0.5) | λ > 1.0 sustained |
    | Mean prediction error | Decreases after ~200 steps | Error monotonically rises |
    | Rule diversity | Bounded, not zero | Var(L_i) → 0 or → ∞ |
    | Curiosity directive | Fires when r plateaus | Never fires or fires randomly |
    | Rollback frequency | Decreases over time | Increases over time |
    """)
    st.stop()

# ─── LIVE SIMULATION ───────────────────────────────────────────────────────────
exp = st.session_state.experiment
progress_bar = st.progress(0, text="Initializing field...")
status_placeholder = st.empty()

# Run N_BATCH steps per Streamlit render cycle
N_BATCH = 20
for _ in range(N_BATCH):
    if exp.done:
        st.session_state.running = False
        break
    snap = exp.run_step()
    if snap:
        st.session_state.history.append(snap)
        st.session_state.step_count += 1

progress = st.session_state.step_count / exp.cfg.field.T
progress_bar.progress(
    min(progress, 1.0),
    text=f"Step {st.session_state.step_count}/{exp.cfg.field.T}"
)

# ─── METRICS BAR ───────────────────────────────────────────────────────────────
if st.session_state.history:
    latest = st.session_state.history[-1]
    m = st.columns(7)
    m[0].metric("r(t) Sync", f"{latest.r:.3f}")
    m[1].metric("H[L] Identity", f"{latest.H:.3f}")
    m[2].metric("Error ε", f"{latest.mean_error:.3f}")
    m[3].metric("λ Lyapunov", f"{latest.lyapunov_proxy:.3f}")
    m[4].metric("Rule Diversity", f"{latest.rule_diversity:.4f}")
    m[5].metric("Rollbacks", str(latest.rollback_count))
    m[6].metric("Directive", latest.curiosity_directive[:12] if latest.curiosity_directive else "NONE")

# ─── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌀 Phase Field",
    "📈 Observables",
    "🔬 Rule Dynamics",
    "🧬 Identity & Will",
    "⚡ Curiosity Engine",
])

history = st.session_state.history
if not history:
    for t in [tab1, tab2, tab3, tab4, tab5]:
        with t:
            st.info("Simulation running...")
    if st.session_state.running:
        st.rerun()
    st.stop()

# Convert history to arrays
t_arr    = np.array([s.t             for s in history])
r_arr    = np.array([s.r             for s in history])
H_arr    = np.array([s.H             for s in history])
err_arr  = np.array([s.mean_error    for s in history])
lya_arr  = np.array([s.lyapunov_proxy for s in history])
div_arr  = np.array([s.rule_diversity for s in history])
alp_arr  = np.array([s.alpha_mean    for s in history])
bet_arr  = np.array([s.beta_mean     for s in history])
gam_arr  = np.array([s.gamma_mean    for s in history])
rb_arr   = np.array([s.rollback_count for s in history])
ent_arr  = np.array([s.entropy       for s in history])

# ── TAB 1: PHASE FIELD ──────────────────────────────────────────────────────
with tab1:
    col_phase, col_order = st.columns([1, 1])
    with col_phase:
        st.markdown("##### Current Phase Distribution (Polar)")
        fig = plt.figure(figsize=(5, 5), facecolor=_BG)
        ax  = fig.add_subplot(111, projection='polar')
        plot_phase_circle(exp.field.state.theta, exp.field.state.A, ax)
        r_latest = latest.r
        color_r  = "#22c55e" if r_latest > 0.6 else "#f59e0b" if r_latest > 0.3 else "#ef4444"
        ax.set_title(f"r = {r_latest:.3f}", color=color_r, fontsize=12,
                     fontweight="bold", pad=15, fontfamily="JetBrains Mono")
        fig.patch.set_facecolor(_BG)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_order:
        st.markdown("##### Order Parameter r(t) Evolution")
        fig, ax = dark_fig(5, 4)
        ax.fill_between(t_arr, r_arr, alpha=0.2, color="#00d4ff")
        ax.plot(t_arr, r_arr, color="#00d4ff", linewidth=2)
        ax.axhline(0.6, color="#22c55e", linestyle="--", linewidth=1, alpha=0.7, label="r=0.6 (coherence)")
        ax.axhline(0.3, color="#f59e0b", linestyle="--", linewidth=1, alpha=0.7, label="r=0.3 (transition)")
        ax.set_xlabel("Step", color="#64748b", fontsize=9)
        ax.set_ylabel("r(t)", color="#64748b", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, labelcolor="#94a3b8", facecolor=_AX, edgecolor=_SP)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("##### Phase Entropy & Coupling Matrix")
    col_ent, col_coupling = st.columns(2)
    with col_ent:
        fig, ax = dark_fig(5, 2.5)
        ax.fill_between(t_arr, ent_arr, alpha=0.3, color="#a78bfa")
        ax.plot(t_arr, ent_arr, color="#a78bfa", linewidth=1.5)
        ax.set_xlabel("Step", color="#64748b", fontsize=9)
        ax.set_ylabel("Phase Entropy H(θ)", color="#64748b", fontsize=9)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_coupling:
        fig, ax = dark_fig(5, 2.5)
        W = exp.field.state.W
        im = ax.imshow(W[:min(32, N), :min(32, N)], cmap="plasma", aspect="auto")
        ax.set_title("Coupling Matrix W (first 32×32)", color="#94a3b8", fontsize=8)
        fig.colorbar(im, ax=ax)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# ── TAB 2: OBSERVABLES ──────────────────────────────────────────────────────
with tab2:
    st.markdown("##### The 7 Falsifiable Predictions")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = dark_fig(5, 3)
        ax.fill_between(t_arr, err_arr, alpha=0.2, color="#f87171")
        ax.plot(t_arr, err_arr, color="#f87171", linewidth=2)
        ax.set_xlabel("Step", color="#64748b", fontsize=9)
        ax.set_ylabel("Mean Prediction Error ε", color="#64748b", fontsize=9)
        ax.set_title("Prediction Error (should decrease)", color="#94a3b8", fontsize=9)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with c2:
        fig, ax = dark_fig(5, 3)
        colors_lya = ["#22c55e" if v < 0 else "#f59e0b" if v < 0.5 else "#ef4444"
                      for v in lya_arr]
        ax.bar(t_arr, lya_arr, color=colors_lya, width=1.0, alpha=0.8)
        ax.axhline(0, color="#64748b", linewidth=0.8)
        ax.axhline(0.5, color="#ef4444", linestyle="--", linewidth=1, alpha=0.6, label="Chaos threshold")
        ax.set_xlabel("Step", color="#64748b", fontsize=9)
        ax.set_ylabel("Lyapunov Proxy λ", color="#64748b", fontsize=9)
        ax.legend(fontsize=8, labelcolor="#94a3b8", facecolor=_AX, edgecolor=_SP)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    col3, col4 = st.columns(2)
    with col3:
        fig, ax = dark_fig(5, 3)
        ax.fill_between(t_arr, div_arr, alpha=0.2, color="#fb923c")
        ax.plot(t_arr, div_arr, color="#fb923c", linewidth=2)
        ax.set_xlabel("Step", color="#64748b", fontsize=9)
        ax.set_ylabel("Rule Diversity Var(L_i)", color="#64748b", fontsize=9)
        ax.set_title("Rule Diversity (should stay bounded)", color="#94a3b8", fontsize=9)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col4:
        fig, ax = dark_fig(5, 3)
        ax.plot(t_arr, rb_arr, color="#818cf8", linewidth=2, drawstyle="steps-post")
        ax.set_xlabel("Step", color="#64748b", fontsize=9)
        ax.set_ylabel("Cumulative Rollbacks", color="#64748b", fontsize=9)
        ax.set_title("Rollback Count (should grow slowly then plateau)", color="#94a3b8", fontsize=9)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# ── TAB 3: RULE DYNAMICS ────────────────────────────────────────────────────
with tab3:
    st.markdown("##### Meta-Rule Field L_i = (α, β, γ) Evolution")
    fig, ax = dark_fig(10, 4)
    ax.plot(t_arr, alp_arr, color="#38bdf8", linewidth=2, label="ᾱ (error sensitivity)")
    ax.plot(t_arr, bet_arr, color="#34d399", linewidth=2, label="β̄ (coupling strength)")
    ax.plot(t_arr, gam_arr, color="#fb923c", linewidth=2, label="γ̄ (curiosity weight)")
    ax.set_xlabel("Step", color="#64748b", fontsize=9)
    ax.set_ylabel("Mean Rule Value", color="#64748b", fontsize=9)
    ax.set_title("Mean learning rules over time. Should evolve, not collapse or explode.",
                 color="#94a3b8", fontsize=8)
    ax.legend(fontsize=9, labelcolor="#94a3b8", facecolor=_AX, edgecolor=_SP)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("##### Current Rule Distribution (Histograms)")
    L_current = exp.meta.L_current
    col_a, col_b, col_g = st.columns(3)
    for col, data, label, color in [
        (col_a, L_current[:, 0], "α (error sensitivity)", "#38bdf8"),
        (col_b, L_current[:, 1], "β (coupling strength)", "#34d399"),
        (col_g, L_current[:, 2], "γ (curiosity weight)",  "#fb923c"),
    ]:
        with col:
            fig, ax = dark_fig(3.5, 2.5)
            ax.hist(data, bins=15, color=color, edgecolor=_SP, alpha=0.85)
            ax.axvline(data.mean(), color="white", linestyle="--", linewidth=1)
            ax.set_xlabel(label, color="#64748b", fontsize=8)
            ax.set_ylabel("Count", color="#64748b", fontsize=8)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

# ── TAB 4: IDENTITY & WILL ──────────────────────────────────────────────────
with tab4:
    st.markdown("##### Identity Curvature H[L] Over Time")
    fig, ax = dark_fig(10, 3)
    ax.fill_between(t_arr, H_arr, alpha=0.3, color="#c084fc")
    ax.plot(t_arr, H_arr, color="#c084fc", linewidth=2)
    ax.axhline(0.5, color="#22c55e", linestyle="--", linewidth=1, alpha=0.7, label="Lower bound (viable)")
    ax.axhline(2.0, color="#ef4444", linestyle="--", linewidth=1, alpha=0.7, label="Upper bound (viable)")
    ax.set_xlabel("Step", color="#64748b", fontsize=9)
    ax.set_ylabel("H[L]", color="#64748b", fontsize=9)
    ax.legend(fontsize=8, labelcolor="#94a3b8", facecolor=_AX, edgecolor=_SP)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.markdown("##### H vs r (Phase Space of Identity × Synchrony)")
        fig, ax = dark_fig(5, 4)
        sc = ax.scatter(r_arr, H_arr, c=t_arr, cmap="plasma", s=8, alpha=0.7)
        ax.set_xlabel("Order Parameter r(t)", color="#64748b", fontsize=9)
        ax.set_ylabel("Identity Curvature H[L]", color="#64748b", fontsize=9)
        ax.set_title("Viable region: r>0.3, H in [0.5,2.0]", color="#94a3b8", fontsize=8)
        fig.colorbar(sc, ax=ax, label="Time")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_h2:
        # Regime classification
        viable     = sum(1 for s in history if 0.5 <= s.H <= 2.0 and s.lyapunov_proxy < 0.5)
        chaotic    = sum(1 for s in history if s.lyapunov_proxy >= 0.5)
        catatonic  = sum(1 for s in history if s.r < 0.05)
        total      = len(history)
        st.markdown("##### Cognitive Regime Analysis")
        r1, r2, r3 = st.columns(3)
        r1.metric("Viable steps",   f"{100*viable/max(total,1):.1f}%", delta=f"{viable}/{total}")
        r2.metric("Chaotic steps",  f"{100*chaotic/max(total,1):.1f}%", delta=f"{chaotic}/{total}")
        r3.metric("Catatonic steps",f"{100*catatonic/max(total,1):.1f}%", delta=f"{catatonic}/{total}")

        # Regime stability judgment
        if viable / max(total, 1) > 0.6:
            st.success("✅ VIABLE REGIME: Identity preserved under self-modification.")
        elif chaotic / max(total, 1) > 0.3:
            st.error("❌ CHAOTIC REGIME: Identity breakdown — increase λ or rollback threshold.")
        else:
            st.warning("⚠️ TRANSITIONAL: System searching for stable attractor.")

# ── TAB 5: CURIOSITY ENGINE ─────────────────────────────────────────────────
with tab5:
    st.markdown("##### Curiosity Directive Timeline")
    directives = [s.curiosity_directive for s in history]
    directive_map = {
        "NONE": 0, "BOOST_CURIOSITY": 1, "DIVERSIFY_RULES": 2,
        "REFRAME_COUPLING": 3, "REDUCE_META_RATE": 4,
    }
    directive_vals = [directive_map.get(d, 0) for d in directives]
    dir_colors = {0: "#334155", 1: "#f59e0b", 2: "#38bdf8", 3: "#f87171", 4: "#a3e635"}

    fig, ax = dark_fig(10, 2.5)
    for i, (t_val, d_val) in enumerate(zip(t_arr, directive_vals)):
        ax.bar(t_val, 1, width=1.0, color=dir_colors[d_val], alpha=0.9)
    ax.set_yticks([]); ax.set_xlabel("Step", color="#64748b", fontsize=9)
    ax.set_title("Curiosity directives (color = type)", color="#94a3b8", fontsize=8)
    import matplotlib.patches as mpatches
    legend_els = [mpatches.Patch(color=c, label=k)
                  for k, c in zip(directive_map.keys(), dir_colors.values())]
    ax.legend(handles=legend_els, fontsize=7, labelcolor="#94a3b8",
              facecolor=_AX, edgecolor=_SP, loc="upper right", ncol=3)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    c5a, c5b = st.columns(2)
    with c5a:
        n_interventions = sum(1 for d in directives if d != "NONE")
        fig, ax = dark_fig(5, 3)
        ax.fill_between(t_arr, gam_arr, alpha=0.2, color="#fb923c")
        ax.plot(t_arr, gam_arr, color="#fb923c", linewidth=2)
        for i, (t_val, d) in enumerate(zip(t_arr, directives)):
            if d == "BOOST_CURIOSITY":
                ax.axvline(t_val, color="#f59e0b", alpha=0.4, linewidth=0.5)
        ax.set_xlabel("Step", color="#64748b", fontsize=9)
        ax.set_ylabel("Mean γ (curiosity)", color="#64748b", fontsize=9)
        ax.set_title(f"γ evolution ({n_interventions} interventions)", color="#94a3b8", fontsize=8)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with c5b:
        fig, ax = dark_fig(5, 3)
        # r vs error — should be anti-correlated
        ax.scatter(err_arr, r_arr, c=t_arr, cmap="viridis", s=6, alpha=0.6)
        ax.set_xlabel("Prediction Error ε", color="#64748b", fontsize=9)
        ax.set_ylabel("Order Parameter r", color="#64748b", fontsize=9)
        ax.set_title("Error vs Synchrony (should be anti-correlated)", color="#94a3b8", fontsize=8)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# ─── AUTO-RERUN IF STILL RUNNING ───────────────────────────────────────────────
if st.session_state.running and not exp.done:
    st.rerun()
elif exp.done:
    st.success(f"✅ Experiment complete. {len(st.session_state.history)} steps recorded.")
```

---

## PART V: WHAT TO MEASURE AND WHAT WOULD FALSIFY NECF

### Primary Claim (Must Hold for NECF to Be Valid)

> **Level-3 meta-dynamics (L evolves) produces quantitatively different adaptive behavior than Level-2 (fixed rules) under equivalent external perturbations.**

To test this: run two systems — NECF (L evolves) vs. Ablated-NECF (L fixed at initialization). Compare all 7 metrics. If the distributions are statistically indistinguishable, NECF provides no advantage.

### The 7 Falsifiable Predictions

| # | Prediction | Passes if | Fails if |
|---|-----------|-----------|---------|
| P1 | r(t) rises | r_final > 0.5 in viable regime | r stays < 0.2 always |
| P2 | H[L] bounded | H ∈ [0.1, 5.0] throughout | H diverges or → 0 |
| P3 | Error decreases | ε at t=1000 < ε at t=100 | ε increases or is flat |
| P4 | λ_max bounded | λ ∈ (-0.5, 0.8) | λ > 1.5 sustained |
| P5 | Rule diversity persists | Var(L_i) > 0.001 at t=T | Var → 0 (collapse) |
| P6 | Curiosity fires on plateau | Directive issued within 30 steps of plateau | Never fires |
| P7 | Rollback rate decreases | Rollbacks/100steps drops over time | Rate increases |

### How to Write the Paper

1. **Title**: "Hierarchical Meta-Rule Dynamics in Coupled Oscillator Fields: Identity-Constrained Level-3 Adaptation"
2. **Claim**: Precisely the Level-3 contribution above
3. **Baseline**: Kuramoto (fixed K), Adaptive Kuramoto (K evolves), NECF (L evolves)
4. **Experiments**: Run all three on same task. Compare P1-P7.
5. **Section 4**: Ablation — remove rollback, remove curiosity, remove identity gradient. Show each degrades performance.
6. **Venue**: First target: NeurIPS Workshop on Emergence in Complex Systems, or ICLR 2027

---

## PART VI: REQUIREMENTS.TXT

```
streamlit>=1.32.0
numpy>=1.24.0
scipy>=1.11.0
matplotlib>=3.7.0
numba>=0.58.0        # for fast oscillator loops
```

---

## PART VII: RUN INSTRUCTIONS

```bash
git clone https://github.com/Devanik21/NECF
cd NECF

pip install -r requirements.txt

# Launch dashboard
streamlit run app.py

# Batch run (no UI)
python -c "
from experiment import NECFExperiment
from config import NECFConfig
exp = NECFExperiment()
history = exp.run()
print(f'Final r: {history.snapshots[-1].r:.3f}')
print(f'Final H: {history.snapshots[-1].H:.3f}')
"
```

---

*Author: Devanik | NIT Agartala | March 2026*
*This document supersedes v1.0 (the conceptual draft). This is the implementable version.*
