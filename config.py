"""
config.py — All hyperparameters for NECF v2.
=============================================
Single source of truth. All modules import from here.
Change here; everything else adapts automatically.

v2 additions:
  - MetaRuleConfig.kappa_boltzmann: Boltzmann temperature for contagion softmax
    (Teacher Fix #1 — replaces the singular inverse-error weighting)
"""

from dataclasses import dataclass
from dataclasses import field as dc_field
import numpy as np


@dataclass
class FieldConfig:
    N:           int   = 64        # number of oscillator nodes
    dt:          float = 0.01      # integration time step
    T:           int   = 2000      # total steps per experiment
    noise_sigma: float = 0.02      # stochastic amplitude noise (Wiener)


@dataclass
class OscillatorConfig:
    omega_mean:  float = 1.0       # mean natural frequency ω_i
    omega_std:   float = 0.3       # std dev of frequency diversity
    A_init:      float = 0.5       # initial amplitude (uniform)
    theta_init:  str   = "random"  # "random" | "uniform" | "clustered"


@dataclass
class MetaRuleConfig:
    # Initial rule values
    alpha_init:  float = 0.3       # error sensitivity
    beta_init:   float = 0.8       # coupling strength
    gamma_init:  float = 0.1       # curiosity weight

    # Meta-update rates per rule dimension
    mu_alpha:    float = 0.05
    mu_beta:     float = 0.05
    mu_gamma:    float = 0.05

    # Parameter bounds (hard clipping)
    alpha_bounds: tuple = (0.01, 2.0)
    beta_bounds:  tuple = (0.01, 3.0)
    gamma_bounds: tuple = (0.001, 0.5)

    # ─── FIX #1: Boltzmann temperature ───────────────────────────────────
    # Controls sharpness of epistemic contagion weighting.
    # kappa → small (e.g. 0.01): sharp selection, closer to one-hot
    # kappa → large (e.g. 1.0):  diffuse, all nodes equally influential
    # Default 0.1: meaningful discrimination without singularity
    # Recommended range: [0.05, 0.5]
    kappa_boltzmann: float = 0.1


@dataclass
class IdentityConfig:
    kappa:               float = 0.5    # variance penalty weight κ in H[L]
    lambda_identity:     float = 0.1    # identity gradient weight λ in dL/dt
    rollback_threshold:  float = 0.3    # δH per step that triggers rollback
    rollback_eta:        float = 0.05   # rollback gradient step size


@dataclass
class CuriosityConfig:
    plateau_window:  int   = 20         # steps to check for plateau
    plateau_epsilon: float = 0.01       # min improvement to escape plateau
    spike_strength:  float = 0.3        # gamma boost factor on BOOST_CURIOSITY


@dataclass
class EnvironmentConfig:
    # Lorenz attractor coupling
    lorenz_eps:   float = 0.05          # phase kick amplitude
    lorenz_sigma: float = 10.0          # σ parameter
    lorenz_rho:   float = 28.0          # ρ parameter
    lorenz_beta:  float = 8.0 / 3.0    # β parameter

    # Periodic signal
    periodic_eps:  float = 0.03         # amplitude modulation strength
    periodic_freq: float = 0.1          # frequency (cycles per step × dt)

    # Poisson spikes
    spike_rate:    float = 0.02         # P(spike) per node per step
    # Note: spike_mask from spikes is returned by field.step() and
    # passed to observer.record() for masked Lyapunov computation


@dataclass
class WillConfig:
    mu_will:                    float = 1.0    # identity distortion tradeoff μ
    attractor_detection_threshold: float = 0.1  # bifurcation detection threshold


@dataclass
class NECFConfig:
    field:       FieldConfig       = dc_field(default_factory=FieldConfig)
    oscillator:  OscillatorConfig  = dc_field(default_factory=OscillatorConfig)
    meta_rule:   MetaRuleConfig    = dc_field(default_factory=MetaRuleConfig)
    identity:    IdentityConfig    = dc_field(default_factory=IdentityConfig)
    curiosity:   CuriosityConfig   = dc_field(default_factory=CuriosityConfig)
    environment: EnvironmentConfig = dc_field(default_factory=EnvironmentConfig)
    will:        WillConfig        = dc_field(default_factory=WillConfig)
    seed:        int               = 42


# Convenience singleton — import this for quick experiments
DEFAULT_CONFIG = NECFConfig()
