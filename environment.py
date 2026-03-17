"""
environment.py — External driving signals.
Lorenz attractor + periodic signal + Poisson spikes.
"""
import numpy as np


class LorenzDriver:
    def __init__(self, cfg):
        self.sigma = cfg.lorenz_sigma
        self.rho   = cfg.lorenz_rho
        self.beta  = cfg.lorenz_beta
        self.dt    = 0.01
        self.state = np.array([1.0, 0.0, 0.0])

    def step(self) -> np.ndarray:
        x, y, z = self.state
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        self.state += self.dt * np.array([dx, dy, dz])
        return self.state.copy()


class PeriodicDriver:
    def __init__(self, cfg):
        self.freq = cfg.periodic_freq
        self.amp  = cfg.periodic_eps
        self.t    = 0.0

    def step(self, dt: float) -> float:
        self.t += dt
        return self.amp * np.sin(2 * np.pi * self.freq * self.t)


class SpikeDriver:
    def __init__(self, cfg, rng):
        self.rate = cfg.spike_rate
        self.rng  = rng

    def get_spikes(self, N: int) -> np.ndarray:
        return (self.rng.uniform(size=N) < self.rate).astype(float)


class Environment:
    def __init__(self, cfg, rng):
        self.cfg      = cfg
        self.lorenz   = LorenzDriver(cfg.environment)
        self.periodic = PeriodicDriver(cfg.environment)
        self.spikes   = SpikeDriver(cfg.environment, rng)
        self.rng      = rng

    def step(self, dt: float, N: int) -> dict:
        lorenz_state = self.lorenz.step()
        lorenz_x     = lorenz_state[0] / 25.0   # normalize to ~[-1, 1]

        periodic_val = self.periodic.step(dt)

        # spike_mask is float (N,); field.step() converts to bool
        spike_mask   = self.spikes.get_spikes(N)

        node_indices      = np.arange(N)
        lorenz_phase_kick = (
            self.cfg.environment.lorenz_eps
            * lorenz_x
            * np.sin(node_indices * 2 * np.pi / N)
        )

        periodic_amp_mod = periodic_val * np.ones(N)

        return {
            "lorenz_phase_kick": lorenz_phase_kick,
            "periodic_amp_mod":  periodic_amp_mod,
            "spike_mask":        spike_mask,
            "lorenz_state":      lorenz_state,
        }
