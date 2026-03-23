<div align="center">

# Non-Equilibrium Cognitive Field (NECF)

### Identity-Constrained Level-3 Meta-Rule Dynamics in Coupled Oscillator Fields

**Devanik**  
B.Tech ECE '26, NIT Agartala | Samsung Convergence Software Fellow, IISc

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/NumPy%2FNumba-Optimized-013243?style=for-the-badge&logo=numpy&logoColor=white)]()
[![Status](https://img.shields.io/badge/Research-Active-7c3aed?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-22c55e?style=for-the-badge)]()
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)]()

</div>

---

## Abstract

We introduce the **Non-Equilibrium Cognitive Field (NECF)**, a dynamical systems architecture in which local learning rules — not merely state variables — are themselves continuous dynamic quantities evolving under thermodynamic constraints. Prior adaptive oscillator models (Kuramoto, 1975; Ha et al., 2016; arXiv:2505.03648) universally stop at **Level-2**: the coupling strengths $K_{ij}(t)$ evolve, but the *rule governing their evolution* is a fixed hyperparameter. NECF instantiates **Level-3**: each node $i$ carries a local meta-rule vector $\mathcal{L}_i(t) = (\alpha_i, \beta_i, \gamma_i) \in \mathbb{R}^3$ that itself constitutes a dynamical state variable, evolving via Boltzmann-weighted epistemic contagion — a thermodynamically smooth diffusion of successful strategies from low-error to high-error nodes. This evolution is constrained by an **Identity Curvature Functional** $H[\mathcal{L}]$, a scalar energy functional over rule space that prevents both chaotic drift and catatonic homogenization, and is enforced via a Lyapunov-gated rollback mechanism that provides reversibility guarantees otherwise absent from continuous adaptive systems. The system is driven perpetually far from equilibrium by three simultaneous perturbation sources (Lorenz chaos, structured periodic forcing, Poisson phase resets), implementing an open thermodynamic architecture in the sense of Prigogine (1977). We provide a complete, Numba-accelerated Python implementation, a live Streamlit dashboard, and seven falsifiable empirical predictions against which the architecture is tested.

---

## Table of Contents

1. [Theoretical Positioning](#1-theoretical-positioning)
2. [Mathematical Substrate](#2-mathematical-substrate)
3. [Level-3 Meta-Rule Dynamics](#3-level-3-meta-rule-dynamics)
4. [Identity Curvature Functional](#4-identity-curvature-functional)
5. [Non-Equilibrium Driving](#5-non-equilibrium-driving)
6. [Curiosity Engine & Proto-Will](#6-curiosity-engine--proto-will)
7. [Implementation Architecture](#7-implementation-architecture)
8. [The Seven Falsifiable Predictions](#8-the-seven-falsifiable-predictions)
9. [Installation & Usage](#9-installation--usage)
10. [Ablation Hierarchy](#10-ablation-hierarchy)
11. [Mathematical Glossary](#11-mathematical-glossary)
12. [Literature Positioning](#12-literature-positioning)

---

## 1. Theoretical Positioning

### 1.1 The Adaptive Hierarchy

All adaptive dynamical systems can be classified by the highest *level* at which dynamics operate:

| Level | What evolves | What is fixed | Representative systems |
|:---:|---|---|---|
| **0** | Nothing | Everything | Lookup tables, static logic |
| **1** | State $\phi(t)$ | Learning rule $\mathcal{L}$ | Standard Kuramoto (1975), frozen-weight ANNs |
| **2** | State $\phi(t)$, Coupling $K_{ij}(t)$ | Rule governing $K$'s evolution | Adaptive Kuramoto (Ha et al., 2016), SGD on weights |
| **3** | State $\phi(t)$, Rule $\mathcal{L}_i(t)$ | Identity curvature $H[\mathcal{L}]$ | **NECF (this work)** |

The distinction between Level-2 and Level-3 is not cosmetic. In Level-2 adaptive Kuramoto (Ha et al., 2016), the plasticity rule — the equation defining how fast $K_{ij}$ changes — is a static architectural hyperparameter. A system can learn *which coupling strengths to use*, but not *how to learn*. NECF's Level-3 architecture allows nodes to learn *how* to update their coupling, error sensitivity, and exploratory drive — in continuous time, locally, without backpropagation.

### 1.2 Precise Novelty Claim

The following combination does not appear in the literature after exhaustive search (Google Scholar, arXiv, SIAM, Nature, ScienceDirect, March 2026):

> **A coupled oscillator field where local learning rules** $(\alpha_i, \beta_i, \gamma_i)$ **evolve as a second-order dynamical system under Boltzmann-weighted epistemic contagion, constrained by an identity curvature functional** $H[\mathcal{L}]$ **with Lyapunov-gated thermodynamic rollback — implemented as a unified, continuous-time, backpropagation-free architecture.**

Individual components exist. Their specific integration does not:

- Standard Kuramoto: $K$ **fixed**
- Adaptive Kuramoto (Ha et al.): $K_{ij}$ **evolves**, plasticity rule **fixed**
- Hopfield-Kuramoto (arXiv:2505.03648, 2025): joint memory, **no meta-rule evolution**
- MAML (Finn et al., 2017): learns initialization, **freezes during deployment**
- FEP (Friston, 2010–2022): minimizes free energy, **optimization machinery fixed**
- **NECF**: rules governing adaptation are **dynamic state variables** constrained by identity topology

---

## 2. Mathematical Substrate

### 2.1 State Representation

Each node $i \in \{1, \dots, N\}$ carries a complex state variable $\phi_i(t) \in \mathbb{C}$:

$$
\phi_i(t) = A_i(t) \, e^{i\theta_i(t)}
$$

where $A_i \in (0, 1]$ is local amplitude (confidence/energy) and $\theta_i \in [0, 2\pi)$ is phase (causal temporal alignment). The macroscopic synchronization of the field is captured by the **Kuramoto Order Parameter**:

$$
r(t) \, e^{i\psi(t)} = \frac{1}{N} \sum_{j=1}^{N} A_j(t) \, e^{i\theta_j(t)}
$$

where $r \in [0, 1]$ is the synchronization index and $\psi(t)$ is the instantaneous mean-field phase. The limits $r \to 0$ and $r \to 1$ correspond to incoherent and fully synchronized regimes respectively. The NECF is designed to maintain $r$ in the **edge-of-chaos** regime $r \in [0.5, 0.8]$ — coherent enough for information propagation, disordered enough for adaptive flexibility.

### 2.2 Phase Evolution: Generalized Kuramoto with Curiosity

The phase of each node evolves under four simultaneous forces:

$$
\frac{d\theta_i}{dt} = \underbrace{\omega_i}_{\text{intrinsic}} + \underbrace{\beta_i \frac{1}{N} \sum_{j=1}^{N} W_{ij} A_j \sin(\theta_j - \theta_i)}_{\text{synchrony pull (local } \beta_i\text{)}} + \underbrace{\gamma_i \nabla_{\theta_i} U(\theta_i)}_{\text{curiosity gradient}} + \underbrace{\Delta_{\text{ext}}(t)}_{\text{chaos/noise}}
$$

The critical departure from standard Kuramoto is that $\beta_i$ is **not a global constant $K$**. It is an element of the node's dynamically evolving meta-rule vector $\mathcal{L}_i(t)$. Nodes that consistently mispredict their neighbors' phases evolve toward lower $\beta_i$; successful nodes evolve toward higher $\beta_i$ — a mechanism with no analogue in the Kuramoto literature.

The coupling matrix $W_{ij}$ is symmetric and zero-diagonal:

$$
W_{ij} \sim \mathcal{U}(0.5, 1.5), \quad W_{ii} = 0, \quad W_{ij} = W_{ji}
$$

Natural frequencies $\omega_i$ are drawn once and frozen:

$$
\omega_i \sim \mathcal{N}(\omega_{\text{mean}}, \sigma_\omega^2), \quad \omega_{\text{mean}} = 1.0, \quad \sigma_\omega = 0.3
$$

### 2.3 Amplitude Evolution: Confidence Under Error

Node amplitude (local confidence) decays under local prediction error $\varepsilon_i(t)$:

$$
\frac{dA_i}{dt} = -\alpha_i \, \varepsilon_i(t) \, A_i + \sigma \eta_i(t) + \Delta A_{\text{periodic}}(t)
$$

where $\eta_i(t) \sim \mathcal{N}(0, \sigma^2)$ is a Wiener process with $\sigma = 0.02$, preventing amplitude collapse to exactly zero (thermodynamic floor). The **local prediction error** is the squared circular distance from the mean-field phase:

$$
\varepsilon_i(t) = \sin^2\!\left(\frac{\theta_i(t) - \psi(t)}{2}\right) \in [0, 1]
$$

This has the property that $\varepsilon_i \approx \frac{1}{4}(\theta_i - \psi)^2$ for small deviations and saturates to 1 for nodes that are phase-antithetical to the field consensus. The sensitivity to error is governed dynamically by $\alpha_i(t)$ — itself a component of $\mathcal{L}_i(t)$.

### 2.4 The Curiosity Potential $U(\theta_i)$

Drawing from Friston's Active Inference framework, nodes maintain a curiosity gradient that drives exploration of sparse phase regions. The Uncertainty Potential $U(\theta_i)$ is the negative log-likelihood of the local phase density, approximated via circular Kernel Density Estimation with Silverman bandwidth $h = 1.06 \hat{\sigma}_\theta N^{-1/5}$:

$$
p(\theta_i) \approx \frac{1}{N} \sum_{j=1}^{N} \frac{1}{h} \exp\!\left(-\frac{\sin^2(\theta_i - \theta_j)}{2h^2}\right)
$$

$$
U(\theta_i) = -\log p(\theta_i \mid \text{context})
$$

Nodes in dense, highly coherent phase regions experience a **low curiosity gradient** — they are already well-represented and need not explore. Nodes in sparse, chaotic regions experience a **high curiosity gradient**, pulling them toward informationally rich territory. The strength of this pull is governed by $\gamma_i(t)$, the curiosity weight in $\mathcal{L}_i$.

---

## 3. Level-3 Meta-Rule Dynamics

### 3.1 The Meta-Rule Field

This is NECF's core architectural contribution. Each node $i$ carries a **local learning rule vector**:

$$
\mathcal{L}_i(t) = \bigl(\alpha_i(t),\; \beta_i(t),\; \gamma_i(t)\bigr) \in \mathbb{R}^3
$$

with hard bounds enforced by projection:

$$
\alpha_i \in [0.01, 2.0], \quad \beta_i \in [0.01, 3.0], \quad \gamma_i \in [0.001, 0.5]
$$

This vector is **not static**. It evolves as a dynamical system in rule space:

$$
\frac{d\mathcal{L}_i}{dt} = F_{\text{contagion}}(\mathcal{L}_i, \varepsilon_i, W) - \lambda \nabla_{\mathcal{L}_i} H[\mathcal{L}]
$$

where $F_{\text{contagion}}$ implements epistemic contagion (Section 3.2) and $\lambda \nabla H$ is the identity restoring gradient (Section 4).

### 3.2 The Epistemic Contagion Singularity Problem

A naive approach to rule sharing would weight neighbors inversely by prediction error:

$$
w_j^{\text{naive}} = \frac{1}{\varepsilon_j + \epsilon}
$$

This is **physically broken**. Let $\varepsilon_1 = 10^{-6}$, $\varepsilon_2 = 10^{-1}$. Then $w_1 / w_2 = 10^5$. After normalization, the weight vector degenerates to $\delta_{j,1}$ — the entire field violently collapses to the rule of a single "lucky" node. This destroys rule diversity and makes adaptation a discrete switch rather than a field phenomenon. The differential equation becomes ill-conditioned.

### 3.3 Boltzmann Softmax Epistemic Contagion

NECF resolves the singularity by mapping prediction error to thermodynamic energy and applying the canonical ensemble distribution:

$$
w_j = \frac{\exp(-\varepsilon_j / \kappa)}{\sum_k \exp(-\varepsilon_k / \kappa)}
$$

where $\kappa = 0.1$ is the Boltzmann temperature, implemented with log-sum-exp stabilization:

$$
w_j = \frac{\exp\!\left(-\varepsilon_j/\kappa - \max_k(-\varepsilon_k/\kappa)\right)}{\sum_k \exp\!\left(-\varepsilon_k/\kappa - \max_k(-\varepsilon_k/\kappa)\right)}
$$

This guarantees $w_j \in (0, 1)$, $\sum_j w_j = 1$, and $C^\infty$ differentiability everywhere. The sensitivity of weights to error is:

$$
\frac{\partial w_j}{\partial \varepsilon_j} = -\frac{1}{\kappa} w_j (1 - w_j)
$$

which is bounded and well-defined for all $\varepsilon_j$. The temperature $\kappa$ continuously interpolates between two limits: $\kappa \to 0$ recovers crisp winner-take-all, while $\kappa \to \infty$ gives uniform trust in all neighbors.

### 3.4 The Contagion Update

Each node $i$ computes a thermodynamically-weighted target rule from its spatial neighborhood:

$$
\mathcal{L}_i^{\text{target}} = \frac{\sum_j W_{ij} w_j \mathcal{L}_j}{\sum_j W_{ij} w_j + \epsilon}
$$

The epistemic contagion force then pulls $\mathcal{L}_i$ toward $\mathcal{L}_i^{\text{target}}$ at a rate scaled by the node's **own** prediction error $\varepsilon_i$:

$$
F_{\text{contagion},\, i} = \boldsymbol{\mu} \odot (\mathcal{L}_i^{\text{target}} - \mathcal{L}_i) \cdot \varepsilon_i
$$

where $\boldsymbol{\mu} = (\mu_\alpha, \mu_\beta, \mu_\gamma) = (0.05, 0.05, 0.05)$ are per-dimension update rates and $\odot$ denotes elementwise multiplication. The factor $\varepsilon_i$ makes this **thermodynamically self-regulating**: high-error nodes are highly *receptive* and update their rules rapidly toward successful neighbors. Low-error nodes are *stubborn* — they barely modify their rules, acting as stable anchors from which the contagion radiates.

This implements a form of **epistemic natural selection**: rules that produce low prediction error are thermodynamically preferred, diffuse through the network, and suppress strategies that produce high error — all without any external supervisor, gradient oracle, or loss landscape.

---

## 4. Identity Curvature Functional

### 4.1 Motivation: The Two Failure Modes

Left unconstrained, epistemic contagion suffers two catastrophic asymptotic behaviors:

1. **Chaotic drift**: Under continuous perturbation from the Lorenz attractor and Poisson spikes, the random walk in rule space causes $\|\mathcal{L}_i(t)\|$ to grow without bound.
2. **Catatonic homogenization**: Because contagion is a weighted averaging operation, $\text{Var}(\mathcal{L}_i)$ monotonically decreases toward zero. All nodes converge to identical rules, destroying the spatial diversity required for distributed cognition. The system mathematically degenerates to Level-1.

### 4.2 The Functional

The **Identity Curvature Functional** $H[\mathcal{L}]$ is a scalar energy functional defined over the rule field:

$$
H[\mathcal{L}] = \underbrace{\frac{1}{N} \sum_{i=1}^{N} \bigl\|\mathcal{L}_i(t) - \mathcal{L}_i(0)\bigr\|^2}_{\text{Drift Penalty}} + \underbrace{\kappa_{\text{id}} \cdot \overline{\text{Var}}(\mathcal{L}_i)}_{\text{Homogenization Penalty}}
$$

where $\overline{\text{Var}}(\mathcal{L}_i) = \frac{1}{3}\sum_{d=1}^{3} \text{Var}_i(\mathcal{L}_{i,d})$ is the mean variance across the three rule dimensions and $\kappa_{\text{id}} = 0.5$.

The two terms have complementary roles:
- The **drift penalty** is an $L_2$ structural memory. It creates a parabolic energy well centered at $\mathcal{L}_i(0)$, providing an elastic restoring force toward the node's topological origin.
- The **homogenization penalty** penalizes the collapse of rule diversity. As nodes converge toward the mean rule $\bar{\mathcal{L}}$, the variance term *decreases*, but the system takes the *gradient* of $H$, and the variance gradient acts as a **repulsive field** pushing nodes away from $\bar{\mathcal{L}}$.

### 4.3 Identity Gradient

The continuous gradient that appears in the rule evolution equation is derived analytically:

$$
\nabla_{\mathcal{L}_i} H[\mathcal{L}] = \frac{2}{N}\bigl(\mathcal{L}_i - \mathcal{L}_i(0)\bigr) + \kappa_{\text{id}} \frac{2}{N}\bigl(\mathcal{L}_i - \bar{\mathcal{L}}\bigr)
$$

The interplay between contagion (pulling toward successful neighbors) and identity gradient (pulling toward initial topology and away from the mean) generates a **strange attractor** in meta-rule space — a bounded, high-dimensional orbit that preserves both collective adaptivity and individual structural identity.

### 4.4 Thermodynamic Rollback

Discrete numerical integration ($dt = 0.01$) combined with sudden chaotic spikes from the Lorenz attractor can cause instantaneous ruptures in the rule field that the continuous gradient cannot intercept in time. NECF implements a hard fail-safe via **Lyapunov-gated rollback**. At each timestep, the change in identity curvature is computed:

$$
\delta H(t) = H[\mathcal{L}(t)] - H[\mathcal{L}(t - dt)]
$$

If $\delta H > \delta_{\text{thresh}} = 0.3$, the system is undergoing **acute identity trauma**. The rule field is instantly reverted one timestep, and a corrective gradient step is applied:

$$
\mathcal{L}(t) \leftarrow \mathcal{L}(t - dt) - \eta_{\text{rb}} \, \nabla_{\mathcal{L}} H\bigl[\mathcal{L}(t - dt)\bigr], \quad \eta_{\text{rb}} = 0.05
$$

This is not gradient descent in the conventional sense. It is a **thermodynamically aware, time-reversible memory function**: the system reverts to a known-stable state *and* takes a corrective step that reduces the likelihood of the same instability recurring. For $T \to \infty$, this mechanism guarantees absolute topological stability as a Lyapunov stability certificate.

The rollback mechanism also has a falsifiable prediction: its firing rate should decrease monotonically over time as the meta-rules converge to a stable attractor basin, providing immunity to future shocks.

---

## 5. Non-Equilibrium Driving

### 5.1 Why Closed Systems Fail Cognitively

A closed dynamical system will eventually minimize free energy completely, reaching thermal equilibrium — a state of maximal entropy where no computation can occur. In the language of Prigogine (1977), the system reaches thermodynamic **death**. Dissipative structures (and cognitive systems) require a **continuous input of free energy** — an external gradient forcing the system perpetually far from equilibrium.

NECF implements three orthogonal perturbation channels:

### 5.2 Lorenz Chaotic Attractor

A classical Lorenz system with parameters $\sigma = 10$, $\rho = 28$, $\beta_L = 8/3$ is integrated in parallel:

$$
\frac{dx}{dt} = \sigma(y - x), \quad \frac{dy}{dt} = x(\rho - z) - y, \quad \frac{dz}{dt} = xy - \beta_L z
$$

Its $x(t)$ coordinate, normalized to $[-1, 1]$, is injected as a **spatially distributed phase kick**:

$$
\Delta\theta_i^{\text{Lorenz}} = \varepsilon_L \cdot x(t) \cdot \sin\!\left(\frac{2\pi i}{N}\right), \quad \varepsilon_L = 0.05
$$

The spatial factor $\sin(2\pi i / N)$ is critical: nodes near $i = N/4$ receive a strong positive kick while nodes near $i = 3N/4$ receive an equally strong negative kick. The field is physically torn in half in a structured, aperiodic, deterministic manner — a far richer test than white noise.

### 5.3 Structured Periodic Forcing

A global sinusoidal signal modulates node amplitudes, providing a predictable resonant carrier:

$$
A_i(t) \leftarrow A_i(t) + \varepsilon_s \sin(2\pi f_s t), \quad \varepsilon_s = 0.03, \quad f_s = 0.1
$$

### 5.4 Poisson Phase Resets

At each timestep, every node has a Poisson spike probability $\lambda_s = 0.02$:

$$
\theta_i(t) \leftarrow \mathcal{U}(0, 2\pi) \quad \text{if } \xi_i \sim \mathcal{U}(0,1) < \lambda_s \cdot dt
$$

These simulate sudden synaptic failures or catastrophic data loss. The **spike mask** is threaded to the observer module for Lyapunov computation (see Section 7.3).

---

## 6. Curiosity Engine & Proto-Will

### 6.1 Plateau Detection and Adaptive Directives

When the mean prediction error $\bar{\varepsilon}$ stagnates over a window of 20 timesteps (variation $< \varepsilon_{\text{plateau}} = 0.01$), the **CuriosityEngine** issues an adaptive directive based on system state:

| Condition | Directive | Action |
|---|---|---|
| Rollbacks $> 5$ | `REDUCE_META_RATE` | Scale $\mathcal{L}$ by $0.95$ |
| Falsified steps $> 3$ | `REFRAME_COUPLING` | Randomly reinitialize 20% of $\beta_i$ |
| Alternating | `BOOST_CURIOSITY` | Amplify $\gamma_i$ by $1 + 0.3$ |
| Alternating | `DIVERSIFY_RULES` | Add $\mathcal{N}(0, 0.05)$ noise to $\mathcal{L}$ |

This implements a meta-meta-level feedback: the curiosity engine monitors the meta-rule dynamics and perturbs them when they stagnate.

### 6.2 Proto-Will: Constrained Empowerment

At bifurcation points, when the system can settle into one of $k$ attractor basins, a **proto-will** objective determines basin selection. Following Klyubin et al. (2005), **empowerment** is the mutual information channel capacity between actions and future states:

$$
\mathcal{E} = \max_{p(a)} I(A;\, S')
$$

In NECF, each candidate attractor $a$ has an associated **Causal Entropy**:

$$
H_{\text{causal}}(a) = -\sum_{S'} p(S' \mid a) \log p(S' \mid a)
$$

and an **Identity Distortion Cost**:

$$
D_{\text{identity}}(a) = \frac{1}{N} \sum_{i=1}^{N} \bigl\|\mathcal{L}_{i,\,\text{attractor}\,a} - \mathcal{L}_{i,\,\text{current}}\bigr\|^2
$$

Basin selection follows:

$$
a^* = \arg\max_a \Bigl[H_{\text{causal}}(a) - \mu_{\text{will}} \cdot D_{\text{identity}}(a)\Bigr], \quad \mu_{\text{will}} = 1.0
$$

This unifies Pearl's Causality (2009) with Friston's Free Energy Principle (2010): the system seeks maximum future causal influence while preserving its current structural identity. It is the computational analogue of *acting with intention rather than impulse*.

---

## 7. Implementation Architecture

### 7.1 Module Overview

```
NECF/
├── config.py           — Single-source hyperparameter truth (NECFConfig dataclass)
├── field.py            — Level-1: Numba-compiled oscillator field
├── meta_dynamics.py    — Level-3: Boltzmann epistemic contagion + rollback
├── identity.py         — Identity Curvature Functional H[L] + gradient
├── curiosity.py        — KDE uncertainty potential + plateau directives
├── environment.py      — Lorenz / periodic / spike drivers
├── observer.py         — Masked Lyapunov proxy + snapshot recording
├── experiment.py       — Orchestration (Level-1 / Level-2 / Level-3 ablation)
└── Notebook/           — Jupyter notebooks (12 experiments, PyTorch batched)
```

### 7.2 Numba-Accelerated Kuramoto Kernel

The $O(N^2)$ Kuramoto synchrony sum is compiled to native machine code via `@numba.njit`:

```python
@njit(parallel=True, cache=True, fastmath=True)
def _kuramoto_sync_pull_numba(theta, A, W, beta):
    N = theta.shape[0]
    sync_pull = np.zeros(N)
    for i in prange(N):       # parallel outer loop
        s = 0.0
        for j in range(N):    # no (N,N) matrix allocated
            s += W[i, j] * A[j] * np.sin(theta[j] - theta[i])
        sync_pull[i] = beta[i] * s / N
    return sync_pull
```

The NumPy broadcasting alternative allocates a fresh $(N \times N)$ matrix per step: at $N = 64$, $T = 2000$, this generates 64 MB of garbage-collected memory churn. The Numba kernel eliminates this entirely, achieving approximately $10$–$50\times$ speedup for $N \geq 64$ and enabling interactive dashboard use. A JIT pre-warm in `__init__` prevents first-step latency.

### 7.3 Masked Lyapunov Proxy

The standard Lyapunov proxy $\lambda \approx \frac{1}{\Delta t} \log \|\delta\theta(t)\|$ is **incorrect** in the presence of external Poisson spikes. A spiked node's phase can jump from $0.1$ to $5.9$ radians in one step — a phase distance of $5.8$. The raw $L_2$ norm of $\delta\theta$ explodes, triggering the rollback mechanism continuously and freezing all dynamics.

The **masked Lyapunov proxy** computes phase divergence only on nodes that were NOT spiked in the current or previous step:

$$
\lambda_{\text{proxy}} = \frac{1}{\Delta t} \log\!\left(\frac{\|\delta\theta[\sim\text{spike\_mask}]\|}{\sqrt{N_{\text{valid}}}} + 10^{-10}\right)
$$

The spike mask is explicitly threaded from `environment.step()` through `field.step()` to `observer.record()` at each timestep. If $N_{\text{valid}} < 2$ (almost all nodes spiked simultaneously), the proxy returns `NaN` and the step is classified as `SPIKE_DOMINATED` — a transient physical event, not internal chaos.

### 7.4 Complete Step Ordering

The experiment orchestrator executes each timestep in strict causal order:

```
t → t+1:
  1. Environment.step()      → ext signals (Lorenz kick, periodic mod, spike mask)
  2. CuriosityEngine.U(θ)    → uncertainty potential ∇U(θ_i) for curiosity pull
  3. OscillatorField.step()  → dθ/dt, dA/dt integration; returns (ε, spike_mask)
  4. CuriosityEngine.update()→ record ε̄, detect plateau
  5. CuriosityEngine.directive → modify L if plateauing
  6. MetaDynamics.step()     → dL/dt contagion + identity gradient + rollback check
  7. Observer.record()       → masked λ, snapshot, regime classification
```

This ordering is not arbitrary. Field dynamics at step $t$ must use rules $\mathcal{L}(t)$, not $\mathcal{L}(t+1)$. The curiosity directive modifies $\mathcal{L}$ *before* the meta-dynamics step so that the contagion and identity gradient operate on the post-directive state.

---

## 8. The Seven Falsifiable Predictions

Any scientific theory must make bold, falsifiable predictions prior to empirical testing. NECF makes seven:

| # | Prediction | Pass Condition | Fail Condition |
|:---:|---|---|---|
| **P1** | Macroscopic coherence | $r_{\text{final}} > 0.5$ (viable regime) | $r$ remains $< 0.2$ throughout |
| **P2** | Identity stability | $H[\mathcal{L}] \in [0.1, 5.0]$ for all $t$ | $H \to \infty$ (drift) or $H \to 0$ (collapse) |
| **P3** | Error suppression | $\bar\varepsilon(t = T) < \bar\varepsilon(t = 100)$ | $\bar\varepsilon$ increases or is flat |
| **P4** | Edge-of-chaos | $\lambda_{\text{proxy}} \in (-0.5, 0.8)$ | $\lambda > 1.5$ sustained |
| **P5** | Rule diversity | $\text{Var}(\mathcal{L}_i) > 10^{-3}$ at $t = T$ | $\text{Var} \to 0$ (homogenization) |
| **P6** | Curiosity response | Directive issued within 30 steps of plateau | Directive never fires |
| **P7** | Rollback rate decays | Rollbacks/100 steps $\downarrow$ over time | Rate increases or stays constant |

These predictions are jointly necessary. A system that passes P1 but fails P5 has synchronized, but at the cost of rule diversity — it has become a glorified Level-1 Kuramoto model. The conjunction of P1 through P7 constitutes evidence for **genuine Level-3 adaptive dynamics** rather than a trivially synchronized or trivially chaotic system.

### 8.1 Regime Classification

The observer classifies each timestep into one of five regimes based on the masked Lyapunov proxy, order parameter, and identity curvature:

```python
VIABLE         :  λ ∈ (-0.5, 0.8)  AND  r > 0.15  AND  H ∈ (0.05, 10.0)
CHAOTIC        :  λ > 0.8   (internal divergence, not spike noise)
CATATONIC      :  r < 0.05  (complete desynchronization)
IDENTITY_BROKEN:  H > 10.0 or H < 0.01
SPIKE_DOMINATED:  N_valid < 2 (transient; not a failure mode)
```

A healthy NECF run should remain `VIABLE` for $> 60\%$ of timesteps.

---

## 9. Installation & Usage

### 9.1 Requirements

```
streamlit>=1.32.0
numpy>=1.24.0
scipy>=1.11.0
matplotlib>=3.7.0
numba>=0.58.0
```

### 9.2 Installation

```bash
git clone https://github.com/Devanik21/Non-Equilibrium-Cognitive-Field
cd Non-Equilibrium-Cognitive-Field
pip install -r requirements.txt
```

### 9.3 Interactive Dashboard

```bash
streamlit run app.py
```

The dashboard provides five live tabs: Phase & Synchrony, Observable Metrics, Rule Dynamics, Identity & Will, and Curiosity Engine — all updating in real-time as the simulation runs.

### 9.4 Batch Simulation

```python
from experiment import NECFExperiment
from config import NECFConfig

cfg = NECFConfig()
exp = NECFExperiment(cfg)
history = exp.run_batch()

snapshots = history.snapshots
final = snapshots[-1]
print(f"Final r:            {final.r:.4f}")
print(f"Final H[L]:         {final.H:.4f}")
print(f"Final mean error:   {final.mean_error:.4f}")
print(f"Rule diversity:     {final.rule_diversity:.6f}")
print(f"Total rollbacks:    {final.rollback_count}")
print(f"Lyapunov proxy:     {final.lyapunov_proxy:.4f}")
```

### 9.5 Custom Configuration

```python
from config import NECFConfig, FieldConfig, MetaRuleConfig, IdentityConfig

cfg = NECFConfig(
    field=FieldConfig(N=128, T=5000, dt=0.01),
    meta_rule=MetaRuleConfig(
        kappa_boltzmann=0.15,   # sharper contagion
        beta_init=1.2,          # stronger initial coupling
    ),
    identity=IdentityConfig(
        kappa=0.8,              # stronger diversity penalty
        rollback_threshold=0.2, # tighter rollback gate
    ),
    seed=1337
)

exp = NECFExperiment(cfg)
history = exp.run_batch()
```

---

## 10. Ablation Hierarchy

### 10.1 Level-1 Ablation (Frozen Rules)

Rules $\mathcal{L}_i$ are fixed at initialization for the entire run. The field evolves as a standard Kuramoto model with local (but static) parameters:

```python
exp_l1 = NECFExperiment.level1_only(cfg)
history_l1 = exp_l1.run_batch()
```

This is the **primary null hypothesis baseline**. If NECF (Level-3) cannot statistically outperform Level-1 on P1–P7, the meta-rule dynamics provide no measurable benefit.

### 10.2 Level-2 Ablation (Global Adaptive Coupling)

Only $\beta$ adapts, globally (not per-node), via a mean-field error signal. This reproduces the Ha et al. (2016) adaptive Kuramoto setting:

```python
exp_l2 = NECFExperiment.level2_adaptive_coupling(cfg)
history_l2 = exp_l2.run_batch()
```

$$
\beta(t + dt) = \beta(t) + 0.01 \cdot (0.5 - \bar{\varepsilon}(t)) \cdot dt
$$

$\alpha$ and $\gamma$ remain frozen; no per-node differentiation; no epistemic contagion; no identity constraint.

### 10.3 Falsification via Configuration Ablation

| Ablation | Config Change | Predicted Failure |
|---|---|---|
| Disable identity | `kappa=0.0`, `lambda_identity=0.0` | Rule diversity → 0 (P5 fails) |
| Disable contagion | `kappa_boltzmann→∞` | All nodes equally weighted; no adaptation |
| Winner-takes-all | `kappa_boltzmann=0.001` | Rule field snaps to single node (P5 fails) |
| No rollback | `rollback_threshold→∞` | H diverges under Lorenz spikes (P2 fails) |
| Close thermodynamics | `lorenz_eps=0`, `spike_rate=0`, `periodic_eps=0` | Rules collapse to homogeneous fixed point |

These ablations are designed to be **falsifying**: they test specific architectural claims by removing individual components and observing the predicted degradation. A system that degrades correctly under ablation provides stronger evidence for causal, not merely correlational, contributions from each component.

---

## 11. Mathematical Glossary

### 11.1 Substrate Variables (Level-1)

| Symbol | Interpretation | Range |
|:---:|---|---|
| $N$ | Number of oscillator nodes | $\{16, 32, 64, 128\}$ |
| $\phi_i(t)$ | Complex state $A_i e^{i\theta_i}$ | $\mathbb{C}$ |
| $\theta_i(t)$ | Phase angle | $[0, 2\pi)$ |
| $A_i(t)$ | Amplitude (confidence) | $(0, 1]$ |
| $\omega_i$ | Intrinsic natural frequency (fixed) | $\mathcal{N}(1.0, 0.09)$ |
| $W_{ij}$ | Symmetric coupling adjacency matrix | $(0, 1.5]$, $W_{ii}=0$ |
| $r(t)$ | Kuramoto order parameter | $[0, 1]$ |
| $\psi(t)$ | Mean-field phase | $[0, 2\pi)$ |
| $\varepsilon_i(t)$ | Local prediction error | $[0, 1]$ |
| $U(\theta_i)$ | Curiosity potential $(-\log p)$ | $[0, \infty)$ |

### 11.2 Meta-Rule Variables (Level-3)

| Symbol | Interpretation | Range |
|:---:|---|---|
| $\mathcal{L}_i(t)$ | Local learning rule vector | $\mathbb{R}^3$ |
| $\alpha_i(t)$ | Error sensitivity (amplitude decay rate) | $[0.01, 2.0]$ |
| $\beta_i(t)$ | Coupling strength (synchrony pull) | $[0.01, 3.0]$ |
| $\gamma_i(t)$ | Curiosity weight (uncertainty gradient) | $[0.001, 0.5]$ |
| $\boldsymbol{\mu}$ | Meta-update rate vector $(\mu_\alpha, \mu_\beta, \mu_\gamma)$ | $0.05$ each |
| $w_j$ | Boltzmann epistemic weight of node $j$ | $(0, 1)$ |
| $\kappa$ | Boltzmann temperature | $0.1$ |

### 11.3 Identity Functional Variables

| Symbol | Interpretation | Value |
|:---:|---|---|
| $H[\mathcal{L}]$ | Identity Curvature Functional | $[0, \infty)$ |
| $\mathcal{L}_i(0)$ | Structural memory anchor | Fixed at $t=0$ |
| $\kappa_{\text{id}}$ | Homogenization penalty weight | $0.5$ |
| $\lambda$ | Identity gradient weight in $d\mathcal{L}/dt$ | $0.1$ |
| $\delta_{\text{thresh}}$ | Rollback trigger threshold | $0.3$ |
| $\eta_{\text{rb}}$ | Rollback gradient step size | $0.05$ |

### 11.4 Environmental Variables

| Symbol | Interpretation | Value |
|:---:|---|---|
| $\varepsilon_L$ | Lorenz phase kick amplitude | $0.05$ |
| $(\sigma, \rho, \beta_L)$ | Lorenz parameters | $(10, 28, 8/3)$ |
| $\varepsilon_s$ | Periodic amplitude modulation | $0.03$ |
| $f_s$ | Periodic forcing frequency | $0.1$ |
| $\lambda_s$ | Poisson spike rate (per node per step) | $0.02$ |
| $\sigma$ | Wiener process amplitude noise | $0.02$ |

---

## 12. Literature Positioning

### 12.1 Confirmed Prior Art

| NECF Concept | Exists As | Citation |
|---|---|---|
| Coupled phase oscillators | Kuramoto model | Kuramoto (1975); Strogatz (1994) |
| Lyapunov stability for oscillators | van Hemmen & Wreszinski | J. Stat. Phys. 72 (1993) |
| Adaptive coupling $K_{ij}(t)$ | Adaptive Kuramoto | Ha, Kim & Zhang, SIAM JADS (2016) |
| Co-evolving structure + dynamics | Neural Reuse | Bassett & Sporns (2023) |
| Hopfield + Kuramoto memory | Joint model | arXiv:2505.03648 (May 2025) |
| Free energy / surprise minimization | Free Energy Principle | Friston (2005–2022) |
| Curiosity as epistemic foraging | Active Inference | Friston (2017) |
| Meta-learning (learn to learn) | MAML | Finn, Abbeel & Levine (2017) |
| Dissipative self-organization | Dissipative structures | Prigogine (Nobel, 1977) |
| Self-modifying agent architecture | ADAS | Hu et al., ICLR (2025) |

### 12.2 The Precise Gap

The above components exist in isolation. What does not exist in any reviewed paper is their **specific integration**: a system where the *update rule* for local coupling, error sensitivity, and curiosity is itself a continuous dynamical variable governed by thermodynamically-smooth epistemic contagion, bounded by a scalar identity functional, and made reversible via Lyapunov-gated rollback — without backpropagation, discrete training phases, or external loss functions.

### 12.3 Target Venues

- **Physical Review E**: Derivation of $H[\mathcal{L}]$, Lyapunov spectrum, phase transition universality
- **NeurIPS Workshop on Emergence in Complex Systems**: Level-3 ablation study, meta-rule evolution trajectories
- **ICLR 2027**: Topological memory basins, finite-size scaling, edge-of-chaos regime analysis

---

## Citation

```bibtex
@misc{devanik2026necf,
  title        = {Non-Equilibrium Cognitive Field: Identity-Constrained Level-3
                  Meta-Rule Dynamics in Coupled Oscillator Fields},
  author       = {Devanik},
  year         = {2026},
  institution  = {NIT Agartala / IISc (Samsung Fellow)},
  url          = {https://github.com/Devanik21/Non-Equilibrium-Cognitive-Field},
  note         = {Research Preprint}
}
```

---

## Ethics & Scope

The NECF is a **proto-cognitive mathematical substrate**. It is not Artificial General Intelligence. It possesses no natural language comprehension, no spatial awareness beyond its own phase topology, and no goal-directed task logic. Claims of superiority over Level-1 and Level-2 systems are strictly mathematical and thermodynamic: NECF maintains internal coherence and topological diversity better under chaotic assault than static systems. Whether this translates to performance on standard machine learning benchmarks remains unproven and is the subject of ongoing work.

All source code is released under the **Apache 2.0 License** to encourage unfettered academic exploration and independent replication.

---

<div align="center">

*"Intelligence is not merely a function of the state; it is a function of the rules governing the state, adapting dynamically to the state."*

**Devanik** | NIT Agartala | Samsung Convergence Software Fellow, IISc | March 2026

</div>
