<div align="center">

# Non-Equilibrium Cognitive Field (NECF)
## Experimental Research Notebooks

**Author:** Devanik | B.Tech ECE '26, NIT Agartala
**Framework:** Level-3 Meta-Rule Dynamics & Boltzmann Epistemic Contagion
**Computational Engine:** Batched PyTorch (CUDA T4-Optimized)

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)]()
[![Status](https://img.shields.io/badge/Research-Active-7c3aed?style=for-the-badge)]()

</div>

---

## 1. Abstract & Context

Most adaptive dynamical systems operate at a maximum of two levels: a state that changes (e.g., node phases $\phi_i$) and a rule that governs the state (e.g., coupling strengths $K_{ij}$). The update rule for $K_{ij}$ is almost always fixed (Level-2).

This repository explores **Level-3 Meta-Rule Dynamics**: what if the rules governing adaptation—specifically error sensitivity ($\alpha$), coupling strength ($\beta$), and uncertainty-seeking curiosity ($\gamma$)—are themselves continuously evolving dynamic variables?

The **Non-Equilibrium Cognitive Field (NECF)** architecture treats the learning rules $\mathcal{L}_i = (\alpha_i, \beta_i, \gamma_i)$ as a spatial field. This field evolves via **Boltzmann-weighted epistemic contagion** (where successful rules diffuse thermodynamically across the network) and is strictly bounded by an **Identity Curvature Functional $H[\mathcal{L}]$** (which prevents chaotic drift and catatonic homogenization).

The notebooks contained in this directory are mathematically rigorous, "0-cheat", $N=64$ coupled-oscillator simulations designed to run on Google Colab T4 GPUs. They empirically prove the structural superiority of Level-3 dynamics over traditional baselines.

---

## 2. Notebook Inventory

### 📄 `NECF_Research_Notebook_2.ipynb`
The foundational GPU-accelerated empirical verification.
*   **Backend:** PyTorch tensor operations over a batch dimension ($B=50$ to $B=200$).
*   **Scale:** $T=10,000$ simulation steps.
*   **Content:** Implements the core 7 falsifiable predictions of the NECF system. It maps the critical coupling $K_c$ phase transitions, sweeps the Boltzmann temperature $\kappa$, grids the identity landscape against rollbacks $\delta_{thresh}$, and performs the core $L1$ vs $L2$ vs $L3$ ablation study. It calculates the Kaplan-Yorke fractal dimension $D_{KY}$ to prove bounded chaos.

### 📄 `NECF_Research_Notebook_Final.ipynb`
The vastly expanded, massively comprehensive research artifact.
*   **Content:** Contains all 7 foundational experiments, but expands them with deep academic framing, formal LaTeX equations, and explicit physical analogies (Kuramoto oscillators, Prigogine dissipative structures).
*   **Advanced Additions:**
    *   **E8:** Maps the true 3D spatial trajectories of the meta-rules $(\alpha, \beta, \gamma)$ across time, proving the existence of a stable, "breathing" meta-attractor.
    *   **E9:** Computes the **Free Energy Topology**, running $B=200$ independent trials and utilizing $K$-Means clustering and PCA on the final steady states to mathematically count the number of "identities" (attractor basins) the system can functionally hold in its memory.
    *   **E10:** Conducts an Environmental Driver Ablation, sequentially stripping away Lorenz chaos, periodic signals, and Poisson spikes to prove that non-equilibrium thermodynamics is mathematically essential for continuous Level-3 adaptation.

---

## 3. Experimental Index (E1-E10)

These notebooks execute the following rigorous empirical tests:

| Experiment | Focus | Core Mathematical Insight |
|:---|---|---|
| **E1** | **Synchronization Onset ($K$ sweep)** | Tests mean-field universality: $r \sim (K-K_c)^\beta$ |
| **E2** | **Boltzmann Temperature Scan** | Finds optimal rule diffusion $\kappa^* = \arg\max H_w(\kappa) \cdot r(\kappa)$ |
| **E3** | **Identity Stability Landscape** | Maps the "viable" regime of identity preservation $H[\mathcal{L}] \in (0.1, 5.0)$ |
| **E4** | **Main Ablation Study** | Proves $L3$ significantly outperforms $L1, L2$ via Cohen's $d$ and $p < 0.05$ |
| **E5** | **Lyapunov Spectrum (Continuous QR)** | Confirms bounded chaos: Max Lyapunov $\lambda_1 \in (-0.5, 0.8)$ |
| **E6** | **Epistemic Contagion Rate** | Verifies the theoretical diffusion power law $\tau_{mix} \sim \kappa^{0.12}$ |
| **E7** | **Predictions Dashboard** | Synthesizes all data to verify the 7 falsifiable hypotheses |
| **E8** | **3D Meta-Rule Trajectories** | Physically maps the non-collapsing, non-diverging rule attractor |
| **E9** | **Free Energy Topology (K-Means)** | Quantifies distinct "memory" attractor basins across $B=200$ trials |
| **E10**| **Thermodynamic Driver Ablation** | Proves the necessity of open-system perturbations (Lorenz, Spikes) |

---

## 4. Hardware & Execution Guidelines

**Requirements:**
*   A CUDA-capable GPU is highly recommended (Google Colab T4 is the target environment).
*   `torch`, `numpy`, `scipy`, `matplotlib`, `scikit-learn`

**Performance Notes:**
Because the system simulates independent, $N=64$ fully-coupled networks, the computational complexity is $O(B \cdot N^2)$. The custom `NECFBatched` PyTorch class parallelizes this across the `B` dimension.
*   **E4 (Ablation):** Runs 50 parallel trials ($B=50$) for 10,000 steps. Takes ~45 seconds on a T4 GPU.
*   **E9 (Topology):** Runs 200 parallel trials ($B=200$) for 1,500 steps.

To execute the code, simply upload the notebooks to Google Colab, change the Runtime type to **T4 GPU**, and select "Run All".

---
<div align="center">
<i>"Intelligence is not a function of the state; it is a function of the rules governing the state, adapting to the state."</i>
</div>
