---
session_id: NECF-2026-200-T3
date: 2026-07-19
topic: Lyapunov Spectrum QR
seed: 20260719
N: 16
K: 0.7243
---

# NECF Session Log — 2026-07-19

**Session ID:** `NECF-2026-200-T3`
**Topic:** Lyapunov Spectrum via Continuous QR Decomposition: Dynamical Complexity of the NECF Substrate

---

## 1. Method

The maximal Lyapunov exponent $\\lambda_1$ and the full spectrum
$\\lambda_1 \\geq \\lambda_2 \\geq \\cdots \\geq \\lambda_N$ are computed via
the **continuous QR decomposition** method (Benettin et al. 1980).

Starting from an orthonormal frame $Q^{(0)} = I$, at each step:

$$Q^{(t+1)} = Q_{\\rm new},\\quad R^{(t+1)}
  \\gets \\text{QR}\\bigl(Q^{(t)} + \\Delta t\\, J(\\theta^{(t)})
  Q^{(t)}\\bigr)$$

The Lyapunov exponents are accumulated as

$$\\lambda_i = \\frac{1}{T \\Delta t}
  \\sum_{t=1}^{T} \\ln |R_{ii}^{(t)}|$$

where the Jacobian of the Kuramoto flow is

$$J_{ij} = \\begin{cases}
  \\frac{K}{N}\\cos(\\theta_j - \\theta_i) & i \\neq j \\\\
  -\\frac{K}{N}\\sum_{k \\neq i}\\cos(\\theta_k - \\theta_i) & i = j
\\end{cases}$$

**Kaplan–Yorke dimension:**

$$D_{\\rm KY} = j + \\frac{\\sum_{k=1}^j \\lambda_k}{|\\lambda_{j+1}|}$$

where $j$ is the largest index such that $\\sum_{k=1}^j \\lambda_k \\geq 0$.

---

## 2. Parameters

- $N = 16$ oscillators, $\\omega_i \\sim \\mathcal{N}(1.0, 0.3^2)$
- Coupling $K = 0.7243$ (sampled with seed-based variation)
- Warmup: 500 steps;  accumulation: 800 steps
- $\\Delta t = 0.01$

---

## 3. Results

### 3.1 Lyapunov Spectrum (first 8 exponents)

| Exponent | Value | Character |
|---|---|---|
| $\lambda_{{ 1 }}$ | -0.17325 | negative — contracting direction
| $\lambda_{{ 2 }}$ | -0.33335 | negative — contracting direction
| $\lambda_{{ 3 }}$ | -0.56951 | negative — contracting direction
| $\lambda_{{ 4 }}$ | -0.61916 | negative — contracting direction
| $\lambda_{{ 5 }}$ | -0.62278 | negative — contracting direction
| $\lambda_{{ 6 }}$ | -0.62544 | negative — contracting direction
| $\lambda_{{ 7 }}$ | -0.63448 | negative — contracting direction
| $\lambda_{{ 8 }}$ | -0.64374 | negative — contracting direction

### 3.2 Summary Statistics

| Quantity | Value |
|---|---|
| $\\lambda_1$ (maximal) | **-0.17325** |
| $\\sum_i \\lambda_i$ (Lyapunov sum) | -9.52781 |
| $\\text{tr}(J)$ at final $\\theta$ (consistency check) | -10.28564 |
| Kaplan–Yorke dimension $D_{\\rm KY}$ | **0.0000** |
| Number of positive exponents | 0 / 16 |
| Dynamical regime | Stable (λ₁ ≤ 0) |

---

## 4. Interpretation

With $K = 0.7243$ and $N = 16$, the system is in the
**stable** regime
($\\lambda_1 = -0.1732$).
The Kaplan–Yorke dimension $D_{\\rm KY} = 0.000$ estimates the fractal
dimension of the attractor.  A value close to or below N/2 indicates a contracting, low-dimensional attractor.
The Lyapunov sum $\\sum_i \\lambda_i = -9.5278$ compared to
$\\text{tr}(J) = -10.2856$ confirms internal consistency of the
QR accumulation (these quantities are related but not identical due
to finite-time averaging).

---

## 5. Connection to NECF

The masked Lyapunov proxy in `observer.py` estimates $\\lambda_1$ by
tracking $\\|\\delta\\theta\\|$ on non-spiked nodes only (Fix\\#2).
This session's QR-derived $\\lambda_1 = -0.1732$ provides a
ground-truth reference: the proxy should stay within
$O(\\sqrt{N})$ of this value for the viable cognitive regime.
NECF's meta-dynamics target $\\lambda_1 \\in (-0.50,\\, +0.80)$,
corresponding to bounded chaos — neither frozen nor exploding.

---
*NECF-2026-200-T3 · generated 2026-07-19 · seed 20260719*
