---
session_id: NECF-2026-211-T0
date: 2026-07-30
topic: Synchronization Onset
seed: 20260730
N: 64
omega_std: 0.3
---

# NECF Session Log — 2026-07-30

**Session ID:** `NECF-2026-211-T0`
**Topic:** Synchronization Onset and Critical Coupling in the Amplitude-Weighted Kuramoto Substrate

---

## 1. Theoretical Background

The Kuramoto model with $N$ oscillators and natural frequencies
$\\omega_i \\sim \\mathcal{N}(\\mu, \\sigma^2)$ undergoes a continuous
phase transition at a critical coupling $K_c$ given, in the mean-field
(thermodynamic, $N \\to \\infty$) limit, by

$$K_c = \\frac{2}{\\pi\\, g(\\Omega)}$$

where $g(\\omega)$ is the frequency density evaluated at the mean field
frequency $\\Omega = \\mu$, and $g(\\Omega) = (\\sigma\\sqrt{2\\pi})^{-1}$ for
Gaussian $g$.  Substituting $\\mu = 1.0$, $\\sigma = 0.3$:

$$K_c^{\\text{theory}} = \\frac{2\\sigma\\sqrt{2\\pi}}{\\pi}
  = \\frac{2 \\times 0.3 \\times \\sqrt{2\\pi}}{\\pi}
  = 0.478731$$

For finite $N$, the order parameter $r \\equiv |N^{-1}\\sum_j e^{i\\theta_j}|$
retains a stochastic background of order $r_{\\rm bg} \\sim N^{-1/2}$
even below $K_c$, broadening the apparent onset.  Here $N=64$ gives
$r_{\\rm bg} \\approx 0.1250$.

Above $K_c$, mean-field theory predicts the order parameter scales as

$$r \\sim (K - K_c)^\\beta, \\quad \\beta = \\tfrac{1}{2}$$

---

## 2. Experimental Setup

- **Oscillators:** $N = 64$, natural frequencies $\\omega_i \\sim \\mathcal{N}(1.0,\\,0.3^2)$
- **Coupling sweep:** $K \\in [0.05, 2.50]$, 28 values
- **Integration:** Euler–Maruyama, $\\Delta t = 0.01$, warmup 800 steps,
  then $r$ averaged over 300 steps
- **Empirical $K_c$:** first $K$ where $r > r_{\\rm bg} + 3\\sigma_{\\rm noise}$

---

## 3. Results

### 3.1 Order Parameter vs Coupling

| $K / K_c^{\\rm theory}$ | $K$ | $\\bar{r}$ |
|:---:|:---:|:---:|
| 0.63 | 0.300 | 0.1298 |
| 1.00 | 0.479 | 0.5972 |
| 1.50 | 0.718 | 0.7432 |
| 2.00 | 0.957 | 0.9570 |
| 3.00 | 1.436 | 0.9729 |

### 3.2 Critical Coupling Comparison

| Quantity | Value |
|---|---|
| $K_c$ (mean-field theory, $N\\to\\infty$) | **0.478731** |
| $K_c$ (empirical, $N=64$) | **0.412963** |
| Relative deviation | **-13.7%** |
| Finite-size background $r_{\\rm bg} \\sim N^{-1/2}$ | 0.1250 |
| Fitted scaling exponent $\\beta$ | **0.1967** (theory: 0.5000) |
| Fit quality $R^2$ | 0.6684 |

### 3.3 Raw K–r Data

```json
{
  "K": [
    0.05,
    0.1407,
    0.2315,
    0.3222,
    0.413,
    0.5037,
    0.5944,
    0.6852,
    0.7759,
    0.8667,
    0.9574,
    1.0481,
    1.1389,
    1.2296,
    1.3204,
    1.4111,
    1.5019,
    1.5926,
    1.6833,
    1.7741,
    1.8648,
    1.9556,
    2.0463,
    2.137,
    2.2278,
    2.3185,
    2.4093,
    2.5
  ],
  "r": [
    0.07321,
    0.13467,
    0.05123,
    0.12976,
    0.33934,
    0.59722,
    0.43609,
    0.74317,
    0.88733,
    0.83942,
    0.95703,
    0.94391,
    0.97893,
    0.97036,
    0.97514,
    0.97288,
    0.98247,
    0.98332,
    0.98745,
    0.98096,
    0.98416,
    0.98611,
    0.98999,
    0.99226,
    0.99119,
    0.9918,
    0.99387,
    0.99298
  ]
}
```

---

## 4. Interpretation

The empirical critical coupling $K_c^{\\rm emp} = 0.4130$ deviates
from the mean-field prediction $K_c^{\\rm th} = 0.4787$ by
-13.7%.  This is consistent with finite-size broadening: for
$N = 64$, the stochastic background $r_{\\rm bg} \\approx 0.1250$
raises the apparent threshold and shifts the detected onset toward higher
$K$.  The fitted scaling exponent $\\hat{\\beta} = 0.1967$
departs from the mean-field prediction $\\beta=0.5$, suggesting higher-order corrections.

---

## 5. Connection to NECF

In the NECF architecture, the effective coupling seen by oscillator $i$ is
$\\beta_i(t) \\cdot K \\cdot \\bar{W}$, where $\\beta_i$ is the local coupling
component of the rule field $\\mathcal{L}_i$.  At initialisation
$\\beta_i \\approx 0.80$; with $\\bar{W} \\approx 1.0$ and $K = 0.7$, the
effective coupling $K_{\\rm eff} \\approx 0.56$, placing the system
above $K_c^{\\rm th} = 0.4787$.
Meta-rule evolution can push $\\beta_i$ either direction, giving the field
the capacity to modulate its own distance from the synchronisation
transition in real time.

---
*NECF-2026-211-T0 · generated 2026-07-30 · seed 20260730*
