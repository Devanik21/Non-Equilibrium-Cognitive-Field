---
session_id: NECF-2026-192-T2
date: 2026-07-11
topic: Identity Stability Landscape
seed: 20260711
N: 32
---

# NECF Session Log — 2026-07-11

**Session ID:** `NECF-2026-192-T2`
**Topic:** Identity Curvature H[L] Stability Landscape: Phase Diagram in (λ, δ_thresh) Space

---

## 1. Theoretical Background

The identity curvature functional

$$\\mathcal{H}[\\mathcal{L}] = \\frac{1}{N}\\sum_i
  \\|\\mathcal{L}_i - \\mathcal{L}_i^{(0)}\\|^2
  \\;+\\; \\kappa \\,\\text{Var}(\\mathcal{L}_i)$$

enters the meta-dynamics as a gradient constraint:

$$\\frac{d\\mathcal{L}_i}{dt} =
  F(\\mathcal{L}_i,\\varepsilon_i) - \\lambda\\,
  \\nabla_{\\mathcal{L}_i}\\mathcal{H}$$

Two parameters control identity preservation:

- $\\lambda$: strength of the identity gradient term.
  Large $\\lambda$ keeps rules close to initialisation (stability)
  but slows adaptation (inflexibility).
- $\\delta_{\\rm thresh}$: rollback threshold on $\\Delta\\mathcal{H}$.
  Small $\\delta$ rolls back aggressively, preventing drift but
  potentially inducing oscillations.

The viable regime is the set $(\lambda, \\delta)$ where:
$r > r_{\\rm bg}$ (not catatonic),
$\\mathcal{H} < H_{\\rm max}$ (not drifted), and
rollback rate $< 15\\%$ (not rollback-heavy).

---

## 2. Experimental Setup

- $N = 32$, $T = 400$ steps, $\\kappa = 0.10$, $K = 0.70$
- $\\lambda \\in {0.01, 0.05, 0.1, 0.25, 0.5, 1.0}$
- $\\delta_{\\rm thresh} \\in {0.1, 0.2, 0.3, 0.5, 0.8}$
- Regimes: ✓ VIABLE · ↓ CATATONIC · ↑ DRIFTED · ⟳ ROLLBACK-HEAVY

---

## 3. Results

### 3.1 Stability Landscape Table

| λ \\ δ_thresh | 0.10 | 0.20 | 0.30 | 0.50 | 0.80 |
|---|---|---|---|---|---|
| 0.01 | ✓ r=0.105 | ✓ r=0.052 | ✓ r=0.049 | ✓ r=0.093 | ✓ r=0.058 |
| 0.05 | ✓ r=0.066 | ↓ r=0.016 | ✓ r=0.097 | ↓ r=0.033 | ↓ r=0.028 |
| 0.10 | ↓ r=0.031 | ↓ r=0.032 | ↓ r=0.028 | ↓ r=0.014 | ↓ r=0.013 |
| 0.25 | ✓ r=0.055 | ✓ r=0.050 | ↓ r=0.011 | ↓ r=0.031 | ↓ r=0.018 |
| 0.50 | ✓ r=0.053 | ✓ r=0.072 | ↓ r=0.017 | ↓ r=0.026 | ↓ r=0.021 |
| 1.00 | ↓ r=0.030 | ✓ r=0.070 | ✓ r=0.053 | ↓ r=0.038 | ✓ r=0.062 |

### 3.2 Summary Statistics

| Quantity | Value |
|---|---|
| Viable fraction | **46.7%** (14/30 cells) |
| Best operating point $(\lambda^*, \\delta^*)$ | $(0.01,\\;0.1)$ |
| $r$ at best point | 0.10527 |
| Cells classified DRIFTED ($\\mathcal{H} > 3$) | 0 |
| Cells classified CATATONIC ($r < 0.04$) | 16 |
| Cells classified ROLLBACK-HEAVY | 0 |

---

## 4. Interpretation

46.7% of the explored $(\lambda, \\delta)$ parameter space is
viable at $T=400$ steps.  The identity gradient is most permissive
at low λ (weak constraint);
the rollback mechanism is most effective
at low δ (aggressive rollback).
The default NECF values (λ=0.10, δ=0.30) fall in the
CATATONIC
cell of this grid.

---
*NECF-2026-192-T2 · generated 2026-07-11 · seed 20260711*
