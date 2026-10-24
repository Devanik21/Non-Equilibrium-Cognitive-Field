---
session_id: NECF-2026-297-T2
date: 2026-10-24
topic: Identity Stability Landscape
seed: 20261024
N: 32
---

# NECF Session Log — 2026-10-24

**Session ID:** `NECF-2026-297-T2`
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
| 0.01 | ↓ r=0.016 | ↓ r=0.016 | ↓ r=0.023 | ↓ r=0.011 | ↓ r=0.031 |
| 0.05 | ✓ r=0.043 | ✓ r=0.073 | ✓ r=0.073 | ✓ r=0.050 | ✓ r=0.067 |
| 0.10 | ✓ r=0.087 | ✓ r=0.077 | ↓ r=0.037 | ↓ r=0.023 | ✓ r=0.065 |
| 0.25 | ✓ r=0.099 | ✓ r=0.062 | ✓ r=0.049 | ✓ r=0.131 | ✓ r=0.043 |
| 0.50 | ✓ r=0.048 | ↓ r=0.040 | ✓ r=0.045 | ✓ r=0.068 | ↓ r=0.022 |
| 1.00 | ↓ r=0.037 | ✓ r=0.062 | ✓ r=0.093 | ✓ r=0.060 | ↓ r=0.039 |

### 3.2 Summary Statistics

| Quantity | Value |
|---|---|
| Viable fraction | **63.3%** (19/30 cells) |
| Best operating point $(\lambda^*, \\delta^*)$ | $(0.25,\\;0.5)$ |
| $r$ at best point | 0.13059 |
| Cells classified DRIFTED ($\\mathcal{H} > 3$) | 0 |
| Cells classified CATATONIC ($r < 0.04$) | 11 |
| Cells classified ROLLBACK-HEAVY | 0 |

---

## 4. Interpretation

63.3% of the explored $(\lambda, \\delta)$ parameter space is
viable at $T=400$ steps.  The identity gradient is most permissive
at intermediate λ (balanced constraint);
the rollback mechanism is most effective
at moderate δ (selective rollback).
The default NECF values (λ=0.10, δ=0.30) fall in the
CATATONIC
cell of this grid.

---
*NECF-2026-297-T2 · generated 2026-10-24 · seed 20261024*
