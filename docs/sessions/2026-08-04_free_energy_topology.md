---
session_id: NECF-2026-216-T5
date: 2026-08-04
topic: Free Energy Topology
seed: 20260804
N: 32
K: 0.8
---

# NECF Session Log — 2026-08-04

**Session ID:** `NECF-2026-216-T5`
**Topic:** Free Energy Landscape Topology: Attractor Basin Counting and Lorenz-Perturbation Escape Rates

---

## 1. Theoretical Background

The Kuramoto free energy (Lyapunov functional) is

$$F[\\theta] = -\\frac{K}{2N}\\sum_{i,j}\\cos(\\theta_i - \\theta_j)$$

Stable fixed points are local minima of $F$; the number of distinct
minima equals the number of attractor basins accessible from random
initial conditions.  In the presence of open-system driving (Lorenz
perturbation), the system may escape a local minimum with rate

$$k_{\\rm esc} = \\frac{\\text{trials where }
  r_{\\rm after} < 0.5\\,r_{\\rm before}}{N_{\\rm trials}}$$

A high escape rate indicates a shallow free-energy landscape;
low escape rate indicates deep, robust attractors.

---

## 2. Experimental Setup

- $N = 32$, $K = 0.8$, $T = 600$ settling steps, 40 trials
- **Basin detection:** final mean phases $\\psi$ clustered by circular
  distance $|\\psi_i - \\psi_j| < 0.4$ rad
- **Escape test:** 50-step Lorenz kick with amplitude 0.10,
  escape declared if $r_{\\rm after} < 0.5\\,r_{\\rm before}$

---

## 3. Results

| Metric | Value |
|---|---|
| Distinct attractor basins detected | **9** |
| Basin sizes (trials per basin) | [7, 6, 6, 5, 4, 4, 3, 3, 2] |
| Mean final $r$ across all trials | 0.63409 ± 0.19457 |
| Lorenz-perturbation escape rate $k_{\\rm esc}$ | **0.0000** (0/40 trials) |
| Landscape character | multi-basin (complex topology) |

---

## 4. Interpretation

The free energy landscape at $K = 0.8$, $N = 32$ exhibits
**9 distinct attractor basins**
from 40 random initialisations.
Basins are roughly equally populated, indicating a nearly degenerate landscape.
The Lorenz escape rate $k_{\\rm esc} = 0.0000$ implies that
the attractor is robust under chaotic perturbation (most trajectories return).

---

## 5. Connection to NECF

The CuriosityEngine in NECF deliberately injects Lorenz-phase kicks
to prevent premature attractor lock-in.  With the measured
$k_{\\rm esc} = 0.0000$, a curiosity kick amplitude of 0.10
escapes the field's current basin in approximately
$1/k_{\\rm esc} = inf$ such interventions.
This sets the minimum intervention budget for the CuriosityEngine's
REFRAME directive.

---
*NECF-2026-216-T5 · generated 2026-08-04 · seed 20260804*
