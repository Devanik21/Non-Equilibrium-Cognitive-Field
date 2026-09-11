---
session_id: NECF-2026-254-T1
date: 2026-09-11
topic: Boltzmann Temperature Scan
seed: 20260911
N: 32
---

# NECF Session Log — 2026-09-11

**Session ID:** `NECF-2026-254-T1`
**Topic:** Boltzmann Contagion Temperature Scan: Optimal κ for Rule-Field Diffusion

---

## 1. Theoretical Background

In NECF's epistemic contagion mechanism, the influence weight of node $j$
on node $i$ follows a Boltzmann (softmax) distribution over prediction
errors:

$$w_j(\\kappa) = \\frac{e^{-\\varepsilon_j/\\kappa}}
  {\\sum_k e^{-\\varepsilon_k/\\kappa}}, \\quad \\kappa > 0$$

The information content of this distribution is quantified by the weight
entropy

$$H_w(\\kappa) = -\\sum_j w_j \\ln w_j \\in [0,\\, \\ln N]$$

Two limiting regimes:

- **$\\kappa \\to 0$:** $w \\to \\delta_{j^*}$ (winner-takes-all),
  $H_w \\to 0$.  Contagion is maximally selective but brittle.
- **$\\kappa \\to \\infty$:** $w \\to N^{-1}$ (uniform),
  $H_w \\to \\ln N = 3.4657$.  Contagion is diffuse; no
  discrimination between good and bad rules.

A thermodynamically optimal $\\kappa^*$ balances discrimination power
against field-level diversity, maximising $H_w(\\kappa)\\cdot r(\\kappa)$.

---

## 2. Experimental Setup

- **Field:** $N = 32$, NECF run to $T = 350$ steps per $\\kappa$
- **$\\kappa$ grid:** 22 values log-spaced in $[0.01, 5.00]$
- **Optimal $\\kappa^*$:** $\\arg\\max_\\kappa H_w(\\kappa) \\cdot r(\\kappa)$
- **Transition point $\\kappa_c$:** $\\arg\\max_\\kappa \\frac{dH_w}{d\\ln\\kappa}$

---

## 3. Results

### 3.1 Temperature Scan Summary

| $\\kappa$ | $H_w$ | $\\text{Var}(\\mathcal{L}_i)$ | $r$ |
|---|---|---|---|
| 0.0100 | 0.39318 | 0.000081 | 0.07993 |
| 0.0243 | 1.86986 | 0.000076 | 0.02757 |
| 0.0591 | 2.82468 | 0.000094 | 0.03201 |
| 0.1436 | 3.24255 | 0.000089 | 0.04637 |
| 0.3490 | 3.42778 | 0.000064 | 0.07848 |
| 0.8483 | 3.45821 | 0.000064 | 0.08315 |
| 2.0620 | 3.46471 | 0.000083 | 0.04438 |
| 5.0119 | 3.46553 | 0.000084 | 0.04604 |

### 3.2 Key Parameters

| Quantity | Value |
|---|---|
| Uniform-weight entropy $\\ln N$ | 3.46574 |
| Optimal temperature $\\kappa^*$ | **0.84834** |
| $H_w(\\kappa^*)$ | 3.45821 |
| Transition temperature $\\kappa_c$ | **0.01000** |
| Discrimination ratio at $\\kappa^*$: $H_w/\\ln N$ | 0.9978 |

---

## 4. Interpretation

The transition from selective to diffuse contagion occurs at
$\\kappa_c = 0.0100$, identified as the peak of
$dH_w/d\\ln\\kappa$.  The optimal operating point
$\\kappa^* = 0.8483$ lies above
this transition, in the
diffuse regime.
At $\\kappa^*$, the weight entropy is $3.45821$, representing
99.8% of the theoretical maximum $\\ln N = 3.4657$.
The default NECF value $\\kappa = 0.10$ sits at $H_w = 3.1447$.

---

## 5. Connection to NECF

The choice $\\kappa = 0.10$ (current default) was validated against the
Fix\\#1 singularity analysis: for a field of $N=32$ nodes with typical
$\\varepsilon \\in [0.02, 0.35]$, the maximum Boltzmann weight at
$\\kappa=0.10$ is $\\approx 0.053$, versus $\\gg 10^4$ under the prior
inverse-error formula.  This session confirms that $\\kappa=0.10$
deviates from
one order of magnitude of the empirical optimum $\\kappa^* = 0.8483$.

---
*NECF-2026-254-T1 · generated 2026-09-11 · seed 20260911*
