---
session_id: NECF-2026-100-T1
date: 2026-04-10
topic: Boltzmann Temperature Scan
seed: 20260410
N: 32
---

# NECF Session Log — 2026-04-10

**Session ID:** `NECF-2026-100-T1`
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
| 0.0100 | 1.11744 | 0.000068 | 0.03038 |
| 0.0243 | 1.26866 | 0.000077 | 0.06743 |
| 0.0591 | 2.77373 | 0.000095 | 0.04306 |
| 0.1436 | 3.31630 | 0.000089 | 0.00525 |
| 0.3490 | 3.42917 | 0.000072 | 0.05990 |
| 0.8483 | 3.45804 | 0.000082 | 0.03175 |
| 2.0620 | 3.46486 | 0.000062 | 0.03734 |
| 5.0119 | 3.46559 | 0.000057 | 0.08556 |

### 3.2 Key Parameters

| Quantity | Value |
|---|---|
| Uniform-weight entropy $\\ln N$ | 3.46574 |
| Optimal temperature $\\kappa^*$ | **2.77241** |
| $H_w(\\kappa^*)$ | 3.46509 |
| Transition temperature $\\kappa_c$ | **0.02431** |
| Discrimination ratio at $\\kappa^*$: $H_w/\\ln N$ | 0.9998 |

---

## 4. Interpretation

The transition from selective to diffuse contagion occurs at
$\\kappa_c = 0.0243$, identified as the peak of
$dH_w/d\\ln\\kappa$.  The optimal operating point
$\\kappa^* = 2.7724$ lies above
this transition, in the
diffuse regime.
At $\\kappa^*$, the weight entropy is $3.46509$, representing
100.0% of the theoretical maximum $\\ln N = 3.4657$.
The default NECF value $\\kappa = 0.10$ sits at $H_w = 3.1614$.

---

## 5. Connection to NECF

The choice $\\kappa = 0.10$ (current default) was validated against the
Fix\\#1 singularity analysis: for a field of $N=32$ nodes with typical
$\\varepsilon \\in [0.02, 0.35]$, the maximum Boltzmann weight at
$\\kappa=0.10$ is $\\approx 0.053$, versus $\\gg 10^4$ under the prior
inverse-error formula.  This session confirms that $\\kappa=0.10$
deviates from
one order of magnitude of the empirical optimum $\\kappa^* = 2.7724$.

---
*NECF-2026-100-T1 · generated 2026-04-10 · seed 20260410*
