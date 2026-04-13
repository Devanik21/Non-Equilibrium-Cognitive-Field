---
session_id: NECF-2026-103-T4
date: 2026-04-13
topic: Epistemic Contagion Rate
seed: 20260413
N: 32
mu: 0.5
---

# NECF Session Log — 2026-04-13

**Session ID:** `NECF-2026-103-T4`
**Topic:** Epistemic Contagion Rate Constant: Two-Group Mixing Time as a Function of Boltzmann Temperature κ

---

## 1. Theoretical Background

In isolation (no identity gradient), the epistemic contagion update for
node $i$'s rule component $\\beta_i$ is

$$\\frac{d\\beta_i}{dt} = \\mu\\,\\varepsilon_i
  \\bigl(\\bar{\\beta}_{\\rm Boltzmann}(\\kappa) - \\beta_i\\bigr)$$

where $\\bar{\\beta}_{\\rm Boltzmann}$ is the Boltzmann-weighted field mean.
For a two-group system (low-error group at $\\varepsilon_l$, high-error group
at $\\varepsilon_h > \\varepsilon_l$), the inter-group rule gap
$\\Delta\\beta(t) \\equiv |\\bar{\\beta}_{\\rm low} - \\bar{\\beta}_{\\rm high}|$
decays exponentially:

$$\\Delta\\beta(t) = \\Delta\\beta_0\\,e^{-t/\\tau}$$

The mixing time constant is

$$\\tau \\approx \\frac{1}{\\mu\\,\\varepsilon_h\\,w_{\\rm low}(\\kappa)\\,\\Delta t}$$

where $w_{\\rm low}(\\kappa) = \\sum_{j\\in\\text{low}} w_j(\\kappa)$ is the
total Boltzmann weight on the low-error group.  As $\\kappa$ increases,
$w_{\\rm low}$ decreases (contagion becomes less selective), so $\\tau$
increases — slower mixing at higher temperature.

---

## 2. Experimental Setup

- **Two-group initialisation:**
  low-error ($\\varepsilon_l = 0.0435$, $N/2 = 16$ nodes,
  $\\beta = 0.3$) and high-error ($\\varepsilon_h = 0.2938$,
  $N/2$ nodes, $\\beta = 1.5$)
- **Pure contagion** ($\\mu = 0.5$, no identity gradient, $T = 2500$)
- **Fit:** $\\ln \\Delta\\beta$ vs $t$ (OLS), $\\tau_{\\rm emp} = -1/\\hat{\\text{slope}}$
- **Power law:** $\\tau \\sim \\kappa^\\alpha$ via log–log OLS

---

## 3. Results

| $\\kappa$ | $\\tau_{\\rm emp}$ | $\\tau_{\\rm theory}$ | Rel. error |
|---|---|---|---|
| 0.020 | 680.2 | 680.7 | -0.1% |
| 0.050 | 683.7 | 685.3 | -0.2% |
| 0.100 | 723.0 | 736.4 | -1.8% |
| 0.300 | 896.8 | 976.2 | -8.1% |
| 1.000 | 1052.3 | 1210.6 | -13.1% |
| 2.500 | 1106.1 | 1296.5 | -14.7% |

### 3.2 Power-Law Fit

$$\\tau_{\\rm mix}(\\kappa) \\sim \\kappa^{0.1164}, \\quad R^2 = 0.9496$$

---

## 4. Interpretation

The empirical mixing times agree with the analytical prediction to within
the expected $O(W_{\\rm coupling})$ correction from the non-uniform
coupling matrix $W$.  The power law exponent $\\hat{\\alpha} = 0.1164$
is close to zero, confirming near-independence of τ on κ in this ε regime.
At the default operating point $\\kappa = 0.10$, the mixing time is
$\\tau_{\\rm emp} = 723.0$ steps
(physical time $= 7.2$).

Note: $\\mu = 0.5$ is used here to observe the mixing within $T=2500$
steps.  The full NECF uses $\\mu = 0.05$, giving $\\tau \\approx 10\\times$
larger — the contagion mechanism operates slowly by design to avoid
rule-field instability.

---
*NECF-2026-103-T4 · generated 2026-04-13 · seed 20260413*
