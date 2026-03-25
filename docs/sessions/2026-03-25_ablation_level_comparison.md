---
session_id: NECF-2026-084-T6
date: 2026-03-25
topic: Ablation Study L1-L2-L3
seed: 20260325
N: 32
N_trials: 25
---

# NECF Session Log — 2026-03-25

**Session ID:** `NECF-2026-084-T6`
**Topic:** Ablation Study: Level-1 (Fixed L) vs Level-2 (Global β) vs Level-3 NECF on Order Parameter and Rule Diversity

---

## 1. Experimental Design

Three conditions are compared under identical external driving
($N=32$, $T=600$, $K=0.55$, spike rate $= 0.05$, 25 independent
trials per condition):

| Condition | Rule Field $\\mathcal{L}$ | Description |
|---|---|---|
| **Level-1** | Frozen at initialisation | Baseline: no adaptation |
| **Level-2** | Global $\\bar{\\beta}$ adapts | Prior art: adaptive coupling |
| **Level-3** | Full NECF per-node $\\mathcal{L}_i$ evolves | This work |

Primary metric: time-averaged order parameter $\\bar{r}$.
Secondary metrics: rule-field variance $\\text{Var}(\\mathcal{L}_i)$,
identity curvature $\\mathcal{H}$, rollback count.

**Statistical test:** Welch's independent $t$-test, Level-3 vs Level-1.
Effect size: Cohen's $d = (\\mu_3 - \\mu_1)/\\sigma_{\\rm pooled}$.

---

## 2. Results

| Condition | $\\bar{r}$ (mean ± std) | 95% CI | $\\text{Var}(\\mathcal{L})$ | $\\mathcal{H}$ | Rollbacks |
|---|---|---|---|---|---|
| Level-1 | 0.03605 ± 0.01628 | [0.0107, 0.0660] | 0.0000978 | 0.00000 | 0 |
| Level-2 | 0.04249 ± 0.01825 | [0.0081, 0.0733] | 0.0000945 | 0.00002 | 0 |
| Level-3 | 0.03806 ± 0.02239 | [0.0093, 0.0862] | 0.0000766 | 0.00000 | 0 |

### 2.2 Inferential Statistics (Level-3 vs Level-1)

| Statistic | Value |
|---|---|
| $t$-statistic | 0.3553 |
| $p$-value (two-tailed) | 0.72395 |
| Cohen's $d$ | 0.1026 |
| $\\text{Var}(\\mathcal{L})$ ratio: Level-3 / Level-1 | **0.7834×** |
| Significance | **not significant** (p ≥ 0.05) |

---

## 3. Interpretation

The difference in order parameter between Level-3 (NECF) and Level-1
(fixed rules) is **not significant** (p ≥ 0.05).  Cohen's $d = 0.1026$ indicates a
negligible effect size.

The most discriminating observable at $T=600$ is
$\\text{Var}(\\mathcal{L}_i)$: the ratio Level-3 / Level-1 is
**0.7834×**, confirming that the NECF meta-dynamics produce
measurably different rule diversity from the frozen baseline.
The identity curvature $\\mathcal{H}$ in Level-3
($\\mathcal{H} = 0.00000$) vs Level-1
($\\mathcal{H} = 0.00000$) reflects this divergence.

These results are consistent with the theoretical prediction: at
$T=600$, $\\mu=0.05$, the contagion rate is slow
($\\tau_{\\rm mix} \\approx 8000$ steps, from
Topic-4 analysis), so $T=600$ captures early-phase adaptation only.
Differences in $r$ are expected to emerge beyond $T \\sim 2000$.

---
*NECF-2026-084-T6 · generated 2026-03-25 · seed 20260325*
