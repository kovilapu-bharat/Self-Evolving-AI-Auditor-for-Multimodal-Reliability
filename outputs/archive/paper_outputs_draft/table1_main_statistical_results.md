# Table 1: Main Results and Statistical Comparison Table

**Repository:** `SEAA (Self-Evolving AI Auditor for Multimodal Reliability)`  
**Benchmark:** MM-IMDb (4,000 Evaluations Across 8 Canonical Failure Regimes; 500 Unique Clustered Movie Entities)  
**Statistical Design:** Repeated-measures paired analysis clustered at the movie entity level ($N=500$, $df=499$).  
**Cluster Dependence Metrics:** Intraclass Correlation Coefficient $\rho_{\text{ICC}} = 0.1319$, Variance Inflation Factor $\text{VIF} = 1.9232$, Effective Sample Size $N_{\text{eff}} \approx 2,080$.

| Policy / Model Configuration | Condition-Mean Micro-F1 | Condition-Mean Macro-F1 | Globally Pooled Micro-F1 | Instance-Level Mean F1 | Delta vs Unrec. | Clustered 95% CI vs Unrec | Clustered Paired $t$-test ($df=499$) | Wilcoxon Signed-Rank Test | Bootstrap $p$-value ($B=5000$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **SEAA Adaptive Recovery Policy** | **0.4709** | **0.3611** | **0.4690** | **0.4671** | **+0.0697 (+17.4%)** | **[+0.0619, +0.0843]** | **$t = 12.79, p = 1.34 \times 10^{-32}$** | **$W = 23,564.0, p = 1.15 \times 10^{-31}$** | **$p < 0.0002$** |
| **No Recovery (Static Late Fusion)** | 0.4012 | 0.3165 | 0.3973 | 0.3940 | Reference (0.0000) | --- | --- | --- | --- |
| **Fixed Heuristic Routing Policy** | 0.3942 | 0.2927 | 0.3905 | 0.3901 | -0.0070 (-1.7%) | [-0.0185, +0.0045] | $t = -1.20, p = 0.231$ | $W = 39,210.0, p = 0.245$ | --- |
| **Ablation: Learning Disabled (Untrained)** | 0.4377 | 0.3361 | 0.4348 | 0.4340 | +0.0365 (+9.1%) | [+0.0270, +0.0460] | $t = 7.55, p = 2.11 \times 10^{-13}$ | $W = 31,450.0, p = 4.88 \times 10^{-13}$ | $p < 0.0002$ |
| **Ablation: Diagnosis Features Removed** | 0.5054 | 0.3714 | 0.5056 | 0.5042 | +0.1042 (+26.0%) | [+0.0934, +0.1150] | $t = 18.91, p = 4.12 \times 10^{-60}$ | $W = 15,210.0, p = 3.20 \times 10^{-56}$ | $p < 0.0002$ |
| **Theoretical Ceiling: Oracle Upper Bound** | 0.5695 | 0.4421 | 0.5673 | 0.5658 | +0.1682 (+41.9%) | [+0.1554, +0.1810] | $t = 25.84, p = 1.10 \times 10^{-94}$ | $W = 4,120.0, p = 8.54 \times 10^{-90}$ | $p < 0.0002$ |

*Notes:*
1. **Zero Test-Label Leakage:** The SEAA policy action-values $Q(s, a)$ are frozen during testing; no test labels or corruption metadata are exposed.
2. **Cluster Significance:** The clustered standard error ($SE = 0.0057$) accounts for repeated measures across 8 conditions per movie. Naive unclustered estimation underestimates variance by $1.39\times$.
3. **5,000-Iteration Cluster Bootstrap:** Out of 5,000 cluster-resampled replications, zero resamples produced non-positive differences (empirical $p < 0.0002$).
