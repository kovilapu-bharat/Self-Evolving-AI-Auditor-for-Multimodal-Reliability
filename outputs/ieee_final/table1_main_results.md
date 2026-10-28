# Table 1: SEAA Master Performance Benchmark & Repeated-Measures Clustered Statistics

**Evaluation Protocol:** Evaluated across 500 held-out MM-IMDb test movies repeated over all 8 failure conditions ($N = 4,000$ total evaluations per policy).  
**Statistical Clustering:** Clustered at the movie entity level ($N_{\text{clusters}} = 500, df = 499$). Intraclass Correlation $\text{ICC}(1) = 0.1319$, Variance Inflation Factor $\text{VIF} = 1.9232$, Effective Sample Size $N_{\text{eff}} \approx 2,080$.

| Policy / Model Configuration | Condition-Mean Micro-F1 *(Primary)* | Condition-Mean Macro-F1 | Globally Pooled Micro-F1 *(Secondary)* | Globally Pooled Macro-F1 | Instance-Level Mean F1 *(Test Unit)* | Gain vs No Rec. (Micro-F1) | Gain vs Fixed (Micro-F1) | Clustered Paired $t$-test ($df=499$) | Wilcoxon Signed-Rank ($W$) | Clustered 95% Confidence Interval |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Proposed: Adaptive SEAA Policy** | **0.4709** | **0.3593** | **0.4690** | **0.3523** | **0.4671** | **+0.0697 (+17.4%)** | **+0.0768 (+19.5%)** | Reference | Reference | Reference |
| **Baseline 1: No Recovery (Static Late Fusion, $\alpha=0.55$)** | 0.4012 | 0.3165 | 0.3973 | 0.3030 | 0.3940 | Reference (0.0000) | +0.0071 (+1.8%) | $t = 12.79, p = 1.34 \times 10^{-32}$ | $W = 23,564.0, p = 1.15 \times 10^{-31}$ | $[+0.0619, +0.0843]$ |
| **Baseline 2: Fixed Policy (Heuristic Router)** | 0.3942 | 0.2945 | 0.3905 | 0.2898 | 0.3901 | -0.0070 (-1.7%) | Reference (0.0000) | $t = 12.82, p = 1.02 \times 10^{-32}$ | $W = 24,620.5, p = 9.43 \times 10^{-30}$ | $[+0.0652, +0.0888]$ |
| **Theoretical Ceiling: Oracle Upper Bound** | 0.5704 | 0.4421 | 0.5673 | 0.4399 | 0.5757 | +0.1692 (+42.2%) | +0.1762 (+44.7%) | — | — | — |

---

### Methodological & Inferential Notes:
1. **Primary Metric:** Condition-Averaged Micro-F1 is the pre-specified primary endpoint across the 8 benchmark regimes.
2. **Gain Precision:** Full-precision difference for Adaptive vs. Fixed Policy is $0.4709375 - 0.3941500 = +0.0767875$ (rounds to **+0.0768**). If calculated from rounded 4-decimal values ($0.4709 - 0.3942$), it yields **+0.0767**.
3. **Repeated-Measures Dependence:** Standard error in naive unclustered analysis is underestimated by $1.39\times$ ($\text{SE}_{\text{naive}} = 0.0041$ vs $\text{SE}_{\text{clustered}} = 0.0057$), demonstrating why entity clustering is methodologically essential.
4. **5,000-Iteration Clustered Bootstrap:** 0 out of 5,000 resamples with replacement yielded $\Delta \le 0$, empirically confirming an upper bound of $p < 0.0002$ across all metric combinations.
