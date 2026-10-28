# SEAA Master Artifact & Metric Audit Report for IEEE Paper Outputs

**Project:** A Self-Evolving AI Auditing Algorithm for Adaptive Reliability Diagnosis and Recovery in Multimodal AI Systems (SEAA)  
**Repository Root:** `C:\Users\bharat kumar\OneDrive\Desktop\SEAA`  
**Execution Phase:** Step A — Audit Only (Strict Verification, Zero Scientific Code/Model Modifications)  
**Date of Audit:** 2026-10-28  
**Auditor Engine:** Antigravity AI Engine  
**Status:** COMPLETE (All 6 Requested IEEE Deliverables Audited, Sources Traced, Discrepancies Reconciled)

---

## 1. Executive Summary & Audit Scope

This document provides a complete, independent scientific audit of all codebase components, saved model checkpoints, benchmark evaluation CSVs/JSONs, statistical tests, and visual artifacts for the **SEAA** framework. 

Per the Master Directive (`AGENTS.md`) and the Task Specification (`SEAA_Antigravity_IEEE_outputs_implementation_prompt.md`), this step (Step A) adheres strictly to:
1. **Zero Test-Label Leakage:** Verified that no test labels were used to guide action selection, update Q-tables, or tune thresholds.
2. **Cluster-Level Statistical Unit:** Verified that repeated evaluations across 8 conditions are clustered at the movie entity level ($N=500$ clusters, $df=499$).
3. **No Unapproved Experiments:** Zero models retrained, zero synthetic data fabricated, zero benchmark runs initiated.
4. **Honest Discrepancy Reporting:** All numerical differences between benchmark runs and earlier reports are isolated, mathematically explained, and documented.

---

## 2. Inventory of Primary Source Artifacts

The table below lists the verified source of truth for every metric and configuration parameter:

| Output Target | Primary Source File | File Size / Modification Date | Status |
|---|---|---|:---:|
| **Fig 1: Architecture** | `src/models/text_baseline.py`<br>`src/models/image_baseline.py`<br>`src/models/multimodal_baseline.py`<br>`src/policy/adaptive_policy.py`<br>`src/auditing/diagnosis.py`<br>`outputs/checkpoints/` | Codebase & model checkpoints | **VERIFIED** |
| **Fig 2: Baseline Models** | `outputs/metrics/baseline_comparison.csv`<br>`outputs/checkpoints/multimodal_baseline.json` | 4 lines CSV<br>Checkpoints (.joblib, .json) | **VERIFIED** |
| **Fig 3: Recovery by Cond.** | `outputs/recovery/run_20261025_phase5_adaptive_recovery/policy_comparison_metrics.csv` | 9 lines CSV (500 samples $\times$ 8 conditions) | **VERIFIED** |
| **Fig 4: Action Heatmap** | `outputs/recovery/run_20261025_phase5_adaptive_recovery/sample_level_decisions.csv`<br>`outputs/recovery/run_20261025_phase5_adaptive_recovery/policy_action_distributions.json` | 4,000 rows CSV<br>Action counts JSON | **VERIFIED** |
| **Fig 5: Ablation Study** | `outputs/recovery/ablation_study_results.csv`<br>`outputs/recovery/ablation_study_results.json`<br>`scripts/run_ablation_benchmarks.py` | 8 rows CSV<br>Detailed benchmark JSON | **AUDITED & MECHANISM EXPLAINED**<br>*(Pending Protocol Decision)* |
| **Table 1: Main Results** | `outputs/recovery/run_20261025_phase5_adaptive_recovery/sample_level_decisions.csv`<br>`outputs/recovery/phase5_final_statistical_check.md` | 4,000 instance decisions<br>5,000 bootstrap resamples | **VERIFIED & REPRODUCED** |

---

## 3. Systematic Verification of the Six Requested IEEE Outputs

### 3.1 Figure 1 — SEAA Architecture Diagram Specifications

All technical architecture parameters were inspected directly from the saved model checkpoints in `outputs/checkpoints/` and the source implementations:

1. **Text Modality Pipeline:**
   - **Input:** Movie plot synopses / summary text $x^T$.
   - **Feature Extraction:** Sublinear TF-IDF (`ngram_range=(1, 2)`, `sublinear_tf=True`, `min_df=2`).
   - **Feature Dimensionality:** Exactly **10,000 features** (`max_features=10000`, confirmed from `text_baseline.joblib`). *(Resolves earlier ambiguity mentioning 3,000 features; the actual model uses 10,000).*
   - **Predictive Model:** `OneVsRestClassifier(LogisticRegression(C=1.0, max_iter=1000))` across 23 canonical genres.

2. **Vision Modality Pipeline:**
   - **Input:** Official movie poster artwork RGB $224 \times 224$.
   - **Feature Extraction:** Pretrained frozen **ResNet-50** (`weights=ResNet50_Weights.DEFAULT`, final classification layer stripped).
   - **Feature Dimensionality:** Exactly **2,048 dimensions** (`data/processed/resnet50_features.npy`, shape `(25959, 2048)`, 212.6 MB float32). *(Resolves earlier ambiguity mentioning 502 features; 502 was a subset sample count, the embedding dimension is 2,048).*
   - **Preprocessing:** `StandardScaler()`.
   - **Predictive Model:** `OneVsRestClassifier(LogisticRegression(C=0.5, max_iter=1000))` across 23 canonical genres.

3. **Multimodal Late Fusion Baseline:**
   - Static weighted late fusion: $\hat{p}^{MM} = \alpha_0 \hat{p}^T + (1 - \alpha_0) \hat{p}^V$ with $\alpha_0 = 0.55$ (confirmed in `multimodal_baseline.json`).
   - Genre decision thresholds $\tau_c \in [0.10, 0.48]$ tuned per class on the validation split to maximize Micro-F1.

4. **Inline Reliability Monitoring Signals (No Ground Truth Required):**
   - **Presence Indicators:** $P_T = \mathbb{I}(x^T \neq \emptyset)$, $P_V = \mathbb{I}(x^V \neq \emptyset)$.
   - **Predictive Binary Entropy (Uncertainty):**
     $$H(\hat{p}) = -\frac{1}{C} \sum_{c=1}^C \left[ \hat{p}_c \log_2(\hat{p}_c) + (1 - \hat{p}_c) \log_2(1 - \hat{p}_c) \right]$$
     with probabilities clipped to $[10^{-7}, 1 - 10^{-7}]$.
   - **Bernoulli Jensen–Shannon Divergence (Disagreement):**
     $$M_c = \frac{1}{2}(\hat{p}_c^T + \hat{p}_c^V), \quad D_{JS}(\hat{p}^T \parallel \hat{p}^V) = \frac{1}{2C} \sum_{c=1}^C \left[ D_{KL}(\hat{p}_c^T \parallel M_c) + D_{KL}(\hat{p}_c^V \parallel M_c) \right] \in [0, 1]$$
     Disagreement cutoff calibrated on validation split: $\tau_{JSD} = 0.1139$.
   - **Representation L2 Norms:** $\|z^T\|_2$ and $\|z^V\|_2$.

5. **Failure Diagnosis States ($G_t$):**
   - 7 canonical categorical states: `CLEAN`, `TEXT_FAILURE`, `IMAGE_FAILURE`, `MISSING_TEXT`, `MISSING_IMAGE`, `CROSS_MODAL_CONFLICT`, `HIGH_UNCERTAINTY`.

6. **Contextual Bandit State Space & Admissible Actions:**
   - Context State 4-tuple: $s_t = (P_t, D_t, U_t, G_t)$.
   - 5 Admissible Recovery Actions:
     1. $a_{\text{STATIC\_FUSION}}$: Standard late fusion ($\alpha = 0.55$).
     2. $a_{\text{TEXT\_ONLY}}$: Pure text modality isolation ($\alpha = 1.0$).
     3. $a_{\text{IMAGE\_ONLY}}$: Pure vision modality isolation ($\alpha = 0.0$).
     4. $a_{\text{RELIABILITY\_WEIGHTED}}$: Dynamic continuous gating:
        $$\alpha(x) = \operatorname{clip}\left( 0.55 + 0.40 \tanh\left( 3.0 \cdot [H(\hat{p}^V) - H(\hat{p}^T)] \right), 0.05, 0.95 \right)$$
     5. $a_{\text{CONSERVATIVE\_FUSION}}$: Neutral 50/50 balance ($\alpha = 0.50$).
   - **Physical Admissibility Masking:** If $P_V = 0$, valid actions are strictly $\{a_{\text{TEXT\_ONLY}}\}$. If $P_T = 0$, valid actions are strictly $\{a_{\text{IMAGE\_ONLY}}\}$.

7. **Validation Learning vs. Test-Time Policy Freeze:**
   - **Offline Outcome Training (Validation Split Only):**
     $$R_t = F_1(\hat{y}_{a_t}, y_t) + \lambda \left[ F_1(\hat{y}_{a_t}, y_t) - F_1(\hat{y}_{unrec}, y_t) \right], \quad \lambda = 0.5, \quad R_t \in [-0.5, 1.5]$$
     $$Q_{t+1}(s_t, a_t) = Q_t(s_t, a_t) + \eta_t [R_t - Q_t(s_t, a_t)], \quad \eta_t = \max\left( 0.05, \frac{\eta_0}{(N_{t+1}(s_t, a_t))^{0.5}} \right)$$
   - **Test-Time Evaluation:**
     - Ground-truth test labels are **never** accessible to the auditor or policy during inference.
     - Policy is strictly frozen ($\eta_t = 0$, `learning_enabled = False`, `explore = False`).
     - Action selection is pure greedy exploitation: $a^* = \arg\max_{a \in \mathcal{A}_{valid}} Q(s_t, a)$ with deterministic tie-breaking.

---

### 3.2 Figure 2 — Baseline Model Comparison Verification

**Source File:** `outputs/metrics/baseline_comparison.csv`  
**Test Set Evaluation:** Canonical held-out test split ($N=7,788$ movie samples, 30% of MM-IMDb).

| Model Configuration | Input Modality | Micro-F1 | Macro-F1 | Micro-Precision | Micro-Recall | Mean Avg Precision (mAP) | Hamming Loss | Inference Time (s) | Throughput (samples/s) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Text-Only Baseline** (TF-IDF 10k + LR) | Text Synopsis | **0.6056** | **0.4955** | 0.5594 | 0.6602 | 0.5183 | 0.0929 | 1.3948 | 5,583.5 |
| **Image-Only Baseline** (ResNet-50 + LR) | Poster Embeddings | **0.4151** | **0.2847** | 0.3373 | 0.5396 | 0.2337 | 0.1642 | 1.8027 | 4,320.1 |
| **Multimodal Late Fusion** ($\alpha=0.55$) | Text + Image | **0.5161** | **0.4003** | 0.4301 | 0.6451 | 0.3985 | 0.1307 | 3.4218 | 2,276.0 |

**Verification Status:** **100% VERIFIED.**  
All three models were evaluated under identical multi-label thresholds tuned on the validation split over the complete 7,788 test split.

---

### 3.3 Figure 3 — Recovery Performance Across Eight Failure Regimes

**Source File:** `outputs/recovery/run_20261025_phase5_adaptive_recovery/policy_comparison_metrics.csv`  
**Evaluation Cohort:** 500 unique test movies repeated across all 8 failure regimes ($4,000$ evaluations per policy).

| Condition ID | Regime Name | Corrupted Modality | No Recovery (P1) | Fixed Policy (P2) | Adaptive SEAA (P3) | Theoretical Oracle (P4) | SEAA Gain vs P1 ($\Delta$) | SEAA Gain vs P2 ($\Delta$) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | Clean_Baseline | None | 0.5187 | 0.4652 | **0.5528** | 0.6586 | +0.0341 | +0.0876 |
| 1 | Text_Truncation_25% | Text | 0.4635 | 0.4432 | **0.4700** | 0.6026 | +0.0065 | +0.0268 |
| 2 | Text_Token_Masking_35% | Text | 0.4619 | 0.4364 | **0.4690** | 0.6047 | +0.0071 | +0.0326 |
| 3 | Text_Empty | Text | 0.3836 | **0.3996** | **0.3996** | 0.3996 | +0.0160 | 0.0000 |
| 4 | Image_Gaussian_Blur_r5 | Vision | 0.3466 | 0.2272 | **0.4016** | 0.5882 | +0.0550 | +0.1744 |
| 5 | Image_Gaussian_Noise_s25 | Vision | 0.4121 | 0.3183 | **0.4988** | 0.6147 | +0.0867 | +0.1805 |
| 6 | Image_Missing_Blackout | Vision | 0.3060 | **0.5820** | **0.5820** | 0.5820 | +0.2760 | 0.0000 |
| 7 | Dual_Corruption_Trunc_Noise | Both | 0.3173 | 0.2813 | **0.3937** | 0.5126 | +0.0764 | +0.1124 |
| — | **Condition-Average** | — | **0.4012** | **0.3942** | **0.4709** | **0.5704** | **+0.0697** | **+0.0768** |

**Key Observations for IEEE Presentation:**
1. **Physical Masking Ties:** In Regime 3 (`Text_Empty`) and Regime 6 (`Image_Missing_Blackout`), SEAA and the Fixed Policy achieve identical performance ($0.3996$ and $0.5820$, respectively). This occurs because physical admissibility masks leave only one valid action (`IMAGE_ONLY` and `TEXT_ONLY`), demonstrating that both policies correctly respect physical hardware constraints.
2. **Fixed Policy Collapse Under Optical Blur:** In Regime 4 (`Image_Gaussian_Blur_r5`), the fixed heuristic router collapses to $0.2272$ (worse than no recovery: $0.3466$). SEAA achieves $0.4016$ by dynamically balancing modalities.
3. **Condition Average Presentation:** The condition average ($0.4709$ vs $0.4012$ vs $0.3942$) is displayed as a distinct summary aggregate, not as a 9th regime.

---

### 3.4 Figure 4 — Adaptive Recovery Action Heatmap Verification

**Source Files:**
- `outputs/recovery/run_20261025_phase5_adaptive_recovery/sample_level_decisions.csv`
- `outputs/recovery/run_20261025_phase5_adaptive_recovery/policy_action_distributions.json`

**Percentage Action Distributions (Derived Directly from 4,000 Instance Decisions):**

| Failure Regime (Row) | Static Fusion ($a_1$) | Text Only ($a_2$) | Image Only ($a_3$) | Rel. Weighted ($a_4$) | Conserv. Fusion ($a_5$) | Row Total |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Clean_Baseline** | 12.2% | 72.8% | 0.0% | 15.0% | 0.0% | 100.0% |
| **Text_Truncation_25%** | 10.8% | 73.6% | 0.0% | 15.6% | 0.0% | 100.0% |
| **Text_Token_Masking_35%** | 12.2% | 77.0% | 0.0% | 10.8% | 0.0% | 100.0% |
| **Text_Empty** | 0.0% | 0.0% | **100.0%** | 0.0% | 0.0% | 100.0% |
| **Image_Gaussian_Blur_r5** | 59.2% | 39.0% | 0.0% | 1.8% | 0.0% | 100.0% |
| **Image_Gaussian_Noise_s25** | 25.6% | 67.0% | 0.0% | 7.4% | 0.0% | 100.0% |
| **Image_Missing_Blackout** | 0.0% | **100.0%** | 0.0% | 0.0% | 0.0% | 100.0% |
| **Dual_Corruption_Trunc_Noise** | 22.6% | 69.0% | 0.0% | 8.4% | 0.0% | 100.0% |

**Admissibility Constraint Verification:**
- For `Text_Empty`, $a_{\text{IMAGE\_ONLY}}$ was selected in 100.0% of samples ($500/500$). Zero text actions selected.
- For `Image_Missing_Blackout`, $a_{\text{TEXT\_ONLY}}$ was selected in 100.0% of samples ($500/500$). Zero image actions selected.
- Feasibility masking constraint is **100% strictly enforced**.

---

### 3.5 Figure 5 — Ablation Study Verification & Deep Diagnostic Investigation

**Source Files:**
- `outputs/recovery/ablation_study_results.csv`
- `outputs/recovery/ablation_study_results.json`
- Script: `scripts/run_ablation_benchmarks.py`

#### 3.5.1 Summary of Ablation Results in Existing Artifacts

| Configuration / Policy Variant | Condition-Mean Micro-F1 | Condition-Mean Macro-F1 | Globally Pooled Micro-F1 | Globally Pooled Macro-F1 | Instance-Level Mean F1 | $\Delta$ vs No Recovery |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Full SEAA Policy** | **0.4711** | **0.3611** | **0.4690** | **0.3541** | **0.4675** | **+0.0698** |
| **Ablation: Learning Disabled** (Untrained Q-Table) | 0.4377 | 0.3361 | 0.4348 | 0.3284 | 0.4305 | +0.0365 |
| **Ablation: No Diagnosis Feature** ($G_t \to \text{"UNSPECIFIED"}$) | 0.5054 | 0.3714 | 0.5056 | 0.3743 | 0.4937 | +0.1042 |
| **Ablation: Random Action Selection** | 0.4346 | 0.3306 | 0.4327 | 0.3239 | 0.4295 | +0.0334 |
| **Baseline: No Recovery** (Static Late Fusion) | 0.4012 | 0.3165 | 0.3973 | 0.3030 | 0.3940 | 0.0000 |
| **Baseline: Fixed Policy** (Heuristic Router) | 0.3931 | 0.2927 | 0.3894 | 0.2888 | 0.3890 | -0.0081 |
| **Theoretical Ceiling: Oracle Upper Bound** | 0.5695 | 0.4421 | 0.5673 | 0.4399 | 0.5757 | +0.1682 |

#### 3.5.2 Investigation of the "Diagnosis Removed" Paradox ($0.5054 > 0.4711$)

A critical question raised in the task specification: *Why does removing the diagnosis feature ($G_t$) yield higher performance ($0.5054$) than Full SEAA ($0.4711$)?*

Our deep-dive inspection into `sample_level_decisions.csv` and `diagnosis_evaluation_summary.json` uncovered the exact causal mechanism:
1. **Diagnostic False Positives on Visual Blur:**
   In Regime 4 (`Image_Gaussian_Blur_r5`), the rule-based threshold auditor diagnosed:
   - `TEXT_FAILURE`: **242 / 500 samples (48.4%)**
   - `CROSS_MODAL_CONFLICT`: **113 / 500 samples (22.6%)**
   - `CLEAN`: **114 / 500 samples (22.8%)**
   - `IMAGE_FAILURE`: **Only 3 / 500 samples (0.6%)**
   The optical blur corrupted the image, creating high cross-modal disagreement. However, because the text feature representation was pristine, the auditor's uncertainty/norm heuristic inverted the blame and falsely declared `TEXT_FAILURE`.

2. **Impact on the Fixed Heuristic Policy:**
   Because the fixed policy routes blindly according to the diagnosis label, when it observed `TEXT_FAILURE`, it routed 242 samples directly to **`IMAGE_ONLY`**! Discarding the pristine text synopsis and relying solely on a heavily blurred image caused its Micro-F1 to collapse to **0.2272**.

3. **Impact on Full SEAA vs. No-Diagnosis Policy:**
   - In Full SEAA, the Q-table conditions on $s_t = (P_t, D_t, U_t, G_t)$. When $G_t = \text{TEXT\_FAILURE}$, its learned action value for pure `TEXT_ONLY` is depressed, causing it to fall back to `STATIC_FUSION` (296 times) or `TEXT_ONLY` (195 times), achieving Micro-F1 of **0.3987** on blur.
   - When the diagnosis feature is stripped (`strip_diagnosis = True`), the state simplifies to $s_t = (P_t, D_t, U_t)$. Because the text baseline is fundamentally stronger than the image baseline on MM-IMDb ($0.6056$ vs $0.4151$), the policy learned that whenever both modalities are present and disagreement is high, choosing **`TEXT_ONLY`** provides the highest reward!
   - Consequently, `Ablation_No_Diagnosis` chose `TEXT_ONLY` for the blurred images, jumping to **0.5757 Micro-F1** on blur (approaching the Oracle's $0.5873$).
   - Across the 8 conditions, this blur advantage elevated its condition average from $0.4711$ to $0.5054$.

**Scientific Integrity Conclusion:**
This finding does **not** indicate that outcome-based adaptation is flawed; rather, it proves that:
1. **Outcome learning works:** Disabling learning drops performance from $0.4711$ to $0.4377$ ($-3.34\%$ points), and random actions achieve only $0.4346$.
2. **Heuristic diagnosis is a noisy bottleneck:** The rule-based diagnostic classifier is vulnerable to misattributing failure under spatial blur. When conditioned on this noisy signal, policy performance is constrained compared to relying directly on continuous uncertainty asymmetry ($U_t$) and presence ($P_t$).
3. **Publication Recommendation:** The IEEE paper should present this finding transparently as an insightful empirical discovery on the limits of heuristic failure diagnosis in multimodal ML, rather than obscuring it.

---

### 3.6 Table 1 — Main Results and Statistical Comparison Verification

**Source Data:** Evaluated on all 500 movie entities $\times$ 8 failure conditions ($N=4,000$ decisions) from `sample_level_decisions.csv`.  
**Cluster Unit:** `movie_id` ($N_{\text{clusters}} = 500$, $df = 499$).

#### 3.6.1 Main Performance Benchmark Table

| Metric Level | Evaluation Metric | Baseline 1: No Recovery | Baseline 2: Fixed Policy | Proposed: Adaptive SEAA | Gain vs No Recovery ($\Delta$) | Gain vs Fixed Policy ($\Delta$) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Condition-Averaged** *(Primary)* | **Micro-F1** | 0.4012 | 0.3942 | **0.4709** | **+0.0697 (+17.4%)** | **+0.0768 (+19.5%)** |
| Condition-Averaged | Macro-F1 | 0.3165 | 0.2945 | **0.3593** | +0.0428 (+13.5%) | +0.0648 (+22.0%) |
| **Globally Pooled** *(Secondary)* | **Micro-F1** | 0.3973 | 0.3905 | **0.4690** | **+0.0716 (+18.0%)** | **+0.0784 (+20.1%)** |
| Globally Pooled | Macro-F1 | 0.3030 | 0.2898 | **0.3523** | +0.0493 (+16.3%) | +0.0624 (+21.5%) |
| **Instance-Level** *(Unit for Tests)* | **Mean F1** | 0.3940 | 0.3901 | **0.4671** | **+0.0731 (+18.6%)** | **+0.0770 (+19.7%)** |

*Note on Full Precision vs. Rounded Differences:*
- Adaptive vs Fixed Policy Condition-Averaged Micro-F1: $0.4709375 - 0.3941500 = \mathbf{+0.0767875}$ (rounds to **+0.0768**). If calculated from rounded values ($0.4709 - 0.3942$), it yields **+0.0767**. Both representations are mathematically verified.

#### 3.6.2 Clustered Repeated-Measures Hypothesis Tests ($df = 499$)

| Comparison Contrast | Metric Evaluated | Mean Cluster Delta ($\bar{D}$) | Cluster Std Error (SE) | 95% Confidence Interval | Paired $t$-test Statistic | Exact $p$-value ($t$-test) | Wilcoxon Signed-Rank ($W$) | Exact $p$-value (Wilcoxon) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **SEAA vs No Recovery (P3 vs P1)** | Instance Mean F1 | **+0.0731** | 0.0057 | [+0.0619, +0.0843] | $t = 12.7904$ | $1.34 \times 10^{-32}$ | $W = 23,564.0$ | $1.15 \times 10^{-31}$ |
| **SEAA vs Fixed Policy (P3 vs P2)** | Instance Mean F1 | **+0.0770** | 0.0060 | [+0.0652, +0.0888] | $t = 12.8192$ | $1.02 \times 10^{-32}$ | $W = 24,620.5$ | $9.43 \times 10^{-30}$ |

#### 3.6.3 Clustering & Design Effect Metrics
- **Intraclass Correlation:** $\text{ICC}(1) = 0.1319$.
- **Variance Inflation Factor (Design Effect):** $\text{VIF} = 1 + (8 - 1) \times 0.1319 = 1.9232$.
- **Effective Sample Size:** $N_{\text{eff}} = 4000 / 1.9232 \approx 2,080$ independent observations.
- **SE Inflation:** Naive evaluation underestimates SE by $1.39\times$ ($0.0041$ vs $0.0057$), demonstrating that entity clustering is essential.

#### 3.6.4 5,000-Iteration Cluster Bootstrap ($B = 5,000$, Seed = 42)
- Across 5,000 cluster resamples with replacement, exactly **0 out of 5,000 resamples** yielded $\Delta \le 0$ for all evaluated metrics.
- Empirical upper bound on error rate: $p < \frac{1}{5000} = \mathbf{0.0002}$.
- Bootstrap 95% Percentile CI:
  - Instance Mean F1 (P3 vs P1): $[+0.0619, +0.0843]$
  - Instance Mean F1 (P3 vs P2): $[+0.0652, +0.0888]$
  - Globally Pooled Micro-F1 (P3 vs P1): $[+0.0627, +0.0810]$
  - Globally Pooled Micro-F1 (P3 vs P2): $[+0.0694, +0.0877]$

---

## 4. Reconciliation of Known Discrepancies

The following discrepancies identified in earlier documentation have been traced and fully resolved:

1. **TF-IDF Feature Count (3,000 vs 10,000):**
   - **Resolution:** Checkpoint `outputs/checkpoints/text_baseline.joblib` contains exactly `max_features = 10000`, `vocabulary_` length = 10,000. The active production model uses **10,000 features**.

2. **ResNet-50 Feature Dimensions (502 vs 2,048):**
   - **Resolution:** The cached feature file `data/processed/resnet50_features.npy` has shape `(25959, 2048)`. The image baseline logistic regression estimators have input coefficients of shape `(1, 2048)`. The dimension is **2,048**. (502 was a test sample count from earlier preliminary evaluations).

3. **Macro-F1 Full SEAA (0.3593 vs 0.3611):**
   - **Resolution:** $0.3593$ is the condition-averaged Macro-F1 from the Phase 5 master benchmark run (`policy_comparison_metrics.csv`). $0.3611$ is the condition-averaged Macro-F1 from the separate ablation benchmark run (`ablation_study_results.csv`). Both are valid within their respective runs; Table 1 uses $0.3593$.

4. **Fixed Policy Micro-F1 (0.3942 vs 0.3931) and Macro-F1 (0.2945 vs 0.2927):**
   - **Resolution:** $0.3942 / 0.2945$ is from `policy_comparison_metrics.csv` (Phase 5 master benchmark). $0.3931 / 0.2927$ is from `ablation_study_results.csv`. Table 1 uses the Phase 5 benchmark values ($0.3942 / 0.2945$).

5. **Adaptive Gain over Fixed Policy (+0.0768 vs +0.0767):**
   - **Resolution:** Recomputed full-precision difference is $0.4709375 - 0.3941500 = +0.0767875$, which rounds to **+0.0768**. Subtracting 4-decimal rounded values ($0.4709 - 0.3942$) gives **+0.0767**. Both are reported and explained.

6. **Bootstrap Empirical Proportion as p-value:**
   - **Resolution:** Clarified that observing 0 non-positive resamples out of 5,000 corresponds to an empirical proportion bound of $p < 0.0002$, while formal inferential hypothesis testing is supported by the clustered paired $t$-test ($p = 1.34 \times 10^{-32}$) and Wilcoxon signed-rank test ($p = 1.15 \times 10^{-31}$).

---

## 5. Output Deliverables Readiness Status

| Deliverable Name | File Targets | Status | Recommendation |
|---|---|:---:|---|
| **Figure 1** | `fig1_seaa_architecture.pdf`<br>`fig1_seaa_architecture.png` | **READY** | Generate using verified parameters (TF-IDF 10k, ResNet 2048d, $\alpha=0.55$, frozen test policy). |
| **Figure 2** | `fig2_baseline_comparison.pdf`<br>`fig2_baseline_comparison.png`<br>`fig2_baseline_source.csv` | **READY** | Generate from `outputs/metrics/baseline_comparison.csv` ($N=7,788$). |
| **Figure 3** | `fig3_recovery_by_condition.pdf`<br>`fig3_recovery_by_condition.png`<br>`fig3_recovery_source.csv` | **READY** | Generate from `policy_comparison_metrics.csv` ($N=4,000$). |
| **Figure 4** | `fig4_action_heatmap.pdf`<br>`fig4_action_heatmap.png`<br>`fig4_action_distribution_source.csv` | **READY** | Generate from `sample_level_decisions.csv` (100% verified action distributions). |
| **Figure 5** | `fig5_ablation_study.pdf`<br>`fig5_ablation_study.png`<br>`fig5_ablation_source.csv` | **PROVISIONAL / VERIFIED SOURCE** | Existing data in `ablation_study_results.csv` is fully verified. We recommend plotting existing verified values accompanied by honest text explaining the visual blur diagnostic bottleneck. |
| **Table 1** | `table1_main_results.csv`<br>`table1_main_results.md` | **READY** | Generate from `policy_comparison_metrics.csv` and cluster statistical analysis. |

---

## 6. Proposed Next Steps (Awaiting User Approval)

In accordance with the approval gate defined in `SEAA_Antigravity_IEEE_outputs_implementation_prompt.md`:

1. **Step A (This Report):** Complete. Findings presented for user review. Zero scientific code, checkpoints, or data changed.
2. **Proposed Step B (Output Generation):**
   - Create generation scripts in `scripts/` (or dedicated plotting script).
   - Generate all 5 figures (PDF vector + 300 DPI PNG preview) and Table 1 (CSV + Markdown) in `outputs/ieee_final/`.
   - Generate all companion machine-readable source CSVs (`fig2_baseline_source.csv`, `fig3_recovery_source.csv`, `fig4_action_distribution_source.csv`, `fig5_ablation_source.csv`, `table1_main_results.csv`).
   - Create `reproduction_log.txt` recording full environment, timestamps, and MD5 hashes.
3. **Ablation Protocol Decision (Step C):**
   - **Option 1 (Recommended):** Plot the verified existing ablation data ($0.4711$ vs $0.4377$ vs $0.5054$) and present the diagnosis bottleneck honestly in the paper text.
   - **Option 2:** If the user desires a clean diagnosis ablation where diagnosis strictly helps across all conditions, propose a protocol using `LearnedDiagnosticClassifier` and wait for approval before running.

