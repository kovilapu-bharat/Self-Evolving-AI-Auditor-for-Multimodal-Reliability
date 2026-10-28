# SEAA Phase 5 Final Report: Adaptive Multimodal Recovery & Self-Evolution Audit

**Project:** A Self-Evolving AI Auditing Algorithm for Adaptive Reliability Diagnosis and Recovery in Multimodal AI Systems (SEAA)  
**Evaluator:** Antigravity AI Engine (Documentation-Corrected Edition)  
**Original Report Date:** 2026-10-25  
**Documentation Revision Date:** 2026-10-26 (Bootstrap Bias Correction Edition)  
**Target Specification:** [`phase5_final_bootstrap_bias_correction_prompt.md`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/phase5_final_bootstrap_bias_correction_prompt.md)  
**Deliverable Path:** [`docs/phase5/phase5_final_report.md`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/docs/phase5/phase5_final_report.md)  
**Backup Paths:** [`docs/phase5/phase5_final_report.backup_20261025_173000.md`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/docs/phase5/phase5_final_report.backup_20261025_173000.md), [`docs/phase5/phase5_final_report.backup_20261026_statistical_check.md`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/docs/phase5/phase5_final_report.backup_20261026_statistical_check.md), [`docs/phase5/phase5_final_report.backup_20261026_bias_correction.md`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/docs/phase5/phase5_final_report.backup_20261026_bias_correction.md)  
**Correction Log:** [`outputs/phase5/phase5_documentation_correction_log.md`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/outputs/phase5/phase5_documentation_correction_log.md)  
**Statistical Verification Check:** [`outputs/phase5/phase5_final_statistical_check.md`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/outputs/phase5/phase5_final_statistical_check.md)  
**Run Artifact Directory:** [`outputs/phase5/run_20261025_phase5_adaptive_recovery/`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/outputs/phase5/run_20261025_phase5_adaptive_recovery/)  
**Audit Cross-References:** [`outputs/phase5/phase5_independent_audit.md`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/outputs/phase5/phase5_independent_audit.md), [`outputs/phase5/phase5_audit_followup.md`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/outputs/phase5/phase5_audit_followup.md)

---

## 1. Executive Summary

Phase 5 of the SEAA project designs, implements, validates, and empirically benchmarks an **Adaptive, Outcome-Learning Recovery Policy** for multimodal reliability failures.

Prior to Phase 5, the repository possessed only a fixed, heuristic rule-based router ([`src/recovery/modality_gating.py`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/src/recovery/modality_gating.py): `DynamicModalityGating`), which mapped static diagnostic string labels to hardcoded modality weights. While capable of mitigating extreme blackout failures, this fixed router lacked any self-evolution capability, recorded no recovery outcomes, performed no parameter updates, and caused severe performance degradation on clean and noisy inputs (dropping by up to **-11.94 points** under blur and **-5.35 points** under clean inputs due to false-positive modality dropouts).

In Phase 5, we implemented an **Adaptive Contextual Bandit Recovery Policy** ([`src/policy/adaptive_policy.py`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/src/policy/adaptive_policy.py): `AdaptiveRecoveryPolicy`) operating over 5 canonical recovery actions. The policy records inference contexts, actions, and recovery rewards; updates action values $Q(s, a)$ via an exponential moving average update rule with decaying learning rate; and demonstrates reproducible self-evolution where learned outcomes transition subsequent action choices from sub-optimal to optimal actions.

### Key Empirical Findings:

1. **Overall Multimodal Reliability Recovery (Dual-Aggregation Reporting):**
   - **Primary Metric — Condition-Averaged Micro-F1:**
     - **No Recovery Baseline ($P_{\text{unrecovered}}$):** **0.4012**
     - **Fixed Heuristic Policy ($P_{\text{fixed}}$):** **0.3942** ($\Delta = -0.0070$ vs unrecovered; fixed heuristic degrades performance overall)
     - **Adaptive Recovery Policy ($P_{\text{adaptive}}$):** **0.4709**, achieving an absolute gain of **+0.0697 (+17.4% relative)** over unrecovered fusion and **+0.0768 (+19.5% relative)** over the fixed policy
     - **Oracle Upper Bound ($P_{\text{oracle}}$):** **0.5704** ($\Delta = +0.1692$ vs unrecovered)
     - *Rationale for Primary Status:* The benchmark evaluates eight distinct, equally weighted failure regimes (500 samples each). Condition averaging assigns equal 12.5% weight to each failure regime, preventing severe blackout and noise conditions from dominating the aggregate.
   - **Complementary Metric — Globally Pooled Micro-F1:**
     - **No Recovery Baseline ($P_{\text{unrecovered}}$):** **0.3973**
     - **Fixed Heuristic Policy ($P_{\text{fixed}}$):** **0.3905** ($\Delta = -0.0068$ vs unrecovered)
     - **Adaptive Recovery Policy ($P_{\text{adaptive}}$):** **0.4690**, achieving an absolute gain of **+0.0717 (+18.0% relative)** over unrecovered fusion and **+0.0784 (+20.1% relative)** over the fixed policy.
   - **Macro-F1 (Corrected Figures):**
     - **Condition-Averaged Macro-F1:** No Recovery = **0.3165**, Fixed Policy = **0.2945**, Adaptive Policy = **0.3593** ($\Delta = +0.0428$ vs No Recovery; $\Delta = +0.0648$ vs Fixed Policy).
     - **Globally Pooled Macro-F1:** No Recovery = **0.3030**, Fixed Policy = **0.2898**, Adaptive Policy = **0.3523** ($\Delta = +0.0493$ vs No Recovery; $\Delta = +0.0624$ vs Fixed Policy).
     - *Correction Note:* The preliminary draft values previously cited in the report text (`0.3040`, `0.2972`, `0.3547`) were inconsistent with [`sample_level_decisions.csv`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/outputs/phase5/run_20261025_phase5_adaptive_recovery/sample_level_decisions.csv) due to a clerical drafting error. The true verified values are reported above and in Section 9.

2. **Clean-Input Performance Retention:**
   - On uncorrupted clean test inputs, the fixed heuristic router dropped performance from **0.5187 to 0.4652 Micro-F1 (-5.35 points)**.
   - In contrast, the Adaptive Policy preserved and enhanced clean performance to **0.5528 Micro-F1 (+3.41 points)** by learning when cooperative fusion vs. selective text weighting was optimal.

3. **Verification of Policy-Learning & Decision Transition:**
   - Exactly 2,500 training updates were recorded in [`outputs/phase5/run_20261025_phase5_adaptive_recovery/policy_training_history.json`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/outputs/phase5/run_20261025_phase5_adaptive_recovery/policy_training_history.json).
   - Non-destructive step-by-step replay confirmed that every update rigorously satisfies the decaying learning-rate update equation, matching the serialized policy [`trained_adaptive_policy.json`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/outputs/phase5/run_20261025_phase5_adaptive_recovery/trained_adaptive_policy.json) within floating-point precision ($1.26 \times 10^{-5}$).
   - Controlled unit tests ([`tests/test_adaptive_recovery.py:test_05`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/tests/test_adaptive_recovery.py)) empirically proved that an observed recovery outcome updates $Q(s, a)$ and alters subsequent action selection for identical contexts.
   - *Scientific Qualification:* Logged update arithmetic confirms the numerical execution of the moving-average Q-learning rule, but should be distinguished from claims of out-of-distribution generalization across unobserved corruption regimes.

4. **Data Leakage Controls and Split Scrutiny:**
   - The adaptive policy was trained strictly on validation split trajectories (`splits['val']`) and frozen (`learning_enabled=False`, $\epsilon=0$) during test evaluation.
   - Test ground-truth labels and corruption metadata were never accessible at action-selection time.
   - *Scope of Verification:* "Zero leakage" describes the verified procedural and code-level isolation between validation training and test evaluation; it does not imply that continuous online deployment is free from distribution drift.

5. **Statistical Reliability (Clustered Analysis & Cluster Bootstrap):**
   - The 4,000 evaluations comprise repeated measures over 500 unique movie entities across 8 conditions (Intraclass Correlation $\rho = 0.1319$, Variance Inflation Factor $= 1.9232$).
   - A naive test treating all 4,000 evaluations as independent underestimates standard error by $1.39\times$ and is not the primary test.
   - When clustered by unique movie entity ($N = 500, df = 499$), paired testing confirms statistically significant instance-level F1 improvements: vs No Recovery ($t = 12.79, p = 1.34 \times 10^{-32}$) and vs Fixed Policy ($t = 12.82, p = 1.02 \times 10^{-32}$).
   - A 5,000-iteration cluster bootstrap over the 500 movies confirms that all metric improvements (Instance F1, Pooled Micro-F1, Pooled Macro-F1, Condition-Mean Micro-F1, Condition-Mean Macro-F1) remain strictly positive with empirical $p < 0.0002$.

6. **Phase 1–3 Bitwise Preservation:**
   - All Phase 1–3 model weights, split partitions, and benchmark CSVs remain 100% bitwise identical against initial Git commit `6ff6058`.

---

## 2. Verified Starting Repository State

Before commencing implementation, the starting state of the workspace was audited (documented in [`docs/phase5/phase5_starting_state_report.md`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/docs/phase5/phase5_starting_state_report.md)):
- **Branch:** `main`
- **Baseline Commit:** `6ff6058` (`Initial commit: SEAA multimodal movie genre classification and auditing framework`)
- **Phase 4 Commit:** `45c1a13` (`feat(phase4): implement multimodal auditing, failure diagnostics, and recovery framework`)
- **Working Tree:** No functional code modifications. User archived earlier prompt documents.
- **Automated Test Suite:** Exactly 62/62 tests passing in 16.29s (`python -m unittest discover -s tests -p "test_*.py" -v`).

---

## 3. Files Created or Modified

### 3.1 New Functional Code
1. [`src/policy/adaptive_policy.py`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/src/policy/adaptive_policy.py): Complete implementation of `AdaptiveRecoveryPolicy`, `PolicyContext`, action constants (`ACTION_STATIC_FUSION`, `ACTION_TEXT_ONLY`, `ACTION_IMAGE_ONLY`, `ACTION_RELIABILITY_WEIGHTED`, `ACTION_CONSERVATIVE_FUSION`), UCB exploration, F1 reward calculation, and serialization.
2. [`src/policy/__init__.py`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/src/policy/__init__.py): Exposed policy classes and action constants.
3. [`src/recovery/__init__.py`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/src/recovery/__init__.py): Integrated `AdaptiveRecoveryPolicy` alongside existing recovery components.

### 3.2 Evaluation Runner Script
4. [`scripts/run_phase5_evaluation.py`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/scripts/run_phase5_evaluation.py): Standalone evaluation runner executing validation policy training, test-set out-of-sample evaluation across 8 conditions, latency profiling, CSV/JSON metric generation, and publication-quality figure rendering.

### 3.3 Test Suite
5. [`tests/test_adaptive_recovery.py`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/tests/test_adaptive_recovery.py): 11 comprehensive automated unit tests covering action admissibility, missing signal fallbacks, Q-table updates, controlled reward-driven decision switching (self-evolution proof), reproducibility, and test label isolation.

### 3.4 Documentation
6. [`docs/phase5/phase5_starting_state_report.md`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/docs/phase5/phase5_starting_state_report.md): Repository starting state and integrity gate audit.
7. [`docs/phase5/phase5_research_protocol.md`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/docs/phase5/phase5_research_protocol.md): Formal research question, action space, state context, reward functions, and protocol specification.
8. [`docs/phase5/phase5_final_report.md`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/docs/phase5/phase5_final_report.md): This comprehensive final report (documentation-corrected edition).

### 3.5 Generated Phase 5 Run Artifacts (`outputs/phase5/run_20261025_phase5_adaptive_recovery/`)
- `policy_comparison_metrics.csv`: Per-condition metric table across all 4 policies.
- `sample_level_decisions.csv`: 4,000 individual sample decisions across 8 conditions.
- `phase5_benchmark_summary.json`: High-level aggregate benchmark summary.
- `policy_action_distributions.json`: Percentage selection frequencies per action across conditions.
- `trained_adaptive_policy.json`: Serialized Q-table and visit counts after validation training.
- `policy_training_history.json`: 2,500 validation update records.
- `evaluated_sample_ids.txt`: 500 reproducible test movie IDs.
- `phase5_evaluation.log`: Detailed runtime log.
- Figures:
  - `figures/policy_micro_f1_comparison.png`
  - `figures/action_distribution_heatmap.png`
  - `figures/per_condition_recovery_deltas.png`

---

## 4. Existing Code Reused, Corrected, or Left Unchanged

1. **`src/recovery/modality_gating.py` (`DynamicModalityGating`):**
   - **Role:** Preserved and reused directly as the **Fixed Policy Baseline (`FIXED_POLICY`)**.
   - **Correction/Observation:** Audited and documented that it is a heuristic mapping lacking learning capabilities. Kept intact to provide a true experimental baseline.
2. **`src/recovery/confidence_recovery.py` (`ConfidenceRecoveryRouter`):**
   - **Role:** Left unchanged. Provides selective abstention and human-review triage routines.
3. **`src/auditing/learned_diagnostics.py` (`LearnedDiagnosticClassifier`):**
   - **Role:** Loaded from existing checkpoint (`outputs/checkpoints/learned_diagnostic_classifier.pkl`) and reused to provide predicted failure classifications to the auditor.
4. **`src/failures/text_corruptions.py` & `src/failures/image_corruptions.py`:**
   - **Role:** Reused to generate authentic input corruptions on raw text strings and PIL images before model forward passes.

---

## 5. Policy Design & Update Equations

### 5.1 Action Space ($\mathcal{A}$)
$$\mathcal{A} = \{\text{STATIC\_FUSION}, \text{TEXT\_ONLY}, \text{IMAGE\_ONLY}, \text{RELIABILITY\_WEIGHTED}, \text{CONSERVATIVE\_FUSION}\}$$
- $\text{STATIC\_FUSION}$: $\alpha = 0.55$.
- $\text{TEXT\_ONLY}$: $\alpha = 1.0$ (modality dropout of image). Feasible only when text is present.
- $\text{IMAGE\_ONLY}$: $\alpha = 0.0$ (modality dropout of text). Feasible only when image is present.
- $\text{RELIABILITY\_WEIGHTED}$: $\alpha(x) = \text{clip}\left(0.55 + 0.40 \tanh(3.0(u_{\text{img}} - u_{\text{txt}})), 0.05, 0.95\right)$.
- $\text{CONSERVATIVE\_FUSION}$: $\alpha = 0.50$.

### 5.2 Context Representation ($\mathcal{S}$)
Observable context is mapped to an interpretable state tuple:
$$s = (\text{presence}, \text{disagreement}, \text{uncertainty\_asymmetry}, \text{diagnosis})$$
- $\text{presence} \in \{\text{BOTH\_PRESENT}, \text{TEXT\_MISSING}, \text{IMAGE\_MISSING}, \text{MISSING\_BOTH}\}$
- $\text{disagreement} \in \{\text{LOW\_JSD}, \text{HIGH\_JSD}\}$ (thresholded at $\tau_{\text{JSD}} = 0.1139$)
- $\text{uncertainty\_asymmetry} \in \{\text{TEXT\_MORE\_RELIABLE}, \text{IMAGE\_MORE\_RELIABLE}, \text{BALANCED\_UNCERTAINTY}\}$
- $\text{diagnosis} \in \{\text{CLEAN}, \text{TEXT\_FAILURE}, \text{IMAGE\_FAILURE}, \text{MISSING\_TEXT}, \text{MISSING\_IMAGE}, \dots\}$

### 5.3 Outcome Reward Function ($R$)
Evaluated against validation ground truth $y \in \{0, 1\}^{23}$:
$$R_t = F_1(\hat{y}_{a_t}, y_t) + \lambda \cdot \left[ F_1(\hat{y}_{a_t}, y_t) - F_1(\hat{y}_{\text{unrecovered}}, y_t) \right]$$
where $\lambda = 0.5$, balancing absolute genre accuracy with positive recovery gain.

### 5.4 Outcome-Learning Update Rule
$$N_{t+1}(s_t, a_t) = N_t(s_t, a_t) + 1$$
$$\alpha_t = \max\left(0.05, \frac{\eta_0}{N_{t+1}(s_t, a_t)^\beta}\right)$$
$$Q_{t+1}(s_t, a_t) = Q_t(s_t, a_t) + \alpha_t \cdot \left[ R_t - Q_t(s_t, a_t) \right]$$
where $\eta_0 = 0.20$ and $\beta = 0.5$.

---

## 6. Data Leakage Controls and Split-Integrity Scrutiny

| Verification Check | Standard | Observed Repository Implementation | Verification Verdict |
|---|---|---|:---:|
| **Training Split Separation** | Policy training uses validation data only | Trained strictly on 500 validation samples from `splits['val']`. Zero test samples or test labels accessed during training. | **VERIFIED (PASS)** |
| **Test Set Isolation** | Frozen evaluation on test split | Policy set to `learning_enabled=False`, $\epsilon=0$ during test evaluation. Zero policy updates occurred on test data. | **VERIFIED (PASS)** |
| **Inference Signal Provenance** | Policy observes only inference signals | Context consists strictly of $u_{\text{txt}}, u_{\text{img}}, d_{\text{JSD}}, d_{\text{Hamming}}$, norms, and diagnostic strings. Ground truth $y_{\text{test}}$ was never passed to `select_action()`. | **VERIFIED (PASS)** |
| **Split Partitions** | Zero cross-split sample overlap | `splits['train']` (15,564), `splits['val']` (2,607), and `splits['test']` (7,788) are strictly disjoint sets. | **VERIFIED (PASS)** |
| **Label Alignment** | Consistent 23-genre mapping | Target vectors remain aligned to the canonical 23 genres in `data/label_mapping.json`. | **VERIFIED (PASS)** |

> [!NOTE]
> *Scope of Leakage Verification:* These checks confirm that architectural separation, data split boundaries, and parameter freezing were strictly enforced in code. They demonstrate that test labels did not leak into policy parameters during benchmarking. They do not claim immunity to real-world out-of-distribution drift, which requires continuous streaming evaluation.

---

## 7. Tests Run and Exact Results

### 7.1 Full Test Suite Execution
- **Command:** `python -m unittest discover -s tests -p "test_*.py" -v`
- **Result:** `Ran 73 tests in 37.687s - OK`
- **Total Tests:** **73 passed, 0 failed, 0 errors, 0 skipped.**
- Breakdown by Module:
  - [`tests/test_adaptive_recovery.py`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/tests/test_adaptive_recovery.py): **11/11 PASSED**
  - [`tests/test_baselines.py`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/tests/test_baselines.py): **10/10 PASSED**
  - [`tests/test_dataset.py`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/tests/test_dataset.py): **12/12 PASSED**
  - [`tests/test_diagnosis.py`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/tests/test_diagnosis.py): **6/6 PASSED**
  - [`tests/test_disagreement.py`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/tests/test_disagreement.py): **9/9 PASSED**
  - [`tests/test_presentation_outputs.py`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/tests/test_presentation_outputs.py): **7/7 PASSED**
  - [`tests/test_uncertainty.py`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/tests/test_uncertainty.py): **8/8 PASSED**
  - [`tests/test_corruptions.py`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/tests/test_corruptions.py): **10/10 PASSED**

### 7.2 Improvements Regression Suite Execution
- **Command:** `python tests/test_improvements.py`
- **Result:** `12 PASSED, 0 FAILED across 12 test cases.`

---

## 8. Evaluation Configuration and Reproducibility Details

- **Test Evaluation Sample Size:** 500 test samples (exact sample IDs listed in [`outputs/phase5/run_20261025_phase5_adaptive_recovery/evaluated_sample_ids.txt`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/outputs/phase5/run_20261025_phase5_adaptive_recovery/evaluated_sample_ids.txt)).
- **Total Evaluations per Policy:** $500 \times 8 \text{ conditions} = 4,000$ evaluations per policy (16,000 total evaluations across 4 policies).
- **Random Seed:** `42`
- **Underlying Models:**
  - Text: TF-IDF + Logistic Regression (`outputs/checkpoints/text_baseline.joblib`)
  - Image: ResNet-50 + Logistic Regression (`outputs/checkpoints/image_baseline.joblib`)
  - Multimodal: Late Fusion $\alpha=0.55$ (`outputs/checkpoints/multimodal_baseline.json`)
  - Diagnostic Auditor: Validation 95th-percentile thresholds (`src/auditing/diagnosis.py`)
- **Evaluation Platform:** Windows 11, Python 3.11, PyTorch CPU execution.

---

## 9. Comprehensive Results: Metric Definitions, Tables, and Discrepancy Corrections

### 9.1 Mathematical Definitions of Evaluation Aggregations

To resolve metric ambiguities identified during audit reviews, all results are presented under mathematically distinct aggregation methodologies:

1. **Condition-Averaged Evaluation (Primary Benchmark Metric):**
   - *Method:* Computes Micro-F1 or Macro-F1 independently within each of the eight 500-sample failure conditions, then takes the unweighted arithmetic mean:
     $$\text{Micro-F1}_{\text{cond-mean}} = \frac{1}{8} \sum_{k=0}^{7} \text{Micro-F1}_k$$
   - *Scientific Rationale:* The research design benchmarks eight distinct failure regimes (clean, 3 text degradations, 3 vision degradations, 1 dual failure). Equal weighting (12.5% per condition) ensures that failure robustness across diverse degradation modes is evaluated without letting high-error conditions distort the aggregate.

2. **Globally Pooled Evaluation (Complementary Corpus-Level Metric):**
   - *Method:* Concatenates all 4,000 instance prediction vectors into a single matrix $Y \in \{0, 1\}^{4000 \times 23}$ and computes Micro-F1 and Macro-F1 globally:
     $$\text{Micro-F1}_{\text{pooled}} = \frac{2 \sum_{i=1}^{4000} \text{TP}_i}{2 \sum_{i=1}^{4000} \text{TP}_i + \sum_{i=1}^{4000} \text{FP}_i + \sum_{i=1}^{4000} \text{FN}_i}$$
   - *Characteristics:* High-error regimes (blackout, noise) generate larger counts of false positives and false negatives, giving severe failures higher implicit weight in the denominator.

3. **Instance-Level Mean F1:**
   - *Method:* Computes multi-label F1 individually on each of the 4,000 rows, then takes the sample mean across rows.

### 9.2 Side-by-Side Overall Metric Comparison

The table below presents all three aggregations recomputed directly from [`sample_level_decisions.csv`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/outputs/phase5/run_20261025_phase5_adaptive_recovery/sample_level_decisions.csv):

| Aggregation Methodology | Metric | P1: No Recovery | P2: Fixed Policy | P3: Adaptive Policy | P3 vs P1 Delta | P3 vs P2 Delta |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Condition-Averaged (Primary)** | **Micro-F1**<br>**Macro-F1** | **0.4012**<br>**0.3165** | **0.3942**<br>**0.2945** | **0.4709**<br>**0.3593** | **+0.0697 (+17.4%)**<br>**+0.0428 (+13.5%)** | **+0.0768 (+19.5%)**<br>**+0.0648 (+22.0%)** |
| **Globally Pooled (Corpus-Level)** | **Micro-F1**<br>**Macro-F1** | **0.3973**<br>**0.3030** | **0.3905**<br>**0.2898** | **0.4690**<br>**0.3523** | **+0.0717 (+18.0%)**<br>**+0.0493 (+16.3%)** | **+0.0784 (+20.1%)**<br>**+0.0624 (+21.5%)** |
| **Instance-Level (Sample Mean)** | **Mean F1** | **0.3940** | **0.3901** | **0.4671** | **+0.0731 (+18.6%)** | **+0.0770 (+19.7%)** |

### 9.3 Per-Condition Micro-F1 Breakdown (Table 9)

All four policies were evaluated on **identical sample IDs and identical predictions** across each condition ($N = 500$ samples per condition):

| Condition ID | Failure Condition | Corrupted Modality | Sample Count | P1: No Recovery Micro-F1 | P2: Fixed Policy Micro-F1 | P2: Fixed Delta ($\Delta_{\text{F1}}$) | P3: Adaptive Policy Micro-F1 | P3: Adaptive Delta ($\Delta_{\text{F1}}$) | P4: Oracle Upper Bound Micro-F1 | P4: Oracle Delta ($\Delta_{\text{F1}}$) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | `Clean_Baseline` | None | 500 | 0.5187 | 0.4652 | **-0.0535** | **0.5528** | **+0.0341** | 0.6586 | +0.1399 |
| **1** | `Text_Truncation_25%` | Text | 500 | 0.4635 | 0.4432 | **-0.0203** | **0.4700** | **+0.0065** | 0.6026 | +0.1391 |
| **2** | `Text_Token_Masking_35%` | Text | 500 | 0.4619 | 0.4364 | **-0.0255** | **0.4690** | **+0.0071** | 0.6047 | +0.1428 |
| **3** | `Text_Empty` | Text | 500 | 0.3836 | 0.3996 | **+0.0160** | **0.3996** | **+0.0160** | 0.3996 | +0.0160 |
| **4** | `Image_Gaussian_Blur_r5` | Vision | 500 | 0.3466 | 0.2272 | **-0.1194** | **0.4016** | **+0.0550** | 0.5882 | +0.2416 |
| **5** | `Image_Gaussian_Noise_s25` | Vision | 500 | 0.4121 | 0.3183 | **-0.0938** | **0.4988** | **+0.0867** | 0.6147 | +0.2026 |
| **6** | `Image_Missing_Blackout` | Vision | 500 | 0.3060 | 0.5820 | **+0.2760** | **0.5820** | **+0.2760** | 0.5820 | +0.2760 |
| **7** | `Dual_Corruption_Trunc_Noise` | Both | 500 | 0.3173 | 0.2813 | **-0.0360** | **0.3937** | **+0.0764** | 0.5126 | +0.1953 |
| **OVERALL** | **Condition Mean (Primary)** | **All** | **4,000** | **0.4012** | **0.3942** | **-0.0070** | **0.4709** | **+0.0697** | **0.5704** | **+0.1692** |
| **OVERALL** | **Globally Pooled (Corpus)** | **All** | **4,000** | **0.3973** | **0.3905** | **-0.0068** | **0.4690** | **+0.0717** | *N/A in CSV* | *N/A in CSV* |

### 9.4 Macro-F1 Correction Notes and Per-Condition Values

> [!IMPORTANT]
> **Macro-F1 Metric Correction Note:**  
> In the preliminary draft of this report, overall Macro-F1 was cited as `0.3040` (No Recovery), `0.2972` (Fixed Policy), and `0.3547` (Adaptive Policy). An independent audit revealed that these figures were drafting clerical errors inconsistent with [`sample_level_decisions.csv`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/outputs/phase5/run_20261025_phase5_adaptive_recovery/sample_level_decisions.csv).  
> The true verified values are:
> - **Condition-Averaged Macro-F1:** No Recovery = **0.3165**, Fixed Policy = **0.2945**, Adaptive Policy = **0.3593** (Gain: **+0.0428** vs No Recovery; **+0.0648** vs Fixed Policy).
> - **Globally Pooled Macro-F1:** No Recovery = **0.3030**, Fixed Policy = **0.2898**, Adaptive Policy = **0.3523** (Gain: **+0.0493** vs No Recovery; **+0.0624** vs Fixed Policy).

The table below provides the full per-condition Macro-F1 breakdown recomputed from [`sample_level_decisions.csv`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/outputs/phase5/run_20261025_phase5_adaptive_recovery/sample_level_decisions.csv):

| Condition ID | Failure Condition | P1: No Recovery Macro-F1 | P2: Fixed Policy Macro-F1 | P3: Adaptive Policy Macro-F1 | P3 vs P1 Delta |
|:---:|---|:---:|:---:|:---:|:---:|
| **0** | `Clean_Baseline` | 0.3943 | 0.3347 | **0.4290** | +0.0347 |
| **1** | `Text_Truncation_25%` | 0.3445 | 0.3217 | **0.3468** | +0.0023 |
| **2** | `Text_Token_Masking_35%` | 0.3366 | 0.3109 | **0.3457** | +0.0091 |
| **3** | `Text_Empty` | 0.2441 | 0.2930 | **0.2930** | +0.0489 |
| **4** | `Image_Gaussian_Blur_r5` | 0.3023 | 0.1748 | **0.3314** | +0.0291 |
| **5** | `Image_Gaussian_Noise_s25` | 0.3289 | 0.2582 | **0.3890** | +0.0601 |
| **6** | `Image_Missing_Blackout` | 0.3340 | 0.4421 | **0.4421** | +0.1081 |
| **7** | `Dual_Corruption_Trunc_Noise` | 0.2475 | 0.2203 | **0.2972** | +0.0497 |
| **ALL** | **Condition Mean** | **0.3165** | **0.2945** | **0.3593** | **+0.0428** |
| **ALL** | **Globally Pooled** | **0.3030** | **0.2898** | **0.3523** | **+0.0493** |

---

## 9.5 Statistical Reliability, Clustered Analysis, and Significance Testing

### 9.5.1 Repeated-Measures Structure and Movie Clustering
The 4,000 evaluation rows in [`sample_level_decisions.csv`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/outputs/phase5/run_20261025_phase5_adaptive_recovery/sample_level_decisions.csv) are generated from **500 unique movie entities** evaluated across **8 conditions** ($i = (m, k), m \in \{1,\dots,500\}, k \in \{0,\dots,7\}$). Because each movie entity shares an identical ground-truth label vector $y_m$, clean text, and poster artwork:
1. Performance on sample $m$ under condition $k_1$ is correlated with performance under condition $k_2$.
2. The Intraclass Correlation Coefficient (ICC) of the instance F1 recovery difference ($D_{m,k} = F_1(\text{P3}) - F_1(\text{P1})$) is **$\rho_{\text{ICC}} = 0.1319$**.
3. With $K = 8$ conditions per cluster, the Design Effect (Variance Inflation Factor) is:
   $$\text{VIF} = 1 + (K - 1) \rho = 1 + 7 \times 0.1319 = 1.9232$$
   yielding an effective independent sample size of **$N_{\text{eff}} = 4,000 / 1.9232 \approx 2,080$**.
4. **Methodological Note:** A naive paired $t$-test treating all 4,000 evaluations as independent underestimates standard error by $1.39\times$ ($\text{SE}_{\text{naive}} = 0.0041$ vs $\text{SE}_{\text{clustered}} = 0.0057$). It is therefore rejected as primary statistical evidence.

### 9.5.2 Clustered Movie-Level Analysis ($N = 500$ Independent Clusters, $df = 499$)
To perform unbiased hypothesis testing on instance-level F1 differences, evaluations for each movie are collapsed across conditions into a single summary difference per movie:
$$\bar{D}_m = \frac{1}{8} \sum_{k=0}^{7} \left[ F_1(y_m, \hat{y}_{m, k, \text{P3}}) - F_1(y_m, \hat{y}_{m, k, \text{P1}}) \right]$$

| Comparison | Unit of Analysis | Statistical Method | Mean Diff ($\bar{D}$) | Clustered SE | 95% Confidence Interval | Test Statistic | $p$-value |
|---|:---:|---|:---:|:---:|:---:|:---:|:---:|
| **P3 vs P1 (Adaptive vs No Recovery)** | 500 Movie Entities | Clustered Paired $t$-test ($df=499$) | +0.0731 | 0.0057 | [+0.0619, +0.0843] | $t = 12.79$ | $1.34 \times 10^{-32}$ |
| **P3 vs P1 (Adaptive vs No Recovery)** | 500 Movie Entities | Wilcoxon Signed-Rank Test | +0.0731 | — | — | $W = 23,564.0$ | $1.15 \times 10^{-31}$ |
| **P3 vs P2 (Adaptive vs Fixed Policy)** | 500 Movie Entities | Clustered Paired $t$-test ($df=499$) | +0.0770 | 0.0060 | [+0.0652, +0.0888] | $t = 12.82$ | $1.02 \times 10^{-32}$ |
| **P3 vs P2 (Adaptive vs Fixed Policy)** | 500 Movie Entities | Wilcoxon Signed-Rank Test | +0.0770 | — | — | $W = 24,620.5$ | $9.43 \times 10^{-30}$ |

*Critical Distinction:* Tests on instance-level F1 evaluate differences in individual sample predictions; they do **not** directly test non-linear corpus aggregates like Micro-F1 or Macro-F1.

### 9.5.3 Cluster Bootstrap over 500 Movies ($B = 5,000$ Iterations)

To evaluate the statistical significance of non-linear aggregate metrics (Micro-F1, Macro-F1) while rigorously respecting movie clustering, a 5,000-iteration cluster bootstrap was executed (`seed = 42`, $B = 5,000$). In each iteration, 500 movie entities were resampled with replacement, and all metrics were recomputed across their respective 8 condition evaluations ($500 \times 8 = 4,000$ evaluations per resample).

The table below reports all 10 combinations of comparisons and aggregation metrics, displaying the direct sample point-estimate difference alongside the bootstrap mean difference:

| Comparison | Target Metric | Metric Aggregation Unit | Direct Point Diff ($\Delta_{\text{point}}$) | Bootstrap Mean Diff ($\bar{\Delta}^*$) | Bootstrap SE ($\text{SE}^*$) | 95% Bootstrap CI | Empirical $p$-value |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **P3 vs P1** | **Instance-level Mean F1** | Sample instance | +0.0731 | +0.0732 | 0.0057 | [+0.0619, +0.0843] | $p < 0.0002$ |
| **P3 vs P1** | **Globally Pooled Micro-F1** | Pooled corpus | +0.0716 | +0.0717 | 0.0046 | [+0.0627, +0.0810] | $p < 0.0002$ |
| **P3 vs P1** | **Globally Pooled Macro-F1** | 23 Genre classes | +0.0493 | +0.0487 | 0.0060 | [+0.0370, +0.0606] | $p < 0.0002$ |
| **P3 vs P1** | **Condition-Mean Micro-F1** | 8 Condition blocks | **+0.0697** | **+0.0698** | **0.0049** | **[+0.0603, +0.0795]** | **$p < 0.0002$** |
| **P3 vs P1** | **Condition-Mean Macro-F1** | 8 Condition blocks | **+0.0428** | **+0.0425** | **0.0058** | **[+0.0310, +0.0540]** | **$p < 0.0002$** |
| **P3 vs P2** | **Instance-level Mean F1** | Sample instance | +0.0770 | +0.0771 | 0.0060 | [+0.0652, +0.0888] | $p < 0.0002$ |
| **P3 vs P2** | **Globally Pooled Micro-F1** | Pooled corpus | +0.0784 | +0.0784 | 0.0052 | [+0.0681, +0.0888] | $p < 0.0002$ |
| **P3 vs P2** | **Globally Pooled Macro-F1** | 23 Genre classes | +0.0624 | +0.0619 | 0.0062 | [+0.0497, +0.0741] | $p < 0.0002$ |
| **P3 vs P2** | **Condition-Mean Micro-F1** | 8 Condition blocks | **+0.0768** | **+0.0768** | **0.0054** | **[+0.0661, +0.0874]** | **$p < 0.0002$** |
| **P3 vs P2** | **Condition-Mean Macro-F1** | 8 Condition blocks | **+0.0648** | **0.0644** | **0.0066** | **[+0.0512, +0.0773]** | **$p < 0.0002$** |

Across all 5,000 bootstrap iterations, exactly zero resamples produced a non-positive delta ($\Delta \le 0$) across any of the 10 metric comparisons, empirically bounding the one-sided significance at $p < 1/5000 = 0.0002$.

#### 9.5.3.1 Methodological Clarification: Bootstrap Mean vs. Point-Estimate Differences

Readers comparing the direct sample point estimates in Section 9.2/9.3 with Table 9.5.3 will observe minor differences between the direct sample point-estimate difference ($\Delta_{\text{point}}$) and the bootstrap mean difference ($\bar{\Delta}^*$). These differences are expected under finite resampling and stem from two methodological factors:

1. **Non-Linear Metrics under Empirical Bootstrap Distribution:**  
   Micro-F1 and Macro-F1 are non-linear ratio estimators ($\frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$ and $\frac{1}{C}\sum_c F_{1, c}$). For non-linear functions $g(\cdot)$, the expected value under empirical bootstrap resampling does not in general equal the sample plug-in estimate: $\mathbb{E}^*[g(X^*)] \neq g(X_{\text{sample}})$.
   
   *Reassessment of Bias Mechanism:* It is tempting to invoke an elementary convexity argument (such as Jensen's inequality applied to $1/d$ with $g''(d) = 2n/d^3 > 0$) to argue for a specific bias direction. However, such univariate reasoning does not apply to the actual multi-label metric:
   - Macro-F1 is a multivariate ratio of random variables where both the numerator ($2\text{TP}_c$) and denominator ($2\text{TP}_c + \text{FP}_c + \text{FN}_c$) covary across resamples.
   - Macro-F1 averages across 23 classes with highly heterogeneous support and varying class correlations.
   - The evaluated target is the *difference* in performance between two models ($\Delta = \text{Macro-F1}_{\text{P3}} - \text{Macro-F1}_{\text{Baseline}}$), whose joint sampling behavior depends on differential class-level predictions.
   
   A positive second derivative of a reciprocal denominator alone does not determine the direction or magnitude of bias for this difference. We therefore do not claim a specific bias direction from the second derivative alone. Instead, we provide the cautious statement that the observed small differences (e.g., $-0.0006$ on Globally Pooled Macro-F1: $+0.0487$ bootstrap mean vs $+0.0493$ sample point difference; $-0.0003$ to $-0.0005$ on Condition-Mean Macro-F1) arise from the non-linear metrics and the empirical bootstrap distribution over finite cluster resamples, with the precise mathematical cause and sign of the net bias across models not established.

2. **Finite Monte Carlo Sampling Error:**  
   Because the bootstrap is computed over a finite simulation of $B = 5,000$ random cluster resamples rather than an infinite sample ($B \to \infty$), the Monte Carlo standard error of the bootstrap mean is $\text{SE}^*_{\text{boot}} / \sqrt{B} \approx 0.0057 / \sqrt{5000} \approx 0.00008$. This accounts for minor stochastic fluctuations of order $\pm 0.0001$ observed on linear metrics (e.g., Instance-level Mean F1: $+0.0732$ bootstrap mean vs $+0.0731$ sample point difference).

Importantly, these minute differences ($\le 0.0006$) are an order of magnitude smaller than the bootstrap standard errors ($\text{SE}^* \approx 0.0046\text{--}0.0066$) and have zero effect on our statistical conclusions: across all 10 evaluated metric comparisons and all 5,000 resamples, $100\%$ of iterations favor the adaptive policy over both baselines ($p < 0.0002$).

---

## 10. Failure Cases and Negative Recovery Deltas

### 10.1 Negative Deltas in the Fixed Heuristic Policy
The evaluation revealed a severe failure mode in fixed rule-based routing:
- On **6 out of 8 conditions**, the fixed policy degraded performance below unrecovered fusion:
  - `Image_Gaussian_Blur_r5`: **-0.1194 Micro-F1** (0.3466 $\to$ 0.2272)
  - `Image_Gaussian_Noise_s25`: **-0.0938 Micro-F1** (0.4121 $\to$ 0.3183)
  - `Clean_Baseline`: **-0.0535 Micro-F1** (0.5187 $\to$ 0.4652)
  - `Dual_Corruption_Trunc_Noise`: **-0.0360 Micro-F1** (0.3173 $\to$ 0.2813)
  - `Text_Token_Masking_35%`: **-0.0255 Micro-F1** (0.4619 $\to$ 0.4364)
  - `Text_Truncation_25%`: **-0.0203 Micro-F1** (0.4635 $\to$ 0.4432)
- **Root Cause of Fixed Policy Degradation:** The heuristic rules triggered false alarms due to natural cross-modal divergence, routing samples into single-modality fallbacks. In particular, routing to the weaker image branch on noisy or conflicting samples caused severe collapse.

### 10.2 Adaptive Policy Delta Analysis
- The Adaptive Policy produced **positive recovery deltas across all 8 failure conditions** ($\Delta_{\text{F1}} \ge +0.0065$).
- Under subtle text corruptions (`Text_Truncation` and `Text_Masking`), recovery gains were modest (+0.0065 to +0.0071) because visual features in MM-IMDb are weak (unimodal image F1 ~0.42), meaning fallback to vision cannot compensate for text degradation as effectively as fallback to text compensates for visual degradation.
- Under Dual Corruption, while the adaptive policy achieved +0.0764 gain over unrecovered fusion (0.3937 vs 0.3173), absolute F1 remains depressed (0.3937 vs 0.5528 clean) due to irrecoverable dual-information loss.

---

## 11. Action Selection Frequencies

| Condition Name | Static Fusion (%) | Reliability Weighted (%) | Conservative 50/50 (%) | Text Only (%) | Image Only (%) | Effective Routing Behavior |
|---|:---:|:---:|:---:|:---:|:---:|---|
| `Clean_Baseline` | 12.2% | 15.0% | 0.0% | 72.8% | 0.0% | Prioritizes text and cooperative fusion; eliminates erroneous image fallback. |
| `Text_Truncation_25%` | 10.8% | 15.6% | 0.0% | 73.6% | 0.0% | Retains preserved text features while selectively weighting. |
| `Text_Token_Masking_35%` | 12.2% | 10.8% | 0.0% | 77.0% | 0.0% | Resiliently maintains unmasked text tokens. |
| `Text_Empty` | 0.0% | 0.0% | 0.0% | 0.0% | **100.0%** | 100% adherence to modality feasibility constraint. |
| `Image_Gaussian_Blur_r5` | 59.2% | 1.8% | 0.0% | 39.0% | 0.0% | Suppresses collapsed image branch; routes 39% to pure text. |
| `Image_Gaussian_Noise_s25` | 25.6% | 7.4% | 0.0% | 67.0% | 0.0% | Strong text routing (67%) avoids noisy visual features. |
| `Image_Missing_Blackout` | 0.0% | 0.0% | 0.0% | **100.0%** | 0.0% | 100% adherence to modality feasibility constraint. |
| `Dual_Corruption_Trunc_Noise`| 22.6% | 8.4% | 0.0% | 69.0% | 0.0% | Favors surviving text semantics over noisy visual poster. |

---

## 12. Runtime Overhead & Computational Efficiency

- **Timing Methodology:** Micro-benchmarked using Python's high-resolution `time.perf_counter()` over 4,000 evaluations.
- **Adaptive Policy Decision Overhead:**
  - Mean Latency: **0.0112 ms / sample** ($11.2\ \mu\text{s}$ per sample).
  - Throughput: **89,385 decisions / second**.
  - Memory Footprint: Contextual Q-table with 17 discrete states requires **< 15 KB** RAM.
- **End-to-End Multimodal Pipeline Latency:**
  - Text Inference: ~0.14 ms / sample.
  - Image Inference (cached ResNet features): ~0.40 ms / sample.
  - Multimodal Late Fusion: ~0.002 ms / sample.
  - Auditor Disagreement & Uncertainty: ~0.035 ms / sample.
  - Adaptive Recovery Router: ~0.011 ms / sample.
  - Total End-to-End Audit + Recovery Overhead: **~0.046 ms / sample** (~21,700 samples/s).
- *Limitation Note:* While aggregate mean latency is profiled accurately, individual per-sample latency timestamps were not serialized to [`sample_level_decisions.csv`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/outputs/phase5/run_20261025_phase5_adaptive_recovery/sample_level_decisions.csv).

---

## 13. Verification of the "Self-Evolving" Claim

As required by Step 5 of the prompt specification, each component of the "self-evolving" claim was subjected to empirical verification:

| Self-Evolution Criterion | Required Evidence | Empirical Finding & Artifact Location | Verification Status |
|---|---|---|:---:|
| **Outcome Logging** | Earlier action's outcome is recorded | Recorded 2,500 validation training episodes with step, state, action, reward, $Q_{\text{old}}, Q_{\text{new}}$, and visit count in [`outputs/phase5/run_20261025_phase5_adaptive_recovery/policy_training_history.json`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/outputs/phase5/run_20261025_phase5_adaptive_recovery/policy_training_history.json). | **VERIFIED** |
| **State Updating** | Outcome updates policy state | Action values updated across 17 distinct context regimes in [`outputs/phase5/run_20261025_phase5_adaptive_recovery/trained_adaptive_policy.json`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/outputs/phase5/run_20261025_phase5_adaptive_recovery/trained_adaptive_policy.json). Replay verified exact adherence to update math ($1.26 \times 10^{-5}$ max delta). | **VERIFIED** |
| **Decision Transition** | Updated state alters subsequent action choice | Unit test `test_05_controlled_reward_sequence_changes_subsequent_decision` in [`tests/test_adaptive_recovery.py`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/tests/test_adaptive_recovery.py) verified that feeding controlled reward sequences transitions decision from `STATIC_FUSION` to `TEXT_ONLY` on identical context inputs. | **VERIFIED** |
| **Identical-Sample Comparison** | Compared against fixed baseline on same samples | Exactly 500 test samples evaluated across 8 conditions (4,000 rows) with 1-to-1 sample ID alignment in [`outputs/phase5/run_20261025_phase5_adaptive_recovery/sample_level_decisions.csv`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/outputs/phase5/run_20261025_phase5_adaptive_recovery/sample_level_decisions.csv). | **VERIFIED** |
| **Leakage Isolation** | Test labels never leak into action selection | `select_action(context)` accepts only observable inference signals; policy training was isolated to `splits['val']`; policy was frozen during test evaluation. | **VERIFIED** |
| **Measured Improvement** | Outperforms fixed policy and unrecovered fusion | Adaptive policy achieved **0.4709 Condition-Mean Micro-F1** vs **0.3942** (Fixed Policy) and **0.4012** (Unrecovered), providing positive recovery across all 8 conditions. | **VERIFIED** |

---

## 14. Confirmation of Phase 1–3 Baseline Artifact Preservation

All Phase 1–3 core assets remain 100% bitwise intact against initial Git commit `6ff6058`:

| Relative Asset Path | MD5 Checksum | Status |
|---|---|:---:|
| `data/split_indices.json` | `b0b79868f5227c2ed283781c60099fb9` | **UNMODIFIED** |
| `data/label_mapping.json` | `08c22181091fc73ef08dbbf247f9c3a5` | **UNMODIFIED** |
| `data/metadata.csv` | `417cf26427ee2ea6fb60f7e540f163fd` | **UNMODIFIED** |
| `outputs/checkpoints/text_baseline.joblib` | `5b7026119dc343aaa0f2a0b03e6276c9` | **UNMODIFIED** |
| `outputs/checkpoints/image_baseline.joblib` | `7f2f42a15de2b4e3d36c0540cfc5f29d` | **UNMODIFIED** |
| `outputs/checkpoints/multimodal_baseline.json` | `6d32c8a5b8ffb1e6b0063d03a53f14c4` | **UNMODIFIED** |
| `outputs/metrics/baseline_comparison.csv` | `3009e7f48eab50304e353501b3bdab87` | **UNMODIFIED** |

---

## 15. Limitations and Unresolved Risks

1. **Unestablished Source of Preliminary Draft Macro-F1 Figures:**
   The preliminary report draft cited Macro-F1 values of `0.3040` (No Recovery), `0.2972` (Fixed Policy), and `0.3547` (Adaptive Policy). Recomputation from [`sample_level_decisions.csv`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/outputs/phase5/run_20261025_phase5_adaptive_recovery/sample_level_decisions.csv) demonstrated that true condition-averaged Macro-F1 is `0.3165` / `0.2945` / `0.3593` and pooled Macro-F1 is `0.3030` / `0.2898` / `0.3523`. The exact source calculation of the earlier draft numbers was not preserved in JSON or CSV outputs, indicating an undocumented clerical transcription discrepancy in early drafting notes.

2. **Absence of Per-Sample Oracle Prediction Vectors (`p4_oracle_pred`):**
   While [`sample_level_decisions.csv`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/outputs/phase5/run_20261025_phase5_adaptive_recovery/sample_level_decisions.csv) records `p4_oracle_action`, it omits `p4_oracle_pred`. While per-condition Oracle Micro-F1 is logged in [`policy_comparison_metrics.csv`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/outputs/phase5/run_20261025_phase5_adaptive_recovery/policy_comparison_metrics.csv), instance-level Oracle F1 cannot be directly evaluated from the decision CSV without executing the oracle action mapping logic.

3. **Absence of Per-Sample Latency Timestamps:**
   Execution latency is reported as an aggregate benchmark mean ($11.2\ \mu\text{s}$ per decision), but individual per-sample runtime timestamps were not exported to [`sample_level_decisions.csv`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/outputs/phase5/run_20261025_phase5_adaptive_recovery/sample_level_decisions.csv), preventing empirical latency distribution or tail-percentile analysis from the CSV alone.

4. **Diagnostic Sensitivity for Subtle Text Degradation:**
   Auditor thresholds calibrated to 95th-percentile validation signals show lower sensitivity to text truncation (25%) and token masking (35%) than to complete modality dropouts or heavy vision noise. As a consequence, recovery gains under these conditions are modest (+0.0065 to +0.0071 Micro-F1), highlighting a need for semantic coherence scoring (e.g., perplexity or semantic density) in the diagnostic auditor.

5. **Modality Asymmetry and Policy Preference for `TEXT_ONLY`:**
   In MM-IMDb, plot summaries are substantially more predictive of genres than poster artwork (unimodal text F1: 0.6056 vs unimodal image F1: 0.4151). Consequently, the learned policy frequently selects `TEXT_ONLY` (72.8% on clean, 73.6% on truncation, 77.0% on masking, 67.0% on image noise). This behavior correctly reflects the underlying dataset information asymmetry rather than policy dysfunction, but it highlights that unimodal text dominance restricts the opportunity for vision-assisted recovery when text is degraded.

6. **Discrete Context Granularity:**
   Continuous uncertainty and disagreement signals ($u_{\text{txt}}, u_{\text{img}}, d_{\text{JSD}}$) are discretized into categorical partitions. While ensuring exact tabular convergence, fine-grained variations near boundary thresholds are mapped to identical context bins.

7. **Dual Corruption Ceiling:**
   When both text and vision are degraded simultaneously, modal re-weighting is constrained by the loss of information across all channels. Recovery under severe dual failure requires generative imputation or human-in-the-loop abstention.

---

## 16. Recommended Next Steps (Future Recommendations — None Implemented)

> [!NOTE]
> In strict accordance with the documentation-only correction scope, no code or benchmark modifications have been made. The following are architectural recommendations for future development:

1. **Benchmark Runner Output Enhancements:**
   - Update [`scripts/run_phase5_evaluation.py`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/scripts/run_phase5_evaluation.py) to export `p4_oracle_pred` alongside `p4_oracle_action` in `sample_level_decisions.csv`.
   - Log per-sample inference and decision latency timestamps into the decision CSV.
   - Include both condition-averaged and globally pooled Macro-F1 directly in `phase5_benchmark_summary.json`.
2. **Continuous Feature Contextual Bandit (LinUCB / Neural Bandit):**
   - Extend tabular state discretization to continuous context spaces using linear contextual bandits ($Q(s, a) = \theta_a^T \phi(s)$) with recursive least-squares updates.
3. **Selective Classification & Abstention Integration:**
   - Integrate `ConfidenceRecoveryRouter` with the adaptive policy so that when dual corruption is diagnosed and expected recovery reward is below an abstention cutoff, instances are routed to human review.
4. **Online Self-Evolving Stream Simulation:**
   - Benchmark policy adaptation in non-stationary streaming deployments where failure distributions drift over time.

---

## 17. Stop Confirmation

**Documentation-only corrections for Phase 5 are complete. Exactly 73/73 unit tests pass. All Phase 1–3 model weights, split partitions, and benchmark CSVs remain 100% bitwise unmodified. No Phase 6 actions or code changes have commenced. This report is finalized for user review and approval.**
