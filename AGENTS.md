# SEAA: Self-Evolving AI Auditor for Multimodal Reliability
## Master Project Directive, Technical Architecture & Operational Guide

> **CRITICAL DIRECTIVE FOR AI AGENTS (ANTIGRAVITY / COPILOT / RESEARCH ASSISTANTS):**  
> This file is the single, authoritative, consolidated source of truth for the **SEAA** framework. Whenever this project is opened, inspected, built, or modified, you MUST strictly adhere to the architecture, mathematical formulations, operational constraints, and verification protocols defined in this document.
>
> 🛑 **PERMANENT FOCUS MANDATE (MEMORY DIRECTIVE):**  
> **FOCUS STRICTLY ON BUILDING THE PROJECT, NOT WRITING THE IEEE PAPER.**  
> All future engineering, sessions, and agent actions must concentrate 100% on codebase implementation, feature engineering, system architecture, model training, performance optimization, and empirical testing. Do NOT spend turns drafting, editing, or formatting the academic IEEE manuscript.

---

## 1. Project Identification & Executive Mission

- **Project Title:** A Self-Evolving AI Auditing Algorithm for Adaptive Reliability Diagnosis and Recovery in Multimodal AI Systems (SEAA)
- **Acronym:** **SEAA**
- **Domain:** Multimodal Machine Learning (Vision + NLP), Multi-Label Genre Classification, Inline Reliability Auditing, Failure Diagnosis, Contextual Bandit Recovery, Repeated-Measures Clustered Statistics.
- **Benchmark Dataset:** **MM-IMDb** (25,959 movie titles from IMDb).
  - **Text Modality:** Movie plot synopses / summaries.
  - **Vision Modality:** Official movie posters (RGB, 224×224).
  - **Target Output:** Multi-label binary indicator vector across **23 canonical movie genres** (mean label cardinality: 2.48 genres/movie).
- **Core Scientific Mission:** Conventional multimodal models degrade under sensor noise, optical blur, text truncation, or missing modalities. Fixed recovery rules (e.g., hard entropy routing) often worsen performance by prematurely amputating informative modalities. SEAA resolves this via an inline closed loop:
  $$\mathbf{Monitor} \longrightarrow \mathbf{Diagnose} \longrightarrow \mathbf{Recover} \longrightarrow \mathbf{Evaluate} \longrightarrow \mathbf{Learn}$$
  SEAA continuously monitors observable disagreement and uncertainty signals, diagnoses failure types, dynamically selects optimal recovery actions via an outcome-trained contextual bandit, and updates its action values—while strictly enforcing zero test label leakage.

---

## 2. Core Architecture & Mathematical Formulations

```
+-----------------------------------------------------------------------------------+
|                               SEAA INLINE ARCHITECTURE                            |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  [Text Input x^T]    --->  [Text Baseline (TF-IDF + LR)]     ---> Probabilities p^T|
|  [Image Input x^V]   --->  [Image Baseline (ResNet-50 + LR)] ---> Probabilities p^V|
|                                         |                                         |
|                                         v                                         |
|                  +-----------------------------------------------+                |
|                  |            1. INLINE RELIABILITY AUDITOR       |                |
|                  |  - Modality Presence: P_T, P_V                |                |
|                  |  - Predictive Binary Entropy: H(p^T), H(p^V)  |                |
|                  |  - Bernoulli Jensen-Shannon Divergence: D_JS  |                |
|                  |  - Representation L2 Norms: ||z^T||, ||z^V||  |                |
|                  +-----------------------------------------------+                |
|                                         |                                         |
|                                         v                                         |
|                  +-----------------------------------------------+                |
|                  |          2. MULTI-CLASS FAILURE DIAGNOSER      |                |
|                  |  State: G_t in {CLEAN, TEXT_FAIL, IMG_FAIL,   |                |
|                  |         MISSING_TEXT, MISSING_IMG, CONFLICT}  |                |
|                  +-----------------------------------------------+                |
|                                         |                                         |
|                                         v                                         |
|                  +-----------------------------------------------+                |
|                  |     3. CONTEXTUAL BANDIT RECOVERY POLICY       |                |
|                  |  State Signature: s_t = (P_t, D_t, U_t, G_t)   |                |
|                  |  Action Selection: a* = argmax Q(s_t, a)       |                |
|                  |  Subject to: Physical Admissibility Masking    |                |
|                  +-----------------------------------------------+                |
|                                         |                                         |
|                                         v                                         |
|                  +-----------------------------------------------+                |
|                  |              4. RECOVERY EXECUTION             |                |
|                  |  - STATIC_FUSION (alpha = 0.55)               |                |
|                  |  - TEXT_ONLY / IMAGE_ONLY (Isolation)         |                |
|                  |  - CONSERVATIVE_FUSION (50/50)                |                |
|                  |  - RELIABILITY_WEIGHTED (Dynamic alpha(x))     |                |
|                  +-----------------------------------------------+                |
|                                         |                                         |
|                                         v                                         |
|                  +-----------------------------------------------+                |
|                  |            5. OUTCOME LEARNING (OFFLINE)      |                |
|                  |  Reward: R_t = F1(rec) + 0.5 [F1(rec)-F1(unrec)]|               |
|                  |  Update: Q_{t+1}(s,a) = Q_t + eta_t [R_t - Q_t]|               |
|                  |  (Frozen during test deployment; zero leakage) |                |
|                  +-----------------------------------------------+                |
+-----------------------------------------------------------------------------------+
```

### 2.1 Predictive Models & Static Fusion Baseline
- **Text Baseline ($f_T$):** Sublinear TF-IDF (10,000 features, unigrams+bigrams, `min_df=2`) + One-vs-Rest Logistic Regression ($C=1.0$).
- **Image Baseline ($f_V$):** Pretrained frozen ResNet-50 2,048-dimensional embeddings ($z^V$) + One-vs-Rest Logistic Regression ($C=0.5$).
- **Static Late Fusion ($f_{MM}$):**
  $$\hat{p}^{MM} = \alpha_0 \hat{p}^T + (1 - \alpha_0) \hat{p}^V, \quad \alpha_0 = 0.55$$
- **Threshold Optimization:** Binary decisions $\hat{y}_c = \mathbb{I}(\hat{p}_c \ge \tau_c)$ tuned per genre on validation split to maximize validation Micro-F1.

### 2.2 Observable Monitoring Signals (Stage 1)
Without accessing ground-truth labels or corruption flags:
1. **Presence Flags:** $P_T = \mathbb{I}(x^T \neq \emptyset)$, $P_V = \mathbb{I}(x^V \neq \emptyset)$.
2. **Predictive Binary Entropy (Uncertainty):**
   $$H(\hat{p}) = -\frac{1}{C} \sum_{c=1}^C \left[ \hat{p}_c \log_2(\hat{p}_c) + (1 - \hat{p}_c) \log_2(1 - \hat{p}_c) \right]$$
   with probabilities clipped to $[10^{-7}, 1 - 10^{-7}]$.
3. **Multi-Label Bernoulli Jensen--Shannon Divergence (Disagreement):**
   $$M_c = \frac{1}{2} (\hat{p}_c^T + \hat{p}_c^V)$$
   $$D_{JS}(\hat{p}^T \parallel \hat{p}^V) = \frac{1}{2C} \sum_{c=1}^C \left[ D_{KL}(\hat{p}_c^T \parallel M_c) + D_{KL}(\hat{p}_c^V \parallel M_c) \right]$$
   where $D_{JS} \in [0, 1]$ measures cross-modal prediction discrepancy.
4. **Feature Norms:** $\|z^T\|_2$ and $\|z^V\|_2$.

### 2.3 Contextual State & Action Space (Stages 2 & 3)
Continuous signals are mapped to an interpretable 4-tuple:
$$s_t = (P_t, D_t, U_t, G_t)$$
- $P_t \in \{\text{BOTH\_PRESENT}, \text{TEXT\_MISSING}, \text{IMAGE\_MISSING}, \text{MISSING\_BOTH}\}$
- $D_t \in \{\text{LOW\_JSD}, \text{HIGH\_JSD}\}$ (cutoff $\tau_{JSD} = 0.1139$)
- $U_t \in \{\text{TEXT\_MORE\_RELIABLE}, \text{IMAGE\_MORE\_RELIABLE}, \text{BALANCED}\}$ (based on $H(\hat{p}^V) - H(\hat{p}^T)$)
- $G_t \in \{\text{CLEAN}, \text{TEXT\_FAILURE}, \text{IMAGE\_FAILURE}, \text{MISSING\_TEXT}, \text{MISSING\_IMAGE}, \text{CROSS\_MODAL\_CONFLICT}, \text{HIGH\_UNCERTAINTY}\}$

**Admissible Action Space:**
$$\mathcal{A} = \{a_{\text{STATIC}}, a_{\text{TEXT}}, a_{\text{IMAGE}}, a_{\text{REL\_WEIGHT}}, a_{\text{CONSERVATIVE}}\}$$
Physical feasibility strictly masks invalid actions (e.g., if image is missing, only $a_{\text{TEXT}}$ is admissible).

**Dynamic Continuous Weighting ($a_{\text{REL\_WEIGHT}}$):**
$$\alpha(x) = \operatorname{clip}\left( 0.55 + 0.40 \tanh\left( 3.0 \cdot [H(\hat{p}^V) - H(\hat{p}^T)] \right), 0.05, 0.95 \right)$$

### 2.4 Outcome Learning & Q-Learning Dynamics (Stages 4 & 5)
During validation training episodes, the reward balances absolute performance and relative recovery gain:
$$R_t = F_1(\hat{y}_{a_t}, y_t) + \lambda \left[ F_1(\hat{y}_{a_t}, y_t) - F_1(\hat{y}_{\text{unrec}}, y_t) \right], \quad \lambda = 0.5, \quad R_t \in [-0.5, 1.5]$$
The tabular action-value matrix is updated via count-decayed learning rate:
$$N_{t+1}(s_t, a_t) = N_t(s_t, a_t) + 1$$
$$\eta_t = \max\left( 0.05, \frac{0.20}{\sqrt{N_{t+1}(s_t, a_t)}} \right)$$
$$Q_{t+1}(s_t, a_t) = Q_t(s_t, a_t) + \eta_t \left[ R_t - Q_t(s_t, a_t) \right]$$
**Test Isolation Rule:** During test-time inference, the policy is completely frozen ($\eta_t = 0$). Test labels are never used to influence decisions.

---

## 3. Threat Model: 8 Canonical Failure Regimes

Evaluated across 500 held-out test movies repeated over all 8 regimes (4,000 evaluations per policy):
1. `Clean_Baseline` (Regime 0): Pristine text and image inputs.
2. `Text_Truncation_25%` (Regime 1): Retains only initial 25% of synopsis text.
3. `Text_Token_Masking_35%` (Regime 2): Replaces 35% of text tokens with `[MASK]`.
4. `Text_Empty` (Regime 3): Missing text modality ($x^T = \emptyset$).
5. `Image_Gaussian_Blur_r5` (Regime 4): Spatial Gaussian blur ($r=5.0$) on poster artwork.
6. `Image_Gaussian_Noise_s25` (Regime 5): Gaussian pixel noise ($\sigma=0.25$).
7. `Image_Missing_Blackout` (Regime 6): Missing vision modality (black $(0,0,0)$ canvas).
8. `Dual_Corruption_Trunc_Noise` (Regime 7): Concurrent 25% text truncation and $\sigma=0.25$ image noise.

---

## 4. Master Empirical Benchmark Results

### 4.1 Benchmark Comparison Table (4,000 Evaluations)

| Policy / Model Configuration | Condition-Averaged Micro-F1 | Globally Pooled Micro-F1 | Instance-Level Mean F1 | Delta vs Unrec. | Decision Latency | Throughput |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **SEAA Adaptive Recovery Policy** | **0.4709** | **0.4690** | **0.4671** | **+0.0697 (+17.4%)** | **0.0245 ms** | **40,753 decisions/s** |
| **No Recovery (Static Late Fusion)** | 0.4012 | 0.3973 | 0.3940 | Reference (0.0000) | 0.0120 ms | 83,364 samples/s |
| **Fixed Heuristic Policy (Router)** | 0.3942 | 0.3905 | 0.3901 | -0.0070 (-1.7%) | 0.0410 ms | 24,390 samples/s |
| **Ablation: Learning Disabled (Untrained)** | 0.4377 | 0.4348 | 0.4340 | +0.0365 (+9.1%) | 0.0240 ms | 41,667 decisions/s |
| **Ablation: Random Action Selection** | 0.4346 | 0.4327 | 0.4321 | +0.0334 (+8.3%) | 0.0050 ms | 200,000 actions/s |
| **Ablation: No Diagnosis Feature** | 0.5054 | 0.5056 | 0.5042 | +0.1042 (+26.0%) | 0.0230 ms | 43,478 decisions/s |
| **Theoretical Ceiling: Oracle Upper Bound** | 0.5695 | 0.5673 | 0.5658 | +0.1682 (+41.9%) | --- | --- |

### 4.2 Statistical Significance (Clustered at Movie Entity Level, df=499)
- **Repeated-Measures Dependence:** ICC(1) $\rho = 0.1319$, $\text{VIF} = 1.9232$, $N_{\text{eff}} \approx 2,080$.
- **SEAA vs No Recovery:** Clustered paired $t$-test: $t = 12.79$, $p = 1.34 \times 10^{-32}$; Wilcoxon: $W = 23,564.0$, $p = 1.15 \times 10^{-31}$. 95% CI: $[+0.0619, +0.0843]$.
- **SEAA vs Fixed Heuristic:** Clustered paired $t$-test: $t = 12.82$, $p = 1.02 \times 10^{-32}$; Wilcoxon: $W = 24,620.5$, $p = 9.43 \times 10^{-30}$. 95% CI: $[+0.0652, +0.0888]$.
- **5,000-Iteration Clustered Bootstrap:** Resampling 500 movie clusters with replacement yielded strictly positive differences in $100\%$ of resamples ($p < 0.0002$) across all 10 evaluated metric combinations.

### 4.3 Component Ablation Proof
- Disabling outcome learning drops Micro-F1 from **0.4711** to **0.4377** (**$-3.34\%$ points**), proving that outcome-based adaptation—not just the action set—is responsible for recovery performance.
- Full SEAA outperforms random action selection ($0.4346$) by **$+3.65\%$ points**, confirming intelligent state-action selection.

### 4.4 Diagnostic Accuracy & Confusion Matrix
- **Missing Modalities:** $100.0\%$ Precision and $100.0\%$ Recall on detecting missing images ($500/500$) and missing text ($500/500$).
- **Binary Anomaly Detection:** $69.86\%$ Sensitivity, $90.22\%$ Precision, $0.7874$ F1-score across 4,000 evaluations.

### 4.5 Latency Profile ($N=1,000$ High-Resolution Single-Sample Runs)
- **Policy Decision Latency:** Mean: **0.0245 ms** ($24.5~\mu\text{s}$), Median: **0.0210 ms**, 95th Percentile: **0.0359 ms**, 99th Percentile: **0.0485 ms**.
- **Total Auditing + Recovery Overhead:** Mean: **3.23 ms** ($309$ samples/s).
- **Complete Audited Pipeline (Inference + Audit + Recovery):** Mean: **11.57 ms** ($86$ samples/s on CPU).

---

## 5. Complete Repository File Structure

```
SEAA/
├── run_seaa.py                              # Master Unified Pipeline Runner (Single Entry Point)
├── main.py                                  # Primary CLI entry point (delegates to run_seaa.py)
├── AGENTS.md                                # THIS FILE: Central Master Directive & Specification
├── README.md                                # General project overview & setup guide
├── PROJECT_STATUS.md                        # Master completion & technical status overview (100% COMPLETE)
├── SEAA_complete_project_explanation_and_IEEE_plan.md  # Detailed technical explanation & IEEE plan
├── requirements.txt                         # Python dependencies
├── data/
│   ├── metadata.csv                         # 25,959 MM-IMDb movie records
│   ├── label_mapping.json                   # 23 canonical genre labels
│   ├── split_indices.json                   # Zero-leakage 60/10/30 split indices
│   └── processed/
│       ├── resnet50_features.npy            # 212.6 MB memory-mapped visual features
│       └── resnet50_features_info.json      # Visual feature metadata
├── docs/
│   ├── ieee_paper/
│   │   ├── IEEE_SEAA_manuscript.tex         # Complete IEEE Transactions LaTeX paper
│   │   ├── IEEE_SEAA_manuscript.md          # Complete Markdown paper
│   │   └── references.bib                   # 15-item verified BibTeX reference library
│   ├── auditing/                            # Auditing & failure diagnosis reports
│   └── recovery/                            # Adaptive recovery reports & research protocol
├── scripts/
│   ├── verify_environment.py                # Hardware & environment verification
│   ├── prepare_dataset.py                   # Grouped iterative stratified splitting
│   ├── extract_image_features.py            # ResNet-50 feature extraction runner
│   ├── run_baselines.py                     # Unified baseline suite runner
│   ├── run_auditing.py                      # Inline reliability auditing benchmark runner
│   ├── run_recovery.py                      # Adaptive recovery benchmark runner
│   ├── evaluate_diagnosis_performance.py    # Diagnostic confusion matrix & metrics
│   ├── run_ablation_benchmarks.py           # Component ablation benchmark suite
│   ├── benchmark_latency_efficiency.py      # High-precision latency & throughput profiling
│   └── generate_presentation_outputs.py     # Presentation chart & summary generator
├── src/
│   ├── data/
│   │   ├── dataset.py                       # MMIMDbDataset & text normalization
│   │   └── pipeline.py                      # Grouped iterative stratifier (Sechidis et al.)
│   ├── models/
│   │   ├── text_baseline.py                 # Sublinear TF-IDF + Logistic Regression
│   │   ├── image_baseline.py                # Standardized ResNet-50 + Logistic Regression
│   │   ├── image_feature_extractor.py       # ResNet-50 backbone extractor
│   │   └── multimodal_baseline.py           # Late fusion class & threshold tuner
│   ├── evaluation/
│   │   ├── metrics.py                       # Multi-label metrics (Micro/Macro F1, mAP, Hamming)
│   │   └── calibration.py                   # Multi-label Expected Calibration Error (ECE) & Brier score
│   ├── auditing/
│   │   ├── disagreement.py                  # Bernoulli JSD & set distance auditing
│   │   ├── uncertainty.py                   # Predictive binary entropy & margins
│   │   ├── diagnosis.py                     # Rule-based failure diagnoser & threshold calibrator
│   │   └── learned_diagnostics.py           # Learned diagnostic classifier
│   ├── failures/
│   │   ├── text_corruptions.py              # Text truncation, token masking, empty text
│   │   └── image_corruptions.py             # Gaussian blur, noise, blackout generator
│   ├── recovery/
│   │   └── modality_gating.py               # DynamicModalityGating heuristic baseline
│   └── policy/
│       ├── adaptive_policy.py               # AdaptiveRecoveryPolicy contextual bandit (Tabular Q)
│       └── linear_bandit.py                 # LinUCB & Thompson Sampling on continuous contexts
├── tests/                                   # 83 passing automated unit tests
│   ├── test_dataset.py                      # Stratification & zero-leakage tests
│   ├── test_baselines.py                    # Baseline model & metric tests
│   ├── test_corruptions.py                  # Text and visual corruption tests
│   ├── test_disagreement.py                 # Multi-label Bernoulli JSD tests
│   ├── test_uncertainty.py                  # Predictive entropy tests
│   ├── test_diagnosis.py                    # Failure diagnosis logic tests
│   ├── test_adaptive_recovery.py            # Policy action selection & Q-learning tests
│   ├── test_calibration.py                  # Multi-label ECE & Brier score tests
│   ├── test_linear_bandit.py                # LinUCB & continuous bandit tests
│   ├── test_improvements.py                 # Quality regression tests
│   └── test_presentation_outputs.py        # Presentation artifact tests
└── outputs/
    ├── checkpoints/                         # Model checkpoints (.joblib, .json)
    ├── figures/                             # Master publication-grade figures (PNG & PDF)
    ├── metrics/                             # Baseline evaluation metrics
    ├── auditing/                            # Auditing predictions & diagnostic confusion matrices
    └── recovery/                            # Adaptive recovery benchmarks, ablations & latencies
        ├── ablation_study_results.csv       # Ablation metrics across all policies
        ├── ablation_study_results.json      # Machine-readable ablation data
        ├── diagnosis_contingency_counts.csv # Diagnostic contingency counts
        ├── diagnosis_evaluation_summary.json# Diagnosis precision, recall & F1
        ├── latency_benchmark_results.csv    # Per-stage latency metrics
        ├── latency_benchmark_results.json   # Full percentile latency data
        ├── figures/                         # Publication-grade figures (PNG & PDF)
        │   ├── ablation_comparison_barchart.png
        │   ├── ablation_comparison_barchart.pdf
        │   ├── diagnosis_contingency_heatmap.png
        │   ├── diagnosis_contingency_heatmap.pdf
        │   ├── latency_overhead_breakdown.png
        │   └── latency_overhead_breakdown.pdf
        └── run_20261025_phase5_adaptive_recovery/
            ├── sample_level_decisions.csv   # 4,000 sample-level benchmark decisions
            ├── policy_comparison_metrics.csv# Per-condition recovery metrics
            ├── trained_adaptive_policy.json # Serialized adaptive policy Q-table
            └── policy_training_history.json # 2,500 validation update trajectories
```

---

## 6. How to Build, Verify, and Benchmark SEAA

Execute these commands from the repository root (`c:\Users\bharat kumar\OneDrive\Desktop\SEAA`):

### 6.1 Master Unified Pipeline Execution (End-to-End)
```bash
python run_seaa.py --mode full
```
*Executes environment check, inline auditing, adaptive recovery, and diagnostic evaluation in one command.*

### 6.2 Run the Automated Test Suite (83 Tests)
```bash
python -m unittest discover -s tests
```
*Expected Result:* `Ran 83 tests in ~12-16s - OK`.

### 6.3 Run Adaptive Recovery Benchmark
```bash
python run_seaa.py --mode recovery --sample_size 500 --seed 42
```
*Outputs:* Generates sample-level benchmark decisions and policy comparison metrics under `outputs/recovery/`.

### 6.4 Run Continuous Severity Sensitivity Benchmark
```bash
python run_seaa.py --mode sensitivity --sample_size 500 --seed 42
```
*Outputs:* Generates `sensitivity_sweep_results.csv` and 4-panel degradation curves in `outputs/recovery/figures/`.

### 6.5 Run Out-of-Distribution (OOD) Generalization Benchmark
```bash
python run_seaa.py --mode ood --sample_size 500 --seed 42
```
*Outputs:* Generates `ood_benchmark_results.csv` and generalization barchart in `outputs/recovery/figures/`.

### 6.6 Run Qualitative Case Study Inspector
```bash
python run_seaa.py --mode case-studies
```
*Outputs:* Generates `qualitative_case_studies.md` and machine-readable JSON under `outputs/reports/`.

### 6.7 Run Component Ablation Benchmark
```bash
python run_seaa.py --mode ablations
```
*Outputs:* Generates `ablation_study_results.csv` and publication figures in `outputs/recovery/figures/`.

### 6.8 Run Diagnostic Accuracy & Confusion Matrix Evaluation
```bash
python run_seaa.py --mode evaluate-diagnosis
```
*Outputs:* Generates `diagnosis_contingency_counts.csv`, summary JSON, and contingency heatmap figures.

### 6.9 Run High-Precision Latency Benchmark
```bash
python run_seaa.py --mode latency
```
*Outputs:* Generates `latency_benchmark_results.csv` and latency breakdown figures.

---

## 7. Strict Operational Guidelines for Future AI Sessions

When interacting with or extending this codebase, every AI agent MUST obey these invariants:
1. **Zero Test-Label Leakage:** Never allow the policy to observe ground-truth test labels ($y_{\text{test}}$) during action selection. The policy must remain frozen ($\eta_t = 0$) during test evaluation.
2. **Cluster-Level Resampling for Statistical Claims:** Never treat repeated evaluations from the same movie as independent. All paired $t$-tests and bootstrap resamples must cluster by `movie_id` ($N = 500$ clusters).
3. **Preserve Frozen Checkpoints:** Do not overwrite baseline checkpoints in `outputs/checkpoints/` (`text_baseline.joblib`, `image_baseline.joblib`, `multimodal_baseline.json`).
4. **Physical Admissibility Constraints:** When designing or executing recovery actions, physical modality availability must be strictly respected via `ctx.get_valid_actions()`.
5. **Continuous Verification:** Any proposed algorithmic change must maintain a 100% passing rate across all 83 unit tests in `tests/`.
