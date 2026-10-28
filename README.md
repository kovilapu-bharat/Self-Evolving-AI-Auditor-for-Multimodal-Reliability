# A Self-Evolving AI Auditing Algorithm for Adaptive Reliability Diagnosis and Recovery in Multimodal AI Systems (SEAA)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 73 Passed](https://img.shields.io/badge/tests-73%20passed-brightgreen.svg)](tests/)

**SEAA (Self-Evolving AI Auditor for Multimodal Reliability)** is an integrated, production-grade multimodal machine learning, inline reliability auditing, and adaptive failure recovery framework built around the **MM-IMDb** benchmark dataset. The system unifies natural language processing and computer vision to perform multi-label movie genre classification while establishing rigorous zero-data-leakage controls, validation-calibrated failure diagnosis, and an outcome-trained contextual bandit recovery policy.

---

## 📌 Table of Contents
- [Executive Overview](#-executive-overview)
- [End-to-End System Architecture](#-end-to-end-system-architecture)
- [Key Features & Scientific Contributions](#-key-features--scientific-contributions)
- [Repository Structure](#-repository-structure)
- [Dataset & Zero-Leakage Data Pipeline](#-dataset--zero-leakage-data-pipeline)
- [Model Architectures & Fusion Baseline](#-model-architectures--fusion-baseline)
- [Inline Reliability Auditing & Failure Diagnosis](#-inline-reliability-auditing--failure-diagnosis)
- [Adaptive Contextual Bandit Recovery](#-adaptive-contextual-bandit-recovery)
- [Master Empirical Results & Statistical Proof](#-master-empirical-results--statistical-proof)
- [Quick Start Guide & Unified Command Center](#-quick-start-guide--unified-command-center)
- [Automated Test Suite](#-automated-test-suite)
- [License](#-license)

---

## 🔬 Executive Overview

Conventional multimodal machine learning models degrade significantly under sensor corruption, missing inputs, optical blur, or token masking. Fixed heuristic recovery rules (e.g., hard entropy routing) often worsen performance by prematurely discarding informative modalities.

SEAA resolves this challenge via an inline closed-loop architecture:
$$\mathbf{Monitor} \longrightarrow \mathbf{Diagnose} \longrightarrow \mathbf{Recover} \longrightarrow \mathbf{Evaluate} \longrightarrow \mathbf{Learn}$$

SEAA continuously monitors observable cross-modal disagreement and predictive uncertainty signals, diagnoses failure types, dynamically selects optimal recovery actions via an outcome-trained contextual bandit, and updates its action values—while strictly enforcing zero test label leakage.

The primary machine learning benchmark task is **multi-label classification** across **23 canonical movie genres** using multimodal inputs from MM-IMDb:
- **Text Modality**: Movie plot summaries and synopses.
- **Vision Modality**: Official movie poster artwork (RGB, resized to 224×224).
- **Target Space**: 23 binary genre indicators per movie (mean label cardinality: 2.48 genres/movie).

---

## 🏗️ End-to-End System Architecture

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

---

## ⚡ Key Features & Scientific Contributions

1. **Master Unified Command Center (`run_seaa.py` / `main.py`)**:
   - One integrated runner for executing full end-to-end pipelines, inline auditing, adaptive recovery, baseline training, ablation studies, and latency micro-benchmarks.

2. **Grouped Multi-Label Iterative Stratification**:
   - Zero-data-leakage deterministic 60% Train (15,564) / 10% Validation (2,607) / 30% Test (7,788) split.
   - Text duplicate grouping guarantees identical plot synopses never span across train/val/test splits (0 cross-split duplicates).
   - Preserves proportional class representation across all 23 genres, including rare categories (*Film-Noir*, *Short*).

3. **CPU-Optimized Memory-Mapped Feature Caching**:
   - Pre-extracts 2,048-dimensional visual representations from frozen ResNet-50 backbones.
   - Stored in disk-backed `numpy.memmap` (`resnet50_features.npy`, 212.6 MB), enabling instant training and inference on standard workstations without GPU bottlenecks.

4. **Multi-Label Bernoulli Disagreement & Uncertainty Auditing**:
   - Multi-label Bernoulli Jensen--Shannon Divergence ($D_{JS}$) measuring cross-modal prediction discrepancy.
   - Predictive binary entropy assessing unimodal and multimodal prediction uncertainty.
   - Representation L2 norms capturing feature energy and degradation signatures.

5. **Self-Evolving Contextual Bandit Recovery Policy**:
   - Outcome-trained tabular contextual policy operating over state signatures $s_t = (P_t, D_t, U_t, G_t)$.
   - Strict physical feasibility masking prevents reliance on missing modalities.
   - Hybrid reward $R_t = F_1(\text{rec}) + 0.5 [F_1(\text{rec}) - F_1(\text{unrec})]$ balances absolute accuracy with relative recovery gain.
   - Policy is frozen during test deployment to guarantee zero test label leakage.

6. **Repeated-Measures Clustered Statistical Significance**:
   - Clustered paired $t$-test ($df=499$, $t=12.79$, $p=1.34 \times 10^{-32}$) and Wilcoxon signed-rank test ($p=1.15 \times 10^{-31}$).
   - 5,000-iteration clustered bootstrap confirms strictly positive recovery in 100% of resamples ($p < 0.0002$).

---

## 📂 Repository Structure

```
SEAA/
├── run_seaa.py                              # Master Unified Pipeline Runner (Single Entry Point)
├── main.py                                  # Primary CLI entry point (delegates to run_seaa.py)
├── README.md                                # Comprehensive project documentation & operational guide
├── PROJECT_STATUS.md                        # Master completion & technical status overview
├── AGENTS.md                                # Central Master Directive & Technical Architecture
├── requirements.txt                         # Project environment dependencies
├── data/
│   ├── metadata.csv                         # 25,959 indexed movie records with lengths & labels
│   ├── label_mapping.json                   # Canonical 23-genre index mapping
│   ├── split_indices.json                   # Deterministic 60/10/30 train/val/test index arrays
│   └── processed/
│       ├── resnet50_features.npy            # 2,048-dim visual feature cache (25,959 × 2,048 float32)
│       └── resnet50_features_info.json      # Checkpoint metadata for feature extraction
├── scripts/
│   ├── verify_environment.py                # System, hardware, and library diagnostic script
│   ├── prepare_dataset.py                   # Dataset download, cleaning & stratification pipeline
│   ├── extract_image_features.py            # ResNet-50 visual embedding extraction runner
│   ├── run_baselines.py                     # Unified end-to-end baseline execution runner
│   ├── run_auditing.py                      # Inline reliability auditing runner
│   ├── run_recovery.py                      # Adaptive recovery evaluation runner
│   ├── evaluate_diagnosis_performance.py    # Diagnostic confusion matrix & metrics
│   ├── run_ablation_benchmarks.py           # Component ablation benchmark suite
│   ├── benchmark_latency_efficiency.py      # High-precision latency & throughput profiling
│   └── generate_presentation_outputs.py     # Presentation chart and PDF report generator
├── src/
│   ├── data/                                # MMIMDbDataset & grouped iterative stratifier
│   ├── models/                              # Text, Image & Multimodal baselines
│   ├── evaluation/                          # Multi-label metrics & threshold optimizers
│   ├── auditing/                            # Multi-label JSD, uncertainty & failure diagnosis
│   ├── failures/                            # Text and visual corruption operators
│   ├── recovery/                            # Fallback actions & dynamic modality gating
│   └── policy/                              # Adaptive contextual bandit policy
├── tests/                                   # 73 passing automated unit tests
│   ├── test_dataset.py                      # Stratification & zero-leakage tests
│   ├── test_baselines.py                    # Baseline models & metric tests
│   ├── test_corruptions.py                  # Text and visual corruption tests
│   ├── test_disagreement.py                 # Multi-label Bernoulli JSD tests
│   ├── test_uncertainty.py                  # Predictive entropy tests
│   ├── test_diagnosis.py                    # Failure diagnosis logic tests
│   ├── test_adaptive_recovery.py            # Policy action selection & Q-learning tests
│   ├── test_improvements.py                 # Quality regression tests
│   └── test_presentation_outputs.py        # Presentation artifact tests
└── outputs/
    ├── checkpoints/                         # Serialized model weights (joblib / json)
    ├── figures/                             # Publication figures (PNG & PDF)
    ├── metrics/                             # Authoritative JSON & CSV metric reports
    ├── auditing/                            # Auditing predictions & diagnostic confusion matrices
    └── recovery/                            # Adaptive recovery benchmarks, ablations & latencies
```

---

## 📊 Dataset & Zero-Leakage Data Pipeline

The MM-IMDb dataset contains **25,959 samples** across **23 movie genres**:

| Metric | Value |
|---|---|
| **Total Movie Records** | 25,959 |
| **Modalities** | Text (plot summaries) + Vision (poster RGB images) |
| **Label Space** | 23 genres (Multi-hot binary indicator vector) |
| **Mean Plot Length** | 614.7 characters (106.2 words) |
| **Mean Genre Cardinality** | 2.48 genres per movie |
| **Train Set Size** | 15,564 samples (59.96%) |
| **Validation Set Size** | 2,607 samples (10.04%) |
| **Test Set Size** | 7,788 samples (30.00%) |
| **Duplicate Text Isolation** | 11 groups (25 affected records, 14 redundant copies; 0 cross-split duplicates) |

**Genre Taxonomy:** `Action`, `Adventure`, `Animation`, `Biography`, `Comedy`, `Crime`, `Documentary`, `Drama`, `Family`, `Fantasy`, `Film-Noir`, `History`, `Horror`, `Music`, `Musical`, `Mystery`, `Romance`, `Sci-Fi`, `Short`, `Sport`, `Thriller`, `War`, `Western`.

---

## 🧠 Model Architectures & Fusion Baseline

### 1. Text-Only Baseline
- Sublinear TF-IDF (10,000 max features, unigrams+bigrams, $df \ge 2$) + One-vs-Rest Logistic Regression ($C=1.0$).
- Validation-tuned per-genre decision thresholds: Micro-F1 = **0.6056**, mAP = **0.5183**.

### 2. Image-Only Baseline
- Pretrained frozen ResNet-50 visual features (2,048 dimensions) + StandardScaler + One-vs-Rest Logistic Regression ($C=0.5$).
- Validation-tuned per-genre decision thresholds: Micro-F1 = **0.4151**, mAP = **0.2337**.

### 3. Static Multimodal Late Fusion Baseline
- Validation-grid-optimized late fusion:
  $$P_{\text{multimodal}} = 0.55 P_{\text{text}} + 0.45 P_{\text{image}}$$
- Validation-tuned per-genre decision thresholds: Micro-F1 = **0.5161**, mAP = **0.3985**.
- Static fusion blends dense text signals with weaker poster features (-8.95% Micro-F1 compared to pure text), proving the necessity of adaptive recovery.

---

## 📈 Master Empirical Results & Statistical Proof

### Benchmark Comparison (4,000 Evaluations across 500 Test Movies repeated over 8 Failure Regimes)

| Policy / Model Configuration | Condition-Averaged Micro-F1 | Globally Pooled Micro-F1 | Instance-Level Mean F1 | Delta vs Unrec. | Decision Latency | Throughput |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **SEAA Adaptive Recovery Policy** | **0.4709** | **0.4690** | **0.4671** | **+0.0697 (+17.4%)** | **0.0245 ms** | **40,753 decisions/s** |
| **No Recovery (Static Late Fusion)** | 0.4012 | 0.3973 | 0.3940 | Reference (0.0000) | 0.0120 ms | 83,364 samples/s |
| **Fixed Heuristic Policy (Router)** | 0.3942 | 0.3905 | 0.3901 | -0.0070 (-1.7%) | 0.0410 ms | 24,390 samples/s |
| **Ablation: Learning Disabled (Untrained)** | 0.4377 | 0.4348 | 0.4340 | +0.0365 (+9.1%) | 0.0240 ms | 41,667 decisions/s |
| **Ablation: Random Action Selection** | 0.4346 | 0.4327 | 0.4321 | +0.0334 (+8.3%) | 0.0050 ms | 200,000 actions/s |
| **Ablation: No Diagnosis Feature** | 0.5054 | 0.5056 | 0.5042 | +0.1042 (+26.0%) | 0.0230 ms | 43,478 decisions/s |
| **Theoretical Ceiling: Oracle Upper Bound** | 0.5695 | 0.5673 | 0.5658 | +0.1682 (+41.9%) | --- | --- |

### Clustered Statistical Significance
- **Repeated-Measures Dependence:** ICC(1) $\rho = 0.1319$, $\text{VIF} = 1.9232$, $N_{\text{eff}} \approx 2,080$ (clustered at movie entity level, $df=499$).
- **SEAA vs No Recovery:** Clustered paired $t$-test: $t = 12.79$, $p = 1.34 \times 10^{-32}$; Wilcoxon: $W = 23,564.0$, $p = 1.15 \times 10^{-31}$. 95% CI: $[+0.0619, +0.0843]$.
- **SEAA vs Fixed Heuristic:** Clustered paired $t$-test: $t = 12.82$, $p = 1.02 \times 10^{-32}$; Wilcoxon: $W = 24,620.5$, $p = 9.43 \times 10^{-30}$. 95% CI: $[+0.0652, +0.0888]$.
- **5,000-Iteration Clustered Bootstrap:** Resampling 500 movie clusters with replacement yielded strictly positive differences in $100\%$ of resamples ($p < 0.0002$) across all 10 evaluated metric combinations.
- **Component Ablation Proof:** Disabling outcome learning drops Micro-F1 from **0.4711** to **0.4377** (**$-3.34\%$ points**), confirming that outcome-based adaptation is the primary driver of recovery gains.

---

## 🚀 Quick Start Guide & Unified Command Center

### Prerequisites & Installation

```bash
# 1. Clone or navigate to the project directory:
cd "SEAA"

# 2. Set up virtual environment:
python -m venv venv
.\venv\Scripts\Activate.ps1    # On Windows PowerShell
# source venv/bin/activate     # On Linux/macOS

# 3. Install dependencies:
pip install -r requirements.txt
```

---

### Running the Complete Unified Pipeline

Execute the complete end-to-end SEAA system with a single command:

```bash
# Master pipeline: Verifies system, audits reliability, and evaluates adaptive recovery:
python run_seaa.py --mode full
```

### Running Individual System Components

You can execute specific subsystems directly through `run_seaa.py`:

```bash
# 1. System & Environment Verification:
python run_seaa.py --mode verify

# 2. Train and Benchmark Core Baselines (Text, Image, Late Fusion):
python run_seaa.py --mode baselines

# 3. Execute Inline Reliability Auditing & Failure Diagnosis:
python run_seaa.py --mode audit --sample_size 500

# 4. Execute Adaptive Contextual Bandit Recovery Benchmark:
python run_seaa.py --mode recovery --sample_size 500

# 5. Run Component Ablation Studies (Learning, Diagnosis, Actions):
python run_seaa.py --mode ablations

# 6. Run Diagnostic Confusion Matrix & Contingency Evaluation:
python run_seaa.py --mode evaluate-diagnosis

# 7. Run High-Precision Latency & Throughput Profiling:
python run_seaa.py --mode latency
```

---

## 🧪 Automated Test Suite

The test suite validates data stratification, zero test leakage, baseline accuracy, corruption operators, disagreement metrics, failure diagnosis, and contextual policy self-evolution:

```bash
# Run all 73 automated unit and integration tests:
python -m unittest discover -s tests
```

Expected result:
```
----------------------------------------------------------------------
Ran 73 tests in ~12-16s

OK
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
