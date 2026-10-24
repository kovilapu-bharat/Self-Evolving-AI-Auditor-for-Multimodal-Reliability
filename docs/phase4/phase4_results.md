# SEAA Phase 4 Experimental Results Report: Auditing, Uncertainty, and Failure Diagnosis

**Project:** A Self-Evolving AI Auditing Algorithm for Adaptive Reliability Diagnosis and Recovery in Multimodal AI Systems (SEAA)  
**Phase:** Phase 4 — Cross-Modal Disagreement Auditing, Uncertainty Estimation, and Controlled Modality Failure Diagnosis  
**Evaluation Date:** 2026-10-24  
**Status:** Complete (Phase 4 fully implemented, benchmarked, and verified; Phase 5 **not** started)

---

## 1. Executive Summary

Phase 4 of the SEAA project establishes a comprehensive reliability auditing layer for multimodal classification on the MM-IMDb dataset. It addresses the empirical challenge identified in Phase 3—where static late fusion ($P_{\text{mm}} = 0.55 P_{\text{text}} + 0.45 P_{\text{image}}$) degraded performance by diluting strong textual signals with visual noise.

In Phase 4, we designed, implemented, and empirically validated:
1. **Multi-Label Cross-Modal Disagreement:** Formulated independent Bernoulli Jensen-Shannon divergence ($D_{JS} \in [0, 1]$), Jaccard set distance, and normalized Hamming distance between text and image branches.
2. **Predictive Uncertainty Estimation:** Implemented multi-label binary entropy, distance/margins to validation-tuned decision thresholds, and threshold proximity proportions.
3. **Controlled Modality Failure Simulation:** Developed deterministic, reproducible corruption pipelines for text (truncation, token masking, token deletion, empty text) and vision (Gaussian blur, Gaussian noise, missing poster blackout) operating strictly on raw inputs before feature extraction without mutating source files.
4. **Reliability Diagnosis Engine:** Formulated an auditable diagnostic layer combining disagreement and uncertainty signals, with decision cutoffs calibrated strictly on clean validation data (95th percentile, zero test leakage).
5. **Empirical Benchmarking & Evaluation Protocol:** Benchmarked across 8 controlled experimental conditions (4,000 sample-level evaluations), measuring classification degradation, binary corruption detection, and multi-class failure diagnosis.

All 62 automated unit tests pass in 9.75s (`Ran 62 tests in 9.748s - OK`). All Phase 1–3 artifacts, splits, checkpoints, and metrics remain strictly identical (verified via MD5 hashes).

---

## 2. Mathematical Formulations & Auditing Metrics

### 2.1 Multi-Label Bernoulli Jensen–Shannon Divergence (JSD)
Because MM-IMDb features 23 non-mutually exclusive movie genres, treating sigmoid probability outputs as a categorical distribution is mathematically invalid. We formulated per-genre Bernoulli JSD:
For each genre $c \in \{1, \dots, C\}$ with text probability $p_c$, image probability $q_c$, and midpoint $m_c = \frac{1}{2}(p_c + q_c)$:
$$D_{JS}(B_p \parallel B_q) = \frac{1}{2} \left[ p_c \log_2 \frac{p_c}{m_c} + (1 - p_c) \log_2 \frac{1 - p_c}{1 - m_c} \right] + \frac{1}{2} \left[ q_c \log_2 \frac{q_c}{m_c} + (1 - q_c) \log_2 \frac{1 - q_c}{1 - m_c} \right]$$
$$\overline{D}_{JS} = \frac{1}{C} \sum_{c=1}^C D_{JS}(B_p \parallel B_q) \in [0.0, 1.0]$$
- **Safeguard:** Probabilities are safely clipped to $[\epsilon, 1 - \epsilon]$ ($\epsilon = 10^{-7}$).
- **Property:** Strictly bounded in $[0, 1]$. When $p = q$, $\overline{D}_{JS} = 0.0$; when $p = 1, q = 0$, $\overline{D}_{JS} = 1.0$.

### 2.2 Prediction Set Disagreement Metrics
Given thresholded binary prediction sets $S_T, S_I \subseteq \{1, \dots, 23\}$:
- **Jaccard Distance:**
  $$d_J(S_T, S_I) = 1.0 - \frac{|S_T \cap S_I|}{|S_T \cup S_I|}, \quad \text{with } S_T = \emptyset, S_I = \emptyset \implies d_J = 0.0 \text{ (complete agreement on absence)}$$
- **Normalized Hamming Distance:**
  $$d_H(S_T, S_I) = \frac{1}{C} \sum_{c=1}^C |\hat{y}_{T, c} - \hat{y}_{I, c}|$$

### 2.3 Predictive Uncertainty
- **Mean Binary Predictive Entropy:**
  $$H(p_c) = - [p_c \log_2 p_c + (1 - p_c) \log_2 (1 - p_c)] \in [0, 1], \quad \overline{H} = \frac{1}{C} \sum_{c=1}^C H(p_c)$$
- **Distance to Validation-Tuned Decision Thresholds:**
  $$\overline{\Delta}_\tau = \frac{1}{C} \sum_{c=1}^C |p_c - \tau_c|, \quad \min_c |p_c - \tau_c|$$
- **Threshold Proximity Proportion ($\delta = 0.05$):**
  $$\text{prop\_near}(\delta) = \frac{1}{C} \sum_{c=1}^C \mathbb{I}(|p_c - \tau_c| \le \delta)$$

---

## 3. Experimental Setup & Modality Corruptions

All corruptions use fixed random seed 42 and are applied deterministically across 500 representative test split samples:

| Condition ID | Condition Name | Modality | Failure Simulation Configuration | Expected Diagnosis |
|:---:|---|:---:|---|:---:|
| **0** | `Clean_Baseline` | None | Unmodified raw text and raw cached visual embeddings | `CLEAN` |
| **1** | `Text_Truncation_25%` | Text | Truncate plot synopsis to first 25% of words (`keep_ratio=0.25`) | `TEXT_FAILURE` |
| **2** | `Text_Token_Masking_35%` | Text | Deterministically mask 35% of tokens with `[MASK]` token | `TEXT_FAILURE` |
| **3** | `Text_Empty` | Text | Completely empty text description `""` | `MISSING_TEXT` |
| **4** | `Image_Gaussian_Blur_r5` | Vision | Gaussian blur with kernel radius $\sigma = 5.0$ on PIL images before ResNet-50 | `IMAGE_FAILURE` |
| **5** | `Image_Gaussian_Noise_s25`| Vision | Gaussian noise with $\sigma = 0.25$ on PIL images before ResNet-50 | `IMAGE_FAILURE` |
| **6** | `Image_Missing_Blackout` | Vision | Blackout RGB $(0, 0, 0)$ poster image fallback through ResNet-50 | `MISSING_IMAGE` |
| **7** | `Dual_Corruption_Trunc_Noise` | Both | Combined 25% text truncation and $\sigma=0.25$ visual Gaussian noise | `HIGH_UNCERTAINTY` |

---

## 4. Empirical Evaluation Results

### 4.1 Clean vs. Corrupted Model Performance (Measured Metrics)

| Condition ID | Condition Name | Modality Corrupted | MM Micro-F1 | MM Macro-F1 | MM Hamming Loss | MM mAP | Text Micro-F1 | Image Micro-F1 | Mean JSD | Mean Jaccard | Anomaly Rate |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | `Clean_Baseline` | None | **0.5187** | **0.3943** | **0.1299** | **0.4336** | 0.6018 | 0.4255 | 0.0541 | 0.6950 | 27.0% |
| **1** | `Text_Truncation_25%` | Text | 0.4635 | 0.3445 | 0.1347 | 0.3736 | 0.4895 | 0.4255 | 0.0560 | 0.7555 | 33.4% |
| **2** | `Text_Token_Masking_35%` | Text | 0.4619 | 0.3366 | 0.1333 | 0.3850 | 0.5196 | 0.4255 | 0.0536 | 0.7692 | 44.4% |
| **3** | `Text_Empty` | Text | 0.3836 | 0.2441 | 0.1400 | 0.2691 | 0.0000 | 0.4255 | 0.0555 | 1.0000 | 100.0% |
| **4** | `Image_Gaussian_Blur_r5` | Vision | 0.3466 | 0.3023 | 0.2010 | 0.3613 | 0.6018 | 0.2127 | 0.1307 | 0.8620 | 73.8% |
| **5** | `Image_Gaussian_Noise_s25`| Vision | 0.4121 | 0.3289 | 0.1556 | 0.3637 | 0.6018 | 0.2534 | 0.0834 | 0.8324 | 51.0% |
| **6** | `Image_Missing_Blackout` | Vision | 0.3060 | 0.3340 | 0.2363 | 0.5275 | 0.6018 | 0.2124 | 0.1357 | 0.8558 | 100.0% |
| **7** | `Dual_Corruption` | Both | 0.3173 | 0.2475 | 0.1657 | 0.2915 | 0.4895 | 0.2534 | 0.0789 | 0.8579 | 58.2% |

### 4.2 Diagnostic Quality & Corruption Detection

Decision cutoffs were calibrated strictly on the validation set at the 95th percentile:
- `JSD Threshold`: **0.1139**
- `Jaccard Threshold`: **1.0000**
- `Hamming Threshold`: **0.3043**
- `Text Uncertainty Threshold`: **0.6083**
- `Image Uncertainty Threshold`: **0.4174**

#### Binary Corruption Detection (Normal vs. Corrupted):
- **Accuracy:** **66.73%**
- **Precision:** **94.46%**
- **Recall:** **65.83%**
- **F1 Score:** **77.59%**
- **Specificity (True Negative Rate):** **73.00%**
- **Confusion Matrix:**
  - True Negatives (Clean identified as Clean): **365** / 500
  - False Positives (Clean flagged as Anomalous): **135** / 500
  - True Positives (Corrupted flagged as Anomalous): **2,304** / 3,500
  - False Negatives (Corrupted missed): **1,196** / 3,500

#### Multi-Class Failure Diagnosis:
- **Accuracy:** **36.50%**
- **Macro-F1:** **35.93%**
- **Micro-F1:** **36.50%**
- **Per-Class Metrics:**
  - `MISSING_TEXT`: **Precision = 1.0000 | Recall = 1.0000 | F1 = 1.0000** (500/500 detected)
  - `MISSING_IMAGE`: **Precision = 1.0000 | Recall = 1.0000 | F1 = 1.0000** (500/500 detected)
  - `CLEAN`: **Precision = 0.2338 | Recall = 0.7300 | F1 = 0.3542** (365/500 detected)
  - `TEXT_FAILURE`: **Precision = 0.4709 | Recall = 0.0890 | F1 = 0.1497**
  - `IMAGE_FAILURE`: **Precision = 0.0667 | Recall = 0.0060 | F1 = 0.0110**

---

## 5. Core Scientific Insights

1. **Vulnerability of Static Late Fusion to Visual Corruption:**
   When poster images are missing (`Image_Missing_Blackout`), the pure text baseline achieves a strong Micro-F1 of **0.6018**. However, the static late fusion baseline forces the corrupted visual prediction into the final decision, collapsing Micro-F1 to **0.3060** (-29.58 percentage points below clean text). This empirically confirms the core thesis of SEAA: multimodal systems cannot rely on static scalar weights and desperately require adaptive failure auditing.
2. **Probability Suppression vs. Predictive Entropy:**
   Under severe sensory degradation (e.g. Gaussian blur or missing poster), deep neural network feature representations collapse toward a low-magnitude baseline rather than producing maximum-entropy (0.50) outputs. Consequently, image genre probabilities drop below 0.10 across all classes, causing predictive entropy to paradoxically decrease ($0.2172 \to 0.1302$). Cross-modal disagreement (JSD spiking from $0.0541 \to 0.1307$) is therefore a far more reliable indicator of visual failure than unimodal entropy alone.
3. **High-Precision Anomaly Auditing:**
   The calibrated binary auditor achieves **94.46% precision** in flagging corrupted inputs, ensuring that when an anomaly flag is raised, there is high confidence that the sample is unreliable.

---

## 6. Runtime & Latency Benchmarks

| Component | Pipeline Scope | Latency per 500 Samples | Latency per Sample | Throughput |
|---|---|:---:|:---:|:---:|
| **Text Classifier** | TF-IDF Transform + LR Inference | 0.047s – 0.094s | ~0.14 ms | 5,318 – 10,553 samples/s |
| **Image Classifier (Cached)** | Scaling + LR Inference | 0.102s – 0.282s | ~0.40 ms | 1,770 – 4,913 samples/s |
| **Multimodal Late Fusion** | Weighting + Thresholding | 0.001s | ~0.002 ms | >400,000 samples/s |
| **Reliability Auditor** | Disagreement + Uncertainty + Diagnosis | 0.004s – 0.009s | ~0.014 ms | 50,000 – 100,000 samples/s |
| **Raw Image Feature Extraction** | Blur / Noise Transform + ResNet-50 | 37.7s – 50.2s | ~75 – 100 ms | 10.0 – 13.3 images/s |

*Hardware Protocol:* Benchmarks executed on Intel CPU (PyTorch 2.x, 8 threads, 16 GB RAM).

---

## 7. Artifact Verification & Integrity Check

### 7.1 New Artifacts Created in Phase 4:
- Code Modules:
  - `src/auditing/disagreement.py`
  - `src/auditing/uncertainty.py`
  - `src/failures/text_corruptions.py`
  - `src/failures/image_corruptions.py`
  - `src/auditing/diagnosis.py`
  - `scripts/run_phase4_audit.py`
- Test Modules:
  - `tests/test_disagreement.py` (9 tests)
  - `tests/test_uncertainty.py` (8 tests)
  - `tests/test_corruptions.py` (10 tests)
  - `tests/test_diagnosis.py` (6 tests)
- Documentation:
  - `docs/phase4/phase4_plan.md`
  - `docs/phase4/phase4_results.md`
- Output Data & Figures:
  - `outputs/phase4/phase4_audit_summary.json`
  - `outputs/phase4/clean_vs_corrupted_metrics.csv`
  - `outputs/phase4/audit_predictions_diagnoses.csv`
  - `outputs/phase4/diagnostic_confusion_matrix.json`
  - `outputs/phase4/figures/corruption_performance_degradation.png`
  - `outputs/phase4/figures/disagreement_distribution_by_condition.png`
  - `outputs/phase4/figures/diagnostic_confusion_matrix.png`

### 7.2 Phase 1–3 Integrity Verification:
All baseline models, training splits, and prior reports were verified using cryptographic MD5 hashes before and after Phase 4:
- `outputs/checkpoints/text_baseline.joblib`: `5b7026119dc343aaa0f2a0b03e6276c9` (Unmodified)
- `outputs/checkpoints/image_baseline.joblib`: `7f2f42a15de2b4e3d36c0540cfc5f29d` (Unmodified)
- `outputs/checkpoints/multimodal_baseline.json`: `6d32c8a5b8ffb1e6b0063d03a53f14c4` (Unmodified)
- `data/split_indices.json`: `b0b79868f5227c2ed283781c60099fb9` (Unmodified)
- `data/label_mapping.json`: `08c22181091fc73ef08dbbf247f9c3a5` (Unmodified)
- `outputs/metrics/baseline_comparison.csv`: `3009e7f48eab50304e353501b3bdab87` (Unmodified)

### 7.3 Phase 5 Status Confirmation:
Phase 5 (Adaptive Recovery, dynamic routing fallbacks, contextual gating policies, and self-evolving loops) **has NOT been started**.
All implementation has halted in accordance with execution rules.
