# SEAA Phase 4 Implementation Plan: Auditing, Uncertainty, and Failure Diagnosis

**Project:** A Self-Evolving AI Auditing Algorithm for Adaptive Reliability Diagnosis and Recovery in Multimodal AI Systems (SEAA)  
**Phase:** Phase 4 — Cross-Modal Disagreement Auditing, Uncertainty Estimation, and Controlled Modality Failure Diagnosis  
**Target Date / Status:** Active Plan

---

## 1. Objectives & Scope

Phase 4 bridges the static baseline benchmarks (Phase 3) and adaptive recovery routing (Phase 5). In Phase 3, we observed that static late fusion degraded performance relative to text-only classification (Micro-F1 0.5161 vs 0.6056), highlighting the need for an auditing layer that diagnoses modal unreliability, conflict, and failure.

The core objectives of Phase 4 are:
1. **Cross-Modal Disagreement Auditing:** Quantify probability divergence (multi-label Bernoulli Jensen-Shannon divergence) and prediction set divergence (Jaccard and Hamming distance) between text and image models.
2. **Predictive Uncertainty Estimation:** Implement multi-label binary entropy, distance to validation-tuned decision thresholds, and threshold proximity proportions for text, image, and multimodal predictions.
3. **Controlled Modality Failure Simulation:** Develop reproducible, deterministic corruption pipelines for text (truncation, token masking, token deletion, empty text) and vision (Gaussian blur, Gaussian noise, missing image representation) that strictly preserve raw source data.
4. **Reliability Diagnosis Engine:** Formulate an auditable diagnostic layer that combines disagreement and uncertainty signals into binary corruption detection and multi-class failure diagnosis, with cutoffs calibrated strictly on validation data.
5. **Empirical Benchmarking & Evaluation Protocol:** Execute a comprehensive audit benchmark (`scripts/run_phase4_audit.py`) measuring classification degradation under corruptions, diagnostic performance (precision, recall, F1, confusion matrices), and execution latency.
6. **Strict Integrity Guarantee:** Zero modification to Phase 1–3 model checkpoints, test splits, or presentation assets. No commencement of Phase 5.

---

## 2. Technical Architecture & Module Specifications

### 2.1 Cross-Modal Disagreement (`src/auditing/disagreement.py`)
- **Mathematical Formulation for Multi-Label Bernoulli JSD:**
  For 23 independent binary genre probabilities $p_{T, c}, p_{I, c} \in [\epsilon, 1 - \epsilon]$ with $\epsilon = 10^{-7}$:
  $$M_c = \frac{1}{2}(p_{T, c} + p_{I, c})$$
  $$D_{JS}(B_{T, c} \parallel B_{I, c}) = \frac{1}{2} \left[ p_{T, c} \log_2 \frac{p_{T, c}}{M_c} + (1 - p_{T, c}) \log_2 \frac{1 - p_{T, c}}{1 - M_c} \right] + \frac{1}{2} \left[ p_{I, c} \log_2 \frac{p_{I, c}}{M_c} + (1 - p_{I, c}) \log_2 \frac{1 - p_{I, c}}{1 - M_c} \right]$$
  $$\overline{D}_{JS} = \frac{1}{C} \sum_{c=1}^C D_{JS}(B_{T, c} \parallel B_{I, c}) \in [0, 1]$$
  This properly accounts for the independent Bernoulli nature of multi-label sigmoid outputs, avoiding invalid categorical softmax assumptions.
- **Set Prediction Disagreement:**
  - Jaccard Distance: $d_J(S_T, S_I) = 1 - \frac{|S_T \cap S_I|}{|S_T \cup S_I|}$, with edge case $S_T = \emptyset, S_I = \emptyset \implies d_J = 0.0$ (complete agreement on no positive labels).
  - Normalized Hamming Distance: $d_H(S_T, S_I) = \frac{1}{C} \sum_{c=1}^C |\hat{y}_{T, c} - \hat{y}_{I, c}|$.
- **Per-Genre Breakdown:** Return per-genre disagreement indicators and absolute probability discrepancies $|p_{T, c} - p_{I, c}|$.
- **Validation:** Strict shape, range $[0, 1]$, and finite numeric checks.

### 2.2 Predictive Uncertainty (`src/auditing/uncertainty.py`)
- **Binary Predictive Entropy:**
  $$H(p_c) = - [p_c \log_2 p_c + (1 - p_c) \log_2 (1 - p_c)] \in [0, 1]$$
  $$\overline{H} = \frac{1}{C} \sum_{c=1}^C H(p_c)$$
  Maximized when $p_c = 0.5$ (maximum ambiguity), minimized when $p_c \in \{0, 1\}$.
- **Distance to Decision Thresholds:**
  Given validation-tuned thresholds $\tau = (\tau_1, \dots, \tau_C)$:
  - Mean Margin: $\overline{\Delta}_\tau = \frac{1}{C} \sum_{c=1}^C |p_c - \tau_c|$
  - Min Margin: $\min_c |p_c - \tau_c|$
- **Threshold Proximity Proportion:**
  $$\text{prop\_near}(\delta) = \frac{1}{C} \sum_{c=1}^C \mathbb{I}(|p_c - \tau_c| \le \delta)$$
  Default $\delta = 0.05$.
- **Applicability:** Uniform interface for Text-Only, Image-Only, and Multimodal predictions.

### 2.3 Modality Failure Simulation
- **Text Corruptions (`src/failures/text_corruptions.py`):**
  - Truncation (`truncate_text`): Truncates text by fraction (e.g., 0.25) or word count limit.
  - Token Masking (`mask_tokens`): Deterministically replaces words with `[MASK]` token given a masking rate and seed.
  - Token Deletion / Noise (`delete_tokens`): Randomly drops tokens with configurable probability and seed.
  - Empty Text (`make_empty_text`): Returns empty string `""`.
  - Non-destructive: Takes raw strings, returns corrupted copies without altering dataset storage.
- **Image Corruptions (`src/failures/image_corruptions.py`):**
  - Gaussian Blur (`apply_gaussian_blur`): Blurs PIL Image or tensor with radius $\sigma$.
  - Gaussian Noise (`apply_gaussian_noise`): Adds zero-mean Gaussian noise with std $\sigma \in (0, 1)$ to normalized or pixel arrays, clipping to valid range.
  - Missing Image (`apply_missing_image`): Returns blank zero image tensor consistent with codebase convention.
  - End-to-End Extraction: Utility to extract ResNet-50 features from corrupted images on-the-fly for audit subsets.

### 2.4 Diagnostic Engine (`src/auditing/diagnosis.py`)
- **Signal Aggregation:** Collects text entropy, image entropy, text threshold margin, image threshold margin, mean Bernoulli JSD, Jaccard distance, and missing modality flags.
- **Decision Rules & Calibration:**
  - Calibrated strictly on the validation split: determine 95th percentiles of clean validation disagreement and uncertainty to establish anomaly thresholds.
  - Binary Corruption Detection: Flag sample as anomalous if disagreement or modal uncertainty exceeds calibration thresholds, or if a modality is missing.
  - Multi-class Failure Diagnosis:
    - `CLEAN`: Normal agreement and low uncertainty.
    - `TEXT_CORRUPTED`: High text uncertainty / low text margin combined with high disagreement and stable image prediction.
    - `IMAGE_CORRUPTED`: High image uncertainty combined with high disagreement and stable text prediction.
    - `CROSS_MODAL_CONFLICT`: High disagreement despite moderate confidence in both modalities.
    - `HIGH_UNCERTAINTY`: Both modalities close to thresholds / high entropy.
    - `MISSING_MODALITY`: Missing text or missing image flag active.
- **Reporting:** Structured dictionary and DataFrame including raw numerical scores alongside discrete diagnostic labels.

### 2.5 Audit Benchmark Runner (`scripts/run_phase4_audit.py`)
- Evaluates:
  1. Clean baseline test predictions.
  2. Multiple corruption regimes (Text Truncation, Text Masking, Empty Text, Gaussian Blur, Gaussian Noise, Missing Poster, Dual Corruptions).
  3. Metric impact: $\Delta$ Micro-F1, $\Delta$ Macro-F1, $\Delta$ Hamming Loss, $\Delta$ mAP.
  4. Diagnostic accuracy: Confusion matrix, Precision, Recall, F1 for corruption detection and diagnosis.
  5. Latency profiling: Exact millisecond-level measurement distinguishing cached classifier inference from raw image corruption feature extraction.
- Saves outputs to `outputs/phase4/` (JSON, CSV, figures).

---

## 3. Implementation Steps & Validation Gates

1. **Step 1: Workspace Audit & Plan Documentation** (Completed by this plan).
2. **Step 2: Implement Disagreement Auditing (`src/auditing/disagreement.py`) & Unit Tests (`tests/test_disagreement.py`)**.
   - Gate: All disagreement unit tests pass with mathematical correctness and edge cases handled.
3. **Step 3: Implement Uncertainty Estimation (`src/auditing/uncertainty.py`) & Unit Tests (`tests/test_uncertainty.py`)**.
   - Gate: All uncertainty unit tests pass across edge probabilities and threshold boundaries.
4. **Step 4: Implement Modality Corruptions (`src/failures/text_corruptions.py`, `src/failures/image_corruptions.py`) & Unit Tests (`tests/test_corruptions.py`)**.
   - Gate: Corruptions are deterministic, reproducible, non-destructive, and validate shapes/ranges.
5. **Step 5: Implement Diagnostic Engine (`src/auditing/diagnosis.py`) & Unit Tests (`tests/test_diagnosis.py`)**.
   - Gate: Diagnostic engine produces auditable signals and calibrated labels with test passing.
6. **Step 6: Build Audit Benchmark Runner (`scripts/run_phase4_audit.py`) and Execute Benchmark**.
   - Gate: Benchmark runs end-to-end, producing verified CSV/JSON outputs in `outputs/phase4/`.
7. **Step 7: Full Test Suite Execution & Integrity Check**.
   - Gate: All tests (existing 29 + all new Phase 4 tests) pass 100%. Phase 1–3 artifact hashes intact.
8. **Step 8: Document Results (`docs/phase4/phase4_results.md`) and Stop for User Approval**.

---

## 4. Resource & Memory Safety Constraints
- Execution must run within CPU constraints and standard 16 GB RAM.
- Do not store 7,788 high-resolution poster images simultaneously in RAM; use streaming batches for raw image transformations.
- Fixed seed (42) for all stochastic corruption simulations.
- Do not touch Phase 5 recovery logic.
