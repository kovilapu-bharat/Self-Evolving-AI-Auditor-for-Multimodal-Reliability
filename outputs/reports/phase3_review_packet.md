# SEAA Phase 3: Comprehensive Review Packet

This document compiles the Phase 3 baseline experimental outputs, execution terminal logs, metrics artifacts, unit test outputs, and diagnostic findings for **A Self-Evolving AI Auditing Algorithm for Adaptive Reliability Diagnosis and Recovery in Multimodal AI Systems (SEAA)**.

---

## 📋 Table of Contents
1. [Phase 3 Execution Summary & Terminal Outputs](#1-phase-3-execution-summary--terminal-outputs)
2. [Consolidated Baseline Comparison (`baseline_comparison.csv`)](#2-consolidated-baseline-comparison)
3. [Image-Only Baseline Metrics (`image_only_metrics.json`)](#3-image-only-baseline-metrics)
4. [Multimodal Late Fusion Metrics (`multimodal_metrics.json`)](#4-multimodal-late-fusion-metrics)
5. [Unit Test Suite Execution (`python -m unittest discover tests`)](#5-unit-test-suite-execution)
6. [Phase 3 Diagnostics: Errors, Warnings & Resolutions](#6-phase-3-diagnostics-errors-warnings--resolutions)
7. [Generated Visualizations & Artifact Checklist](#7-generated-visualizations--artifact-checklist)

---

## 1. Phase 3 Execution Summary & Terminal Outputs

### A. Pre-extracted ResNet-50 Feature Caching
- **Extracted Sample Count:** 25,959 / 25,959 posters (100.0% complete)
- **Feature Dimension:** 2,048 float32 values per sample
- **Cache File:** `data/processed/resnet50_features.npy` (212.65 MB raw memory-map)
- **Status:** Complete with zero RAM leakage via chunked disk flushing.

---

### B. Text-Only Baseline Execution
**Command:** `python scripts/train_text_baseline.py`
```text
2026-10-09 12:11:23,482 [INFO] TrainTextBaseline: === Starting Text-Only Baseline Training & Evaluation ===
2026-10-09 12:11:24,511 [INFO] TrainTextBaseline: Loaded 25959 metadata records across 23 genres.
2026-10-09 12:11:24,511 [INFO] TrainTextBaseline: Split sizes: Train=15564, Val=2607, Test=7788
2026-10-09 12:11:26,945 [INFO] TextBaseline: Fitting TF-IDF Vectorizer on 15564 training texts...
2026-10-09 12:11:42,468 [INFO] TextBaseline: TF-IDF fitted in 15.52s. Vocabulary size: 10000
2026-10-09 12:11:42,468 [INFO] TextBaseline: Training One-vs-Rest Logistic Regression classifiers...
2026-10-09 12:11:59,431 [INFO] TextBaseline: Classifiers trained in 16.96s.
2026-10-09 12:11:59,431 [INFO] TextBaseline: Evaluating and tuning thresholds on 2607 validation samples...
2026-10-09 12:12:09,270 [INFO] TextBaseline: Validation Results: Default Micro-F1=0.4744 -> Tuned Micro-F1=0.6042 | Tuned Macro-F1=0.4986 | mAP=0.5213
2026-10-09 12:12:09,273 [INFO] TrainTextBaseline: Evaluating text baseline on 7788 test samples...
2026-10-09 12:12:13,452 [INFO] TrainTextBaseline: === Test Evaluation Results ===
2026-10-09 12:12:13,452 [INFO] TrainTextBaseline: Micro-F1: 0.6056
2026-10-09 12:12:13,452 [INFO] TrainTextBaseline: Macro-F1: 0.4955
2026-10-09 12:12:13,452 [INFO] TrainTextBaseline: Mean Average Precision (mAP): 0.5183
2026-10-09 12:12:13,452 [INFO] TrainTextBaseline: Hamming Loss: 0.0929
2026-10-09 12:12:13,454 [INFO] TrainTextBaseline: Test Inference Time: 3.66s (2126.0 samples/s)
2026-10-09 12:12:13,460 [INFO] TrainTextBaseline: Saved test metrics JSON to: outputs/metrics/text_only_metrics.json
2026-10-09 12:12:13,486 [INFO] TrainTextBaseline: Saved per-genre metrics CSV to: outputs/metrics/text_only_per_genre.csv
2026-10-09 12:12:17,458 [INFO] TrainTextBaseline: Saved test predictions CSV to: outputs/predictions/text_only_test_predictions.csv
2026-10-09 12:12:18,313 [INFO] TextBaseline: Text baseline checkpoint saved to: outputs/checkpoints/text_baseline.joblib
2026-10-09 12:12:20,416 [INFO] TrainTextBaseline: Saved validation & test probability arrays for fusion to: outputs/predictions/text_only_val_predictions.npz
2026-10-09 12:12:20,418 [INFO] TrainTextBaseline: === Text-Only Baseline Finished Successfully ===
```

---

### C. Image-Only Baseline Execution
**Command:** `python scripts/train_image_baseline.py`
```text
2026-10-19 11:55:43,913 [INFO] TrainImageBaseline: === Starting Image-Only Baseline Training & Evaluation ===
2026-10-19 11:55:44,819 [INFO] TrainImageBaseline: Loading ResNet50 features from: data/processed/resnet50_features.npy...
2026-10-19 11:55:46,696 [INFO] TrainImageBaseline: Loaded features: Train=(15564, 2048), Val=(2607, 2048), Test=(7788, 2048)
2026-10-19 11:55:46,696 [INFO] ImageBaseline: Fitting image classifier on 15564 training samples (dim=2048)...
2026-10-19 11:56:43,601 [INFO] ImageBaseline: Image classifier trained in 56.90s.
2026-10-19 11:56:43,617 [INFO] ImageBaseline: Evaluating and tuning thresholds on 2607 validation samples...
2026-10-19 11:56:47,510 [INFO] ImageBaseline: Validation Results: Default Micro-F1=0.3969 -> Tuned Micro-F1=0.4237 | Tuned Macro-F1=0.2954 | mAP=0.2368
2026-10-19 11:56:47,538 [INFO] TrainImageBaseline: Evaluating image baseline on 7788 test samples...
2026-10-19 11:56:49,437 [INFO] TrainImageBaseline: === Image-Only Test Evaluation Results ===
2026-10-19 11:56:49,437 [INFO] TrainImageBaseline: Micro-F1: 0.4151
2026-10-19 11:56:49,444 [INFO] TrainImageBaseline: Macro-F1: 0.2847
2026-10-19 11:56:49,444 [INFO] TrainImageBaseline: Mean Average Precision (mAP): 0.2337
2026-10-19 11:56:49,444 [INFO] TrainImageBaseline: Hamming Loss: 0.1642
2026-10-19 11:56:49,444 [INFO] TrainImageBaseline: Test Inference Time: 1.57s (4972.9 samples/s)
2026-10-19 11:56:49,453 [INFO] TrainImageBaseline: Saved test metrics JSON to: outputs/metrics/image_only_metrics.json
2026-10-19 11:56:49,489 [INFO] TrainImageBaseline: Saved per-genre metrics CSV to: outputs/metrics/image_only_per_genre.csv
2026-10-19 11:56:51,363 [INFO] TrainImageBaseline: Saved test predictions CSV to: outputs/predictions/image_only_test_predictions.csv
2026-10-19 11:56:51,389 [INFO] ImageBaseline: Image baseline checkpoint saved to: outputs/checkpoints/image_baseline.joblib
2026-10-19 11:56:52,463 [INFO] TrainImageBaseline: Saved validation & test probability arrays for fusion to: outputs/predictions/image_only_val_predictions.npz
2026-10-19 11:56:52,468 [INFO] TrainImageBaseline: === Image-Only Baseline Finished Successfully ===
```

---

### D. Multimodal Late Fusion Baseline Execution
**Command:** `python scripts/train_multimodal_baseline.py`
```text
2026-10-19 11:58:13,878 [INFO] TrainMultimodalBaseline: === Starting Multimodal Baseline Training & Evaluation ===
2026-10-19 11:58:14,765 [INFO] TrainMultimodalBaseline: Loaded validation probabilities: Text=(2607, 23), Image=(2607, 23)
2026-10-19 11:58:14,765 [INFO] TrainMultimodalBaseline: Loaded test probabilities: Text=(7788, 23), Image=(7788, 23)
2026-10-19 11:58:14,765 [INFO] MultimodalBaseline: Tuning fusion weight alpha across 21 candidates on validation set...
2026-10-19 11:58:14,919 [INFO] MultimodalBaseline: Optimal validation fusion weight selected: alpha=0.55 (Val Micro-F1=0.4760)
2026-10-19 11:58:16,501 [INFO] MultimodalBaseline: Validation Results: Default Micro-F1=0.4760 -> Tuned Micro-F1=0.5216 | Tuned Macro-F1=0.4050 | mAP=0.4002
2026-10-19 11:58:16,501 [INFO] TrainMultimodalBaseline: Evaluating multimodal model on test split using validation-selected alpha and thresholds...
2026-10-19 11:58:16,746 [INFO] TrainMultimodalBaseline: === Multimodal Test Evaluation Results ===
2026-10-19 11:58:16,746 [INFO] TrainMultimodalBaseline: Selected Alpha (Text Weight): 0.55
2026-10-19 11:58:16,746 [INFO] TrainMultimodalBaseline: Micro-F1: 0.5161
2026-10-19 11:58:16,746 [INFO] TrainMultimodalBaseline: Macro-F1: 0.4003
2026-10-19 11:58:16,746 [INFO] TrainMultimodalBaseline: Mean Average Precision (mAP): 0.3985
2026-10-19 11:58:16,746 [INFO] TrainMultimodalBaseline: Hamming Loss: 0.1307
2026-10-19 11:58:16,749 [INFO] TrainMultimodalBaseline: Saved multimodal metrics JSON to: outputs/metrics/multimodal_metrics.json
2026-10-19 11:58:16,754 [INFO] TrainMultimodalBaseline: Saved multimodal per-genre CSV to: outputs/metrics/multimodal_per_genre.csv
2026-10-19 11:58:18,244 [INFO] TrainMultimodalBaseline: Saved multimodal test predictions CSV to: outputs/predictions/multimodal_test_predictions.csv
2026-10-19 11:58:18,247 [INFO] MultimodalBaseline: Multimodal baseline configuration saved to: outputs/checkpoints/multimodal_baseline.json
2026-10-19 11:58:18,271 [INFO] TrainMultimodalBaseline: Saved consolidated baseline comparison to: outputs/metrics/baseline_comparison.csv
2026-10-19 11:58:18,287 [INFO] TrainMultimodalBaseline: Generating publication-grade figures...
2026-10-19 11:58:23,481 [INFO] TrainMultimodalBaseline: Saved figures to: outputs/figures
2026-10-19 11:58:23,481 [INFO] TrainMultimodalBaseline: === Multimodal Baseline Finished Successfully ===
```

---

## 2. Consolidated Baseline Comparison

**File Location:** `outputs/metrics/baseline_comparison.csv`

### Raw CSV Content (`outputs/metrics/baseline_comparison.csv`)
```csv
model,modality,micro_f1,macro_f1,micro_precision,micro_recall,mean_average_precision,hamming_loss,inference_time_sec,samples_per_sec
Text-Only (TF-IDF + LR),Text,0.6056,0.4955,0.5594,0.6602,0.5183,0.0929,1.5415,5052.2
Image-Only (ResNet50 + LR),Image (Cached Feats),0.4151,0.2847,0.3373,0.5396,0.2337,0.1642,1.9394,4015.7
Multimodal (Validation-Tuned Weighted Late Fusion),Multimodal (Cached E2E),0.5161,0.4003,0.4301,0.6451,0.3985,0.1307,3.602,2162.2
```

### Formatted Performance Summary Table

| Model Architecture | Modality | Micro-F1 | Macro-F1 | Precision | Recall | Mean AP (mAP) | Hamming Loss | Latency (s) | Throughput (samples/s) |
|---|---|---|---|---|---|---|---|---|---|
| **Text-Only (TF-IDF + LR)** | Text | **0.6056** | **0.4955** | **0.5594** | **0.6602** | **0.5183** | **0.0929** | 1.54s | 5,052.2 |
| **Image-Only (ResNet50 + LR)** | Image (Cached Feats) | 0.4151 | 0.2847 | 0.3373 | 0.5396 | 0.2337 | 0.1642 | 1.94s | 4,015.7 |
| **Multimodal (Validation-Tuned Weighted Late Fusion)** | Multimodal (Cached E2E) | 0.5161 | 0.4003 | 0.4301 | 0.6451 | 0.3985 | 0.1307 | 3.60s | 2,162.2 |

> **Timing Methodology & Scopes Note:** Latencies measured via `time.perf_counter()` over 1 warm-up repetition followed by 5 timed repetitions across all 7,788 test samples.
> - **Text-Only:** TF-IDF transformation (10,000 features, unigrams & bigrams) + OneVsRest Logistic Regression `predict_proba` over 23 genres.
> - **Image-Only (Cached Feats):** StandardScaler transformation + OneVsRest Logistic Regression `predict_proba` over 23 genres using precomputed 2,048-dim ResNet50 embeddings (does NOT include raw image decoding or ResNet50 feature extraction).
> - **Multimodal (Cached E2E):** Full pipeline consuming raw text and precomputed visual embeddings (text TF-IDF vectorization + image scaling + dual classifier inference + late fusion weighting $\alpha=0.55$ + thresholding). Probability fusion and thresholding alone requires only **0.000962 seconds** (0.0010s rounded, 8,091,428.6 samples/s).
> - **Raw Image Feature Extraction Reference:** ~794.7 seconds estimated for 7,788 images at 9.8 samples/s. Historical estimate from the Phase 2 log (`outputs/logs/image_feature_extraction.log`); not remeasured in this audit and not directly comparable to cached classifier latency.
> - **Timing Discrepancy Reconciliation:** Earlier draft reports included timing values from an unrepeated single run during training (e.g. Text: 3.66s, Image: 1.57s, Multimodal: 0.00s) and subsequent intermediate benchmark passes (e.g., 1.16s / 1.66s / 2.92s). All timing artifacts across the repository are now synchronized to the standardized benchmark using `time.perf_counter()` (1 warm-up repetition followed by 5 timed repetitions).
> - **Exact Duplicate Reconciliation:** Across 25,959 total records (25,945 distinct texts), exactly 11 duplicate groups affecting 25 records (14 redundant copies) exist. Duplicate grouping strictly placed all copies within their respective split: 5 groups in Train (13 records, 8 redundant), 3 groups in Val (6 records, 3 redundant), and 3 groups in Test (6 records, 3 redundant). Exactly 0 duplicate text strings cross splits.

---

## 3. Image-Only Baseline Metrics

**File Location:** `outputs/metrics/image_only_metrics.json`

### Overview & Summary
* **Model:** One-vs-Rest Logistic Regression ($C=0.5$, `StandardScaler`, frozen ResNet-50 2048-dim features)
* **Training Time:** 56.90s across 15,564 samples
* **Validation Tuning:** Micro-F1 improved from 0.3969 (default threshold 0.5) to **0.4237** via per-genre threshold tuning.
* **Test Performance:** Micro-F1: **0.4151** | Macro-F1: **0.2847** | mAP: **0.2337** | Hamming Loss: **0.1642**

### Per-Genre Metrics Sample (Top & Bottom Performing Genres)
```json
{
  "Animation": {
    "precision": 0.4578,
    "recall": 0.3813,
    "f1": 0.4161,
    "average_precision": 0.454,
    "support": 299,
    "threshold": 0.48
  },
  "Comedy": {
    "precision": 0.5702,
    "recall": 0.5378,
    "f1": 0.5535,
    "average_precision": 0.5898,
    "support": 2577,
    "threshold": 0.44
  },
  "Horror": {
    "precision": 0.4004,
    "recall": 0.5172,
    "f1": 0.4516,
    "average_precision": 0.3683,
    "support": 812,
    "threshold": 0.22
  },
  "Film-Noir": {
    "precision": 0.0504,
    "recall": 0.2353,
    "f1": 0.0829,
    "average_precision": 0.0461,
    "support": 102,
    "threshold": 0.1
  }
}
```

---

## 4. Multimodal Baseline Metrics (Validation-Tuned Weighted Late Fusion)

**File Location:** `outputs/metrics/multimodal_metrics.json`

### Overview & Summary
* **Fusion Architecture:** Validation-Tuned Weighted Late Fusion ($P_{\text{mm}} = \alpha P_{\text{text}} + (1 - \alpha) P_{\text{image}}$)
* **Optimal Parameter Selection (Strictly on Validation Set):**
  - **$\alpha$ (Text Weight):** **0.55** (fixed scalar for all examples)
  - **$1 - \alpha$ (Image Weight):** **0.45**
  - **Validation Gain:** Micro-F1 improved from 0.4760 (default 0.5 threshold) to **0.5216** with tuned thresholds.
* **Test Performance:**
  - **Micro-F1:** **0.5161**
  - **Macro-F1:** **0.4003**
  - **Mean Average Precision (mAP):** **0.3985**
  - **Hamming Loss:** **0.1307**
  - **Micro-Precision:** 0.4301 | **Micro-Recall:** 0.6451
* **Empirical Modality Comparison:**
  - Multimodal late fusion improves over Image-Only by an observed **+10.10 percentage points** in Micro-F1 (0.5161 vs. 0.4151) and **+16.48 percentage points** in mAP (0.3985 vs. 0.2337).
  - Multimodal late fusion does **not** exceed Text-Only, trailing by an observed **-8.95 percentage points** in Micro-F1 (0.5161 vs. 0.6056) and **-11.98 percentage points** in mAP (0.3985 vs. 0.5183).
  - **Statistical Testing Note:** Significance tests (e.g., bootstrap or McNemar's tests) were not performed; all differences are reported as observed empirical differences on the fixed 7,788 test split.
  - **Phase 3 Baseline Scope:** This fixed-parameter late fusion baseline serves as a static reference point and does not represent an adaptive recovery policy, self-evolving loop, or reliability-directed adaptation.

### Per-Genre Test Breakdown (Sample from `outputs/metrics/multimodal_per_genre.csv`)
| Genre | Precision | Recall | F1 Score | Average Precision | Support | Tuned Threshold |
|---|---|---|---|---|---|---|
| **Action** | 0.5050 | 0.6692 | 0.5756 | 0.5606 | 1,064 | 0.20 |
| **Animation** | 0.4552 | 0.4415 | 0.4482 | 0.4687 | 299 | 0.22 |
| **Comedy** | 0.5846 | 0.7610 | 0.6612 | 0.6868 | 2,577 | 0.38 |
| **Drama** | 0.6896 | 0.7712 | 0.7281 | 0.7709 | 4,191 | 0.48 |
| **Horror** | 0.5732 | 0.6872 | 0.6250 | 0.6481 | 812 | 0.24 |
| **Romance** | 0.4727 | 0.6712 | 0.5547 | 0.5367 | 1,609 | 0.26 |
| **Sci-Fi** | 0.5401 | 0.6801 | 0.6020 | 0.6288 | 597 | 0.18 |
| **Western** | 0.6974 | 0.5024 | 0.5840 | 0.6559 | 211 | 0.16 |

---

## 5. Unit Test Suite Execution

**Command:** `python -m unittest discover tests`

### Terminal Output
```text
....C:\Users\bharat kumar\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\sklearn\multiclass.py:90: UserWarning: Label not 2 is present in all training examples.
  warnings.warn(
C:\Users\bharat kumar\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\sklearn\multiclass.py:90: UserWarning: Label not 3 is present in all training examples.
  warnings.warn(
[... additional sklearn single-class synthetic mock warnings ...]
......Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
............
----------------------------------------------------------------------
Ran 22 tests in 8.583s

OK
```

### Breakdown of Verified Checks (22/22 Passed)
* `test_01_fixed_splits_integrity`: Confirmed exact 15,564 / 2,607 / 7,788 split partitions.
* `test_02_label_order_and_dimensions`: Confirmed 23 canonical genre names in invariant alphabetical order.
* `test_03_controlled_metric_computation`: Verified metric calculations on controlled synthetic test cases.
* `test_04_threshold_optimization_improves_or_matches_f1`: Verified that threshold tuning strictly optimizes F1 on validation splits.
* `test_05_text_baseline_pipeline_mock`: Validated text model fitting, threshold tuning, and prediction bounds.
* `test_06_image_baseline_pipeline_mock`: Validated image model fitting, scaler, and prediction bounds.
* `test_07_multimodal_late_fusion_validation_selection`: Verified that $\alpha$ weighting and decision thresholds are chosen strictly on validation splits without test leakage.
* `test_08_saved_predictions_structure_and_ids`: Verified that saved test prediction files have exactly 7,788 records matching test split IDs.
* `test_09_checkpoints_load_and_reproduce`: Confirmed all 3 model checkpoints load cleanly and contain expected parameters ($\alpha=0.55$, thresholds, scalers).
* `test_10_timing_artifacts_validity`: Confirmed `inference_timing.csv` outputs exist with valid, non-zero measured latencies.
* `test_01` to `test_12` in `test_dataset.py`: Comprehensive checks on movie ID uniqueness, multi-hot encoding invertibility, image shape consistency, and zero text overlap across splits.

---

## 6. Phase 3 Diagnostics: Errors, Warnings & Resolutions

| Issue Category | Description | Root Cause | Resolution Implemented |
|---|---|---|---|
| **Error (Resolved)** | `ValueError: Cannot load file containing pickled data when allow_pickle=False` | `resnet50_features.npy` was created using raw disk `np.memmap` (mode `w+`), which writes raw floats without the standard `.npy` header. Calling `np.load` triggered numpy's pickle fallback. | Updated `src/models/image_feature_extractor.py` and `scripts/train_image_baseline.py` to directly load using `np.memmap(features_path, dtype=np.float32, mode="r", shape=(total_samples, feature_dim))`. |
| **Error (Resolved)** | `NameError: name 'time' is not defined` in `train_multimodal_baseline.py` line 172 | The standard library module `import time` was omitted from the top of `scripts/train_multimodal_baseline.py`. | Added `import time` at line 10. |
| **Warning (Benign)** | `UserWarning: Label not X is present in all training examples` during unit testing | `tests/test_baselines.py` constructs a small 10-sample synthetic dataset for unit testing the model interfaces. Rarest labels have zero positive instances in that micro mock. | Expected behavior for synthetic mock unit tests. |
| **Warning (Benign)** | `Warning: You are sending unauthenticated requests to the HF Hub` | Automated Hugging Face dataset cache validation query performed without an API key token. | Download and verification complete successfully via public read access. |

---

## 7. Generated Visualizations & Artifact Checklist

### Generated Figure Artifacts
- **Model Comparison Plot:** `outputs/figures/baseline_model_comparison.png` (High-resolution comparison of Micro-F1, Macro-F1, and mAP across all 3 baseline architectures).
- **Per-Genre F1 Comparison:** `outputs/figures/per_genre_f1_comparison.png` (Grouped bar chart showing individual F1 scores across all 23 genres for Text, Image, and Multimodal).

### Artifact Checklist
- [x] Consolidated metrics: `outputs/metrics/baseline_comparison.csv`
- [x] Text-only metrics & predictions: `outputs/metrics/text_only_metrics.json`, `outputs/predictions/text_only_test_predictions.csv`
- [x] Image-only metrics & predictions: `outputs/metrics/image_only_metrics.json`, `outputs/predictions/image_only_test_predictions.csv`
- [x] Multimodal metrics & predictions: `outputs/metrics/multimodal_metrics.json`, `outputs/predictions/multimodal_test_predictions.csv`
- [x] Checkpoints: `outputs/checkpoints/text_baseline.joblib`, `outputs/checkpoints/image_baseline.joblib`, `outputs/checkpoints/multimodal_baseline.json`
- [x] Cached features: `data/processed/resnet50_features.npy` (212 MB)
- [x] Experimental report: `outputs/reports/phase3_baseline_report.md`
- [x] Full review packet: `outputs/reports/phase3_review_packet.md`
