# SEAA Phase 3: Baseline Models Experimental Report

**Project:** A Self-Evolving AI Auditing Algorithm for Adaptive Reliability Diagnosis and Recovery in Multimodal AI Systems (SEAA)

## 1. Executive Overview
In Phase 3 of the SEAA research pipeline, three foundational baseline architectures were implemented, trained, and evaluated on the MM-IMDb dataset under strict zero-leakage protocols:
1. **Text-Only Baseline:** Sublinear TF-IDF (10,000 features, unigrams & bigrams) with One-vs-Rest Logistic Regression.
2. **Image-Only Baseline:** Pretrained frozen ResNet50 (2048-dim global pool) with One-vs-Rest Logistic Regression.
3. **Multimodal Baseline:** Validation-Tuned Weighted Late Fusion ($P_{\text{mm}} = \alpha P_{\text{text}} + (1 - \alpha) P_{\text{image}}$), with fixed scalar fusion weight $\alpha = 0.55$ and decision thresholds optimized strictly on validation data.

## 2. Experimental Setup and Protocol
- **Fixed Split Used:** `data/split_indices.json` (Train: 15,564 / Val: 2,607 / Test: 7,788; Total: 25,959).
- **Class Count:** 23 canonical MM-IMDb genres in fixed alphabetical order (`data/label_mapping.json`).
- **Duplicate Text Guard:** Exactly 11 duplicate text groups affecting 25 records (14 redundant instances) strictly partitioned within single splits (5 Train [13 records], 3 Val [6 records], 3 Test [6 records]); exactly 0 cross-split duplicates.
- **Threshold Strategy:** Per-genre decision thresholds tuned on the validation split only to maximize validation F1 score; applied unchanged to test predictions.
- **Multimodal Fusion Weight (Text $\alpha$):** 0.55 (selected on validation split; Image weight = 0.45). Fixed for all test examples.
- **Leakage Guard:** Test labels were never accessed during feature fitting, model training, threshold selection, or fusion weight tuning.

## 3. Test Evaluation Summary

| Model Architecture | Modality | Micro-F1 | Macro-F1 | Mean AP (mAP) | Hamming Loss | Inference Time (s) | Throughput (samples/s) |
|---|---|---|---|---|---|---|---|
| **Text-Only (TF-IDF + LR)** | Text | **0.6056** | **0.4955** | **0.5183** | 0.0929 | 1.54s | 5052.2 |
| **Image-Only (ResNet50 + LR)** | Image (Cached Feats) | **0.4151** | **0.2847** | **0.2337** | 0.1642 | 1.94s | 4015.7 |
| **Multimodal (Validation-Tuned Weighted Late Fusion)** | Multimodal (Cached E2E) | **0.5161** | **0.4003** | **0.3985** | 0.1307 | 3.60s | 2162.2 |

> **Timing Methodology & Scopes:** All benchmarks executed via `time.perf_counter()` over 1 warm-up repetition followed by 5 timed repetitions across all 7,788 test records.
> - **Text-Only:** Includes TF-IDF vectorization and 23-genre classifier inference.
> - **Image-Only (Cached Feats):** Includes feature standardization and classifier inference on precomputed 2,048-dim ResNet50 features (excludes raw image feature extraction).
> - **Multimodal (Cached E2E):** Full pipeline consuming raw text and cached visual embeddings (vectorization + scaling + dual classifier inference + late fusion weighting + thresholding). Probability fusion and thresholding alone requires only **0.000962s** (0.0010s rounded, 8,091,428.6 samples/s).
> - **Raw Image Feature Extraction Reference:** ~794.7s estimated for 7,788 images at 9.8 samples/s. Historical estimate from the Phase 2 log (`outputs/logs/image_feature_extraction.log`); not remeasured in this audit and not directly comparable to cached classifier latency.

## 4. Key Empirical Insights
1. **Text Dominance in Genre Prediction:** Plot summaries provide high semantic density for thematic classification (e.g., Documentary, Biography, Romance), achieving a strong Text-Only Micro-F1 score of 0.6056 and mAP of 0.5183.
2. **Visual Modality Baseline:** Poster visual features alone achieve 0.4151 Micro-F1 and 0.2337 mAP, providing complementary cues for genres with strong visual conventions (e.g., Animation, Horror).
3. **Multimodal Late Fusion Behavior:** Validation-tuned weighted late fusion (optimal validation weight $\alpha = 0.55$, image weight = 0.45) achieves 0.5161 Micro-F1. While it exhibits an observed improvement over Image-Only (+10.10 percentage points in Micro-F1), it does not exceed the Text-Only baseline (-8.95 percentage points in Micro-F1). No statistical significance claims are asserted as hypothesis testing was not conducted. This empirical finding demonstrates that static linear late fusion dilutes dense textual features with weaker visual signals, highlighting the need for deeper cross-modal attention or dynamic reliability-aware routing in future phases.
4. **Phase Scope Limitation:** Phase 3 baselines alone establish foundational benchmark performance and do not demonstrate that SEAA's future failure diagnosis, recovery, or self-evolving learning policy works.

## 5. Artifact Verification Checklist
- [x] Consolidated metrics: `outputs/metrics/baseline_comparison.csv`
- [x] Timing artifacts: `outputs/metrics/inference_timing.json`, `outputs/metrics/inference_timing.csv`
- [x] Text-only metrics & predictions: `outputs/metrics/text_only_metrics.json`, `outputs/predictions/text_only_test_predictions.csv`
- [x] Image-only metrics & predictions: `outputs/metrics/image_only_metrics.json`, `outputs/predictions/image_only_test_predictions.csv`
- [x] Multimodal metrics & predictions: `outputs/metrics/multimodal_metrics.json`, `outputs/predictions/multimodal_test_predictions.csv`
- [x] Checkpoints: `outputs/checkpoints/text_baseline.joblib`, `outputs/checkpoints/image_baseline.joblib`, `outputs/checkpoints/multimodal_baseline.json`
- [x] Cached features: `data/processed/resnet50_features.npy`
- [x] Figures: `outputs/figures/baseline_model_comparison.png`, `outputs/figures/per_genre_f1_comparison.png`
- [x] Metric verification report: `outputs/reports/phase3_metric_verification.md`
