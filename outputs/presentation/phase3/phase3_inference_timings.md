# SEAA Phase 3: Standardized Inference Latency & Benchmarking Breakdown

**Project:** A Self-Evolving AI Auditing Algorithm for Adaptive Reliability Diagnosis and Recovery in Multimodal AI Systems (SEAA)  
**Benchmarking Protocol:** Checkpoints loaded prior to timing; 1 warm-up repetition followed by 5 timed repetitions over 7,788 test records using high-resolution `time.perf_counter()` on CPU.  
**Source Provenance:** `outputs/metrics/inference_timing.json` and `outputs/metrics/inference_timing.csv`.

| Component / Pipeline | Median Latency (s) | Min (s) | Max (s) | Std Dev (s) | Throughput (samples/s) | Precise Scope & Measurement Status |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Text Classifier Inference** | **1.5415** | 1.5356 | 1.7150 | 0.0683 | **5,052.2** | TF-IDF vectorization (10,000 features) + OneVsRest Logistic Regression `predict_proba` over 23 genres |
| **Image Classifier Cached Inference** | **1.9394** | 1.3220 | 2.0579 | 0.2982 | **4,015.7** | StandardScaler transformation + OneVsRest Logistic Regression `predict_proba` on precomputed 2,048-dim ResNet50 features |
| **Fusion & Thresholding Only** | **0.000962** | 0.000942 | 0.002194 | 0.000486 | **8,091,428.6** | Probability weighting ($0.55 P_{text} + 0.45 P_{image}$) on already computed probability arrays + 23-genre thresholding. **NOT end-to-end model inference.** |
| **End-to-End Cached Multimodal** | **3.6020** | 3.1565 | 3.9054 | 0.2812 | **2,162.2** | Complete pipeline consuming raw text strings and cached visual embeddings: Text TF-IDF + Image Scaler + Dual Classifier Inference + Late Fusion + Thresholding |
| **Raw Image ResNet50 Reference** | *794.7000\** | N/A | N/A | N/A | *9.8\** | Raw image poster decoding, resizing (224x224), RGB normalization, and ResNet50 forward pass on CPU. |

> **Critical Clarifications:**
> 1. *\*Historical Estimate Status:* The raw image feature extraction value of **794.7 seconds for 7,788 test images** is a **historical estimate from the Phase 2 log** (`outputs/logs/image_feature_extraction.log`: 25,959 images processed at ~9.8 imgs/s). It was **not freshly measured in this audit** and must not be presented as a newly measured benchmark or directly compared to cached classifier latencies.
> 2. *Throughput Formula:* Throughput is computed from unrounded raw median latency ($7,788 / \text{median}$). Displaying 4-decimal rounded values ($0.0010\text{ s}$) creates an apparent manual arithmetic mismatch; authoritative full-precision median ($0.000962498\text{ s}$) governs all throughput calculations.
