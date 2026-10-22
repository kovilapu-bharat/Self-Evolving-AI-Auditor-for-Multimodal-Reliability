# A Self-Evolving AI Auditing Algorithm for Adaptive Reliability Diagnosis and Recovery in Multimodal AI Systems (SEAA)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SEAA (A Self-Evolving AI Auditing Algorithm for Adaptive Reliability Diagnosis and Recovery in Multimodal AI Systems)** is a research-grade multimodal machine learning and reliability auditing framework built around the **MM-IMDb** benchmark dataset. The system integrates natural language processing and computer vision to perform multi-label movie genre classification while establishing rigorous data leakage prevention, validation-based threshold optimization, and architectural scaffolding for failure diagnosis and automated recovery.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Repository Structure](#-repository-structure)
- [Dataset & Data Pipeline](#-dataset--data-pipeline)
- [Model Architectures](#-model-architectures)
- [Experimental Results](#-experimental-results)
- [Quick Start Guide](#-quick-start-guide)
  - [Prerequisites & Installation](#prerequisites--installation)
  - [Running the Pipelines](#running-the-pipelines)
  - [Running Tests](#running-tests)
- [Development Roadmap](#-development-roadmap)
- [License](#-license)

---

## 🔬 Project Overview

The primary machine learning task is **multi-label classification** across **23 canonical movie genres** using multimodal inputs from the MM-IMDb dataset:
- **Text Modality**: Movie plot summaries and synopses.
- **Vision Modality**: Official movie poster images (RGB, resized to 224×224).
- **Target Space**: 23 binary genre indicators per movie with a mean label cardinality of 2.48 genres.

The system is developed across structured phases to guarantee scientific rigor, auditability, and zero data leakage.

---

## ⚡ Key Features

1. **Grouped Multi-Label Iterative Stratification**:
   - Deterministic 60% Train (15,564) / 10% Validation (2,607) / 30% Test (7,788) split.
   - Text duplicate grouping guarantees identical plot synopses never span across train/val/test splits.
   - Preserves proportional class representations across all 23 genres, even for rare categories (e.g., *Film-Noir*, *Short*).

2. **CPU-Optimized Memory-Mapped Feature Caching**:
   - Pre-extracts 2,048-dimensional visual representations from frozen ResNet-50 backbones.
   - Saves embeddings into disk-backed `numpy.memmap` (`resnet50_features.npy`, 212 MB) with incremental batch flushing and resumption checkpoints.
   - Enables instant training and inference on standard workstations without GPU memory bottlenecks.

3. **Strict Zero-Leakage Parameter Selection**:
   - Per-genre decision thresholds and multimodal late-fusion weights ($\alpha$) are optimized **strictly on the validation set**.
   - Test labels are never accessed during feature scaling, classifier fitting, or threshold tuning.

4. **Reliability & Auditing Scaffolding**:
   - Modular structure including `auditing/`, `failures/`, `recovery/`, and `policy/` modules designed for error diagnosis, modality degradation fallbacks, and reinforcement-learning-driven adaptive inference.

---

## 📂 Repository Structure

```
SEAA/
├── README.md                            # Comprehensive project documentation
├── requirements.txt                     # Project environment dependencies
├── .gitignore                           # Git ignore specifications
├── data/
│   ├── metadata.csv                     # 25,959 indexed movie records with lengths & labels
│   ├── label_mapping.json               # Canonical 23-genre index mapping
│   ├── split_indices.json               # Deterministic 60/10/30 train/val/test index arrays
│   └── processed/
│       ├── resnet50_features.npy        # 2,048-dim visual feature cache (25,959 × 2,048 float32)
│       └── resnet50_features_info.json  # Checkpoint metadata for feature extraction
├── scripts/
│   ├── phase1_verify_environment.py     # System, hardware, and library diagnostic script
│   ├── prepare_dataset.py               # Dataset download, cleaning & stratification pipeline
│   ├── extract_image_features.py        # ResNet-50 visual embedding extraction runner
│   ├── train_text_baseline.py           # Text-only baseline training and threshold optimization
│   ├── train_image_baseline.py          # Image-only baseline training script
│   ├── train_multimodal_baseline.py     # Multimodal validation-tuned weighted late fusion & plot generation
│   └── run_baselines.py                 # Unified end-to-end baseline execution runner
├── src/
│   ├── data/
│   │   ├── dataset.py                   # PyTorch MMIMDbDataset, transforms & text normalizer
│   │   └── pipeline.py                  # Multi-label iterative stratifier and data export
│   ├── models/
│   │   ├── text_baseline.py             # TF-IDF + One-vs-Rest Logistic Regression
│   │   ├── image_baseline.py            # ResNet-50 features + One-vs-Rest Logistic Regression
│   │   ├── image_feature_extractor.py   # ResNet-50 extraction with memory mapping
│   │   └── multimodal_baseline.py       # Validation-tuned weighted late fusion
│   ├── evaluation/
│   │   └── metrics.py                   # Micro/Macro F1, mAP, Hamming Loss & threshold tuner
│   ├── auditing/                        # Reliability monitoring & audit log handlers
│   ├── failures/                        # Failure detection & modality disagreement diagnosis
│   ├── recovery/                        # Fallback & recovery mitigation routines
│   └── policy/                          # Experience-based adaptive decision policies
├── tests/
│   ├── test_dataset.py                  # 12 validation checks for split & pipeline integrity
│   └── test_baselines.py                # Unit tests for baseline models & metric validity
└── outputs/
    ├── checkpoints/                     # Serialized model weights (joblib / json)
    ├── logs/                            # Execution logs
    ├── metrics/                         # JSON & CSV metric evaluation reports
    ├── predictions/                     # Predicted probability arrays and test CSVs
    └── reports/                         # Markdown quality reports and environment summaries
```

---

## 📊 Dataset & Data Pipeline

The MM-IMDb dataset contains **25,959 samples** across **23 movie genres**:

| Metric | Value |
|---|---|
| **Total Movie Samples** | 25,959 |
| **Modalities** | Text (plot summaries) + Vision (poster RGB images) |
| **Label Space** | 23 genres (Multi-hot binary vector) |
| **Mean Plot Length** | 614.7 characters (106.2 words) |
| **Mean Genre Cardinality** | 2.48 genres per movie |
| **Train Set Size** | 15,564 samples (59.96%) |
| **Validation Set Size** | 2,607 samples (10.04%) |
| **Test Set Size** | 7,788 samples (30.00%) |
| **Duplicate Text Groups** | 11 distinct groups (25 affected records, 14 redundant copies; 5 Train, 3 Val, 3 Test; 0 cross-split duplicates) |

### Genre Distribution
`Action`, `Adventure`, `Animation`, `Biography`, `Comedy`, `Crime`, `Documentary`, `Drama`, `Family`, `Fantasy`, `Film-Noir`, `History`, `Horror`, `Music`, `Musical`, `Mystery`, `Romance`, `Sci-Fi`, `Short`, `Sport`, `Thriller`, `War`, `Western`.

---

## 🧠 Model Architectures

### 1. Text-Only Baseline
- **Feature Representation**: Sublinear TF-IDF vectorizer (10,000 max features, unigrams & bigrams, sublinear term frequency, minimum document frequency $df \ge 2$).
- **Classifier**: One-vs-Rest Logistic Regression ($C = 1.0$).
- **Decision Thresholds**: Per-genre threshold search on validation split to maximize F1 score.

### 2. Image-Only Baseline
- **Feature Representation**: Pretrained frozen ResNet-50 with penultimate Global Average Pooling (2,048-dimensional embeddings).
- **Classifier**: Standardized feature inputs with One-vs-Rest Logistic Regression ($C = 0.5$).
- **Decision Thresholds**: Per-genre threshold search on validation split.

### 3. Multimodal Weighted Late Fusion
- **Mechanism**: Validation-tuned weighted late fusion:
  $$P_{\text{multimodal}} = \alpha P_{\text{text}} + (1 - \alpha) P_{\text{image}}$$
- **Optimization**: Fixed scalar $\alpha = 0.55$ selected via validation grid search (image weight $0.45$), followed by per-genre threshold tuning on the fused validation probabilities.

---

## 📈 Experimental Results

### Consolidated Baseline Performance (Test Split — 7,788 samples)

| Model Architecture | Modality | Micro-F1 | Macro-F1 | Precision | Recall | Mean AP (mAP) | Hamming Loss | Latency (s) | Throughput (samples/s) |
|---|---|---|---|---|---|---|---|---|---|
| **Text-Only (TF-IDF + LR)** | Text | **0.6056** | **0.4955** | 0.5594 | **0.6602** | **0.5183** | **0.0929** | 1.54s | 5,052.2 |
| **Image-Only (ResNet50 + LR)** | Image (Cached Feats) | 0.4151 | 0.2847 | 0.3373 | 0.5396 | 0.2337 | 0.1642 | 1.94s | 4,015.7 |
| **Multimodal (Validation-Tuned Weighted Late Fusion)** | Multimodal (Cached E2E) | 0.5161 | 0.4003 | 0.4301 | 0.6451 | 0.3985 | 0.1307 | 3.60s | 2,162.2 |

### Key Empirical Insights & Mathematical Corrections
- **Threshold Optimization**: Tuning per-genre decision thresholds on the validation split elevated Text-Only validation Micro-F1 from `0.4744` (default 0.5 threshold) to `0.6042`. This represents an absolute increase of **0.1298** (**12.98 percentage points**) and a relative improvement of **27.36%** ($[0.6042 - 0.4744] / 0.4744 \times 100$).
- **Modality Contribution**: Multimodal weighted late fusion ($\alpha = 0.55$ text, $0.45$ image) shows an observed absolute improvement over Image-Only of **+10.10 percentage points** in Micro-F1 (0.5161 vs. 0.4151) and **+16.48 percentage points** in mAP (0.3985 vs. 0.2337). No statistical significance claims are made as formal hypothesis tests were not conducted.
- **Text Dominance**: Text-Only remains the top-performing baseline (0.6056 Micro-F1, 0.5183 mAP). Static linear late fusion dilutes dense textual features with weaker visual signals (-8.95 percentage points Micro-F1 compared to Text-Only), demonstrating that deep cross-modal attention or dynamic routing is required for multimodal synergy.
- **Inference Timing**: Benchmarked using `time.perf_counter()` over 1 warm-up repetition followed by 5 timed repetitions across all 7,788 test samples. Multimodal late fusion and threshold application alone requires only **0.000962 seconds** (0.0010s rounded, 8,091,428.6 samples/s), while full cached end-to-end inference (text vectorization + image scaling + dual classifier inference + fusion + thresholding) executes in **3.60 seconds** (2,162.2 samples/s). Model loading is excluded; precomputed ResNet50 features are used.
- **Scope Limitation**: Phase 3 baselines establish static benchmark performance and do not demonstrate adaptive recovery, dynamic routing, or self-evolving auditing mechanisms.

---

## 🚀 Quick Start Guide

### Prerequisites & Installation

1. **Clone or Navigate to the Workspace**:
   ```bash
   cd "SEAA"
   ```

2. **Set up Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

### Running the Pipelines

#### 1. System & Environment Verification (Phase 1)
Verify hardware, PyTorch setup, and Hugging Face connectivity:
```bash
python scripts/phase1_verify_environment.py
```

#### 2. Prepare & Stratify the MM-IMDb Dataset (Phase 2)
Download, clean text, compute stratified splits, and generate data quality reports:
```bash
# Run full dataset preparation:
python scripts/prepare_dataset.py

# Or run a quick smoke-test:
python scripts/prepare_dataset.py --smoke_test
```

#### 3. Extract ResNet-50 Visual Features (Phase 3 Prep)
Extract and cache 2048-dim features for all 25,959 posters:
```bash
python scripts/extract_image_features.py
```

#### 4. Train & Evaluate Baselines (Phase 3)
Execute individual baselines or the unified suite:
```bash
# Run individual baselines:
python scripts/train_text_baseline.py
python scripts/train_image_baseline.py
python scripts/train_multimodal_baseline.py

# Or run the complete unified baseline suite:
python scripts/run_baselines.py
```

---

### Running Tests

Validate dataset integrity and baseline model compliance:
```bash
# Run all unit tests:
python -m unittest discover tests

# Or run specific test modules:
python -m unittest tests/test_dataset.py
python -m unittest tests/test_baselines.py
```

---

## 🗺️ Development Roadmap

- [x] **Phase 1**: Hardware diagnostic, PyTorch verification, and dataset connectivity checks.
- [x] **Phase 2**: Grouped iterative stratified splitting, text normalization, and dataset quality reports.
- [x] **Phase 3**: Feature extraction (ResNet-50), Text baseline, Image baseline, Adaptive Late Fusion, and test suite.
- [ ] **Phase 4 (Auditing & Failure Analysis)**:
  - Disagreement analysis between text and image predictions.
  - Uncertainty estimation and confidence interval scoring.
- [ ] **Phase 5 (Recovery & Adaptive Policy)**:
  - Automated fallback routing for corrupted, noisy, or missing modalities.
  - Experience-based reinforcement learning policy for dynamic modal weighting.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
