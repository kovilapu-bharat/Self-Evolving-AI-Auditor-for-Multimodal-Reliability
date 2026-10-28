# SEAA Phase 3: Executive Summary of Key Findings & Supervisor Briefing

**Project:** A Self-Evolving AI Auditing Algorithm for Adaptive Reliability Diagnosis and Recovery in Multimodal AI Systems (SEAA)  
**Author:** Antigravity Autonomous Implementation & Auditing Agent  
**Status:** Phase 3 Formally Closed & Fully Verified; Phase 4 Has NOT Started  
**Target Audience:** Academic Supervisor & Project Committee  

---

### Executive Key Findings

1. **Text Modality Dominance in MM-IMDb Multi-Label Classification:**  
   The Text-Only baseline (TF-IDF with 10,000 unigrams/bigrams and OneVsRest Logistic Regression) achieved the highest overall performance on the 7,788 test records: **0.6056 Micro-F1**, **0.4955 Macro-F1**, and **0.5183 mAP**, with a Hamming loss of **0.0929**. Textual plot summaries provide rich, direct semantic descriptors that drive accurate multi-label assignment.

2. **Image-Only Performance Floor:**  
   The Image-Only baseline (pre-extracted ResNet50 2,048-dim deep visual features with StandardScaler and Logistic Regression) attained **0.4151 Micro-F1**, **0.2847 Macro-F1**, and **0.2337 mAP**, with a Hamming loss of **0.1642**. Movie posters exhibit heavy artistic variance, typography, and ambiguous visual metaphors, making them weaker standalone predictors for multi-label genre classification.

3. **Validation-Tuned Weighted Late Fusion Performance Characteristics:**  
   Multimodal late fusion using validation-selected scalar weights ($\alpha = 0.55$ text, $0.45$ image) and per-genre threshold optimization achieved **0.5161 Micro-F1**, **0.4003 Macro-F1**, and **0.3985 mAP**. While late fusion demonstrated a strong absolute improvement of **+10.10 percentage points Micro-F1** over the Image-Only baseline, it underperformed the Text-Only baseline by **-8.95 percentage points Micro-F1**.

4. **Root Cause Analysis of Static Fusion Dilution:**  
   Linear probability blending with fixed scalar weights treats all test instances uniformly. When movie posters contain non-informative visual cues (e.g., stylized portraits or minimalist title cards), blending them with high-confidence text probabilities dilutes true genre signals below the classification threshold, reducing precision and recall on text-predictable genres.

5. **Impact of Per-Genre Threshold Optimization:**  
   Optimizing individual decision thresholds per genre on the validation split yielded substantial gains over the default 0.50 cutoff. For the text baseline, validation Micro-F1 increased from 0.4744 to 0.6042—an absolute gain of **+12.98 percentage points** (representing a **+27.36% relative improvement**). This confirms that multi-label class imbalance necessitates genre-specific calibrated thresholds.

6. **Inference Latency & Timing Architecture Delineation:**  
   Benchmarking across 5 timed repetitions on 7,788 test samples revealed steady-state CPU latencies:
   - Text Classifier Inference: **1.5415 s** ($5,052.2$ samples/s)
   - Image Classifier Cached Inference: **1.9394 s** ($4,015.7$ samples/s)
   - End-to-End Multimodal Pipeline (consuming raw text and cached image embeddings): **3.6020 s** ($2,162.2$ samples/s)
   - Isolated Fusion & Thresholding: **0.000962 s** ($0.962\text{ ms}$, $8,091,428.6$ samples/s).
   *Scope Distinction:* The isolated fusion overhead represents pure array arithmetic and must never be conflated with full model inference. In contrast, raw poster image feature extraction on CPU was historically estimated from Phase 2 logs at **794.7 s** (~9.8 imgs/s).

7. **Zero Leakage & Data Partition Integrity:**  
   The dataset partitioning (Train: 15,564 / 59.96%; Validation: 2,607 / 10.04%; Test: 7,788 / 30.00%) has zero overlap ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$). All 11 duplicate plot description groups (25 affected records, 14 redundant copies) are 100% strictly confined within single splits, ensuring zero data leakage into the test set.

---

### Project Limitations & Scope Boundary

- **Descriptive Evidence Policy:** All reported metric differences between models are empirical observations. No non-parametric significance testing (e.g., Wilcoxon signed-rank or McNemar's tests) was conducted.
- **Static Baseline Context:** The late fusion baseline is a static linear combination with fixed $\alpha=0.55$. It does not perform adaptive failure diagnosis or dynamic recovery.
- **Phase 4 Roadmap:** Phase 4 (Multimodal Failure Diagnosis) is designed to systematically identify when and why modalities conflict, segmenting failures into modality disagreement, visual feature noise, and threshold miscalibration.
- **Current Status:** Phase 4 and Phase 5 have **NOT** been started. All artifacts presented here represent the closed, verified Phase 3 baseline.
