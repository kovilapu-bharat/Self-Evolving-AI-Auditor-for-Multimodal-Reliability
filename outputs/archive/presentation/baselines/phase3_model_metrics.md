# SEAA Phase 3: Comprehensive Model Baseline Metrics

**Project:** A Self-Evolving AI Auditing Algorithm for Adaptive Reliability Diagnosis and Recovery in Multimodal AI Systems (SEAA)  
**Dataset:** MM-IMDb Multi-Label Classification (23 Genres)  
**Evaluation Set:** Test Split ($N = 7,788$ movie records; Random Seed 42)  
**Source Provenance:** Formally verified from `outputs/metrics/baseline_comparison.csv` and `outputs/reports/phase3_metric_verification.json`.

| Model Architecture | Modality | Micro-F1 | Macro-F1 | Micro-Prec | Macro-Prec | Micro-Rec | Macro-Rec | mAP | Hamming Loss |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Text-Only (TF-IDF + LR)** | Text | **0.6056** | **0.4955** | **0.5594** | **0.5126** | **0.6602** | **0.5179** | **0.5183** | **0.0929** |
| **Image-Only (ResNet50 + LR)** | Image (Cached Feats) | 0.4151 | 0.2847 | 0.3373 | 0.2444 | 0.5396 | 0.3632 | 0.2337 | 0.1642 |
| **Multimodal (Validation-Tuned Weighted Late Fusion)** | Multimodal (Cached E2E) | 0.5161 | 0.4003 | 0.4301 | 0.3523 | 0.6451 | 0.5150 | 0.3985 | 0.1307 |

### Observational Summary:
1. **Text Modality Dominance:** Text-Only achieves the highest performance across all evaluation metrics, reaching **0.6056 Micro-F1** and **0.5183 mAP** with the lowest Hamming loss (**0.0929**).
2. **Modality Gain & Fusion Dilution:** Validation-Tuned Weighted Late Fusion ($\alpha = 0.55$) shows an absolute gain of **+10.10 percentage points Micro-F1** over the Image-Only baseline (**0.5161** vs. **0.4151**). However, it falls short of Text-Only by **-8.95 percentage points Micro-F1**, indicating that naive static late fusion dilutes high-confidence textual semantic signals with lower-confidence visual signals.
3. **Statistical Clarification:** Observed differences are empirical and descriptive. No statistical significance tests were performed.
