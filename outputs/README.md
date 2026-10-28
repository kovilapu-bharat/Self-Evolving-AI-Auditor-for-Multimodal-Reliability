# SEAA Project Outputs Directory Map

This directory contains experimental outputs, trained model checkpoints, and evaluation results for the **SEAA (Self-Evolving AI Auditor)** framework.

---

## 🌟 Primary Directory for IEEE Publication
* **[`ieee_final/`](./ieee_final/)**: **THE MAIN AUTHORITATIVE FOLDER FOR THE IEEE PAPER.**
  Contains all publication-grade figures (Figures 1–5 in vector PDF and 300 DPI PNG), Table 1 (CSV, Markdown), source CSV tables, and the formal verification audit log.

---

## 🔬 Core Active Pipeline Folders (System & Execution)
* **[`checkpoints/`](./checkpoints/)**: Serialized trained machine learning models (`text_baseline.joblib`, `image_baseline.joblib`, `multimodal_baseline.json`).
* **[`metrics/`](./metrics/)**: Baseline benchmark metrics and comparison data (`baseline_comparison.csv`).
* **[`recovery/`](./recovery/)**: Cleaned master adaptive recovery benchmarks, including the canonical 4,000-decision evaluation run (`run_20261025_phase5_adaptive_recovery/`), component ablations, sensitivity sweeps, and latency benchmarks.
* **[`auditing/`](./auditing/)**: Inline reliability auditing logs, disagreement distributions, and corruption degradation metrics.
* **[`reports/`](./reports/)**: Markdown and JSON system verification, environment, and dataset reports.
* **[`predictions/`](./predictions/)**: Model prediction arrays on validation/test sets.
* **[`logs/`](./logs/)**: Execution log files.

---

## 📦 Archive Directory
* **[`archive/`](./archive/)**: All superseded drafts, obsolete duplicate figures, presentation slide decks, and temporary smoke test runs have been cleanly quarantined here to keep the active outputs organized and clutter-free.
  * `archive/paper_outputs_draft/`: Superseded preliminary draft (replaced by `ieee_final/`).
  * `archive/redundant_figures/`: Duplicate figure copies from intermediate milestones.
  * `archive/test_runs/`: Quick sanity checks and reproduction test runs.
  * `archive/presentation/`: Slide deck charts and summary tables.

