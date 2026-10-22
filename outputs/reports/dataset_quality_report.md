# MM-IMDb Dataset Quality & Pipeline Validation Report

**Project:** A Self-Evolving AI Auditing Algorithm for Adaptive Reliability Diagnosis and Recovery in Multimodal AI Systems (SEAA)

## 1. Executive Summary & Verification
- **Source Repository:** `pranavmr/MM-IMDb`
- **Total Samples:** 25,959
- **Distinct Text Values:** 25,945
- **Modalities:** Text (movie plot summaries) + Vision (movie posters, RGB)
- **Target Task:** Multi-label movie genre classification across 23 classes
- **Missing Text Count:** 0
- **Missing / Corrupted Images:** 0
- **Duplicate Text Groups:** 11 distinct duplicate text groups / strings (affecting 25 total records, containing 14 redundant duplicate instances)
  - **Train Split:** 5 duplicate groups (13 affected records, 8 redundant copies)
  - **Validation Split:** 3 duplicate groups (6 affected records, 3 redundant copies)
  - **Test Split:** 3 duplicate groups (6 affected records, 3 redundant copies)
  - **Cross-Split Duplicates:** 0 (100% split confinement; zero duplicate text strings cross splits)

## 2. Split Strategy and Data Leakage Prevention
> **Transparency Note:** The Hugging Face dataset repository `pranavmr/MM-IMDb` provides all 25,959 records under a single `train` split partition. To ensure total reproducibility, zero data leakage, and rigorous class balance, SEAA generates a deterministic, grouped multi-label iterative stratified split.

- **Target Split Proportions:** 60% Train / 10% Validation / 30% Test (Seed: 42)
- **Train Set Size:** 15,564 samples (59.96%)
- **Validation Set Size:** 2,607 samples (10.04%)
- **Test Set Size:** 7,788 samples (30.00%)
- **Leakage Guard:** Duplicate plot strings are grouped so that identical text never spans multiple splits (0 cross-split duplicates across all 11 duplicate groups).
- **Disjoint Verification:** `Train ∩ Val = ∅`, `Train ∩ Test = ∅`, `Val ∩ Test = ∅` (100% disjoint index sets).

## 3. Genre Distribution Across Splits
| Genre | Overall Count | Train Count | Val Count | Test Count | Train % | Val % | Test % |
|---|---|---|---|---|---|---|---|
| **Action** | 3,550 | 2,131 | 355 | 1,064 | 60.0% | 10.0% | 30.0% |
| **Adventure** | 2,710 | 1,627 | 271 | 812 | 60.0% | 10.0% | 30.0% |
| **Animation** | 997 | 597 | 101 | 299 | 59.9% | 10.1% | 30.0% |
| **Biography** | 1,343 | 807 | 134 | 402 | 60.1% | 10.0% | 29.9% |
| **Comedy** | 8,592 | 5,156 | 859 | 2,577 | 60.0% | 10.0% | 30.0% |
| **Crime** | 3,838 | 2,303 | 384 | 1,151 | 60.0% | 10.0% | 30.0% |
| **Documentary** | 2,082 | 1,251 | 208 | 623 | 60.1% | 10.0% | 29.9% |
| **Drama** | 13,967 | 8,381 | 1,395 | 4,191 | 60.0% | 10.0% | 30.0% |
| **Family** | 1,668 | 1,001 | 166 | 501 | 60.0% | 10.0% | 30.0% |
| **Fantasy** | 1,933 | 1,159 | 194 | 580 | 60.0% | 10.0% | 30.0% |
| **Film-Noir** | 338 | 202 | 34 | 102 | 59.8% | 10.1% | 30.2% |
| **History** | 1,143 | 685 | 115 | 343 | 59.9% | 10.1% | 30.0% |
| **Horror** | 2,703 | 1,621 | 270 | 812 | 60.0% | 10.0% | 30.0% |
| **Music** | 1,045 | 626 | 106 | 313 | 59.9% | 10.1% | 30.0% |
| **Musical** | 841 | 504 | 85 | 252 | 59.9% | 10.1% | 30.0% |
| **Mystery** | 2,057 | 1,233 | 206 | 618 | 59.9% | 10.0% | 30.0% |
| **Romance** | 5,364 | 3,219 | 536 | 1,609 | 60.0% | 10.0% | 30.0% |
| **Sci-Fi** | 1,991 | 1,193 | 201 | 597 | 59.9% | 10.1% | 30.0% |
| **Short** | 471 | 282 | 48 | 141 | 59.9% | 10.2% | 29.9% |
| **Sport** | 634 | 380 | 64 | 190 | 59.9% | 10.1% | 30.0% |
| **Thriller** | 5,192 | 3,116 | 519 | 1,557 | 60.0% | 10.0% | 30.0% |
| **War** | 1,335 | 801 | 134 | 400 | 60.0% | 10.0% | 30.0% |
| **Western** | 705 | 423 | 71 | 211 | 60.0% | 10.1% | 29.9% |

## 4. Text & Label Quality Metrics
- **Mean Plot Length:** 614.7 characters (Median: 527)
- **Mean Word Count:** 106.2 words (Min: 1, Max: 1696)
- **Mean Genres per Movie (Cardinality):** 2.48 genres
- **Multi-hot Target Dimension:** 23 binary elements

## 5. Artifact Verification
All required Phase 2 pipeline artifacts have been generated and validated:
- `data/metadata.csv`: Full tabular record index with movie IDs and targets
- `data/label_mapping.json`: Bidirectional mapping for 23 genres
- `data/split_indices.json`: Reproducible split index arrays
- `outputs/reports/dataset_summary.json`: Machine-readable summary statistics
- `outputs/reports/dataset_quality_report.md`: This comprehensive documentation
