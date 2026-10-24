# SEAA Phase 4 — Antigravity Implementation Prompt

## Project
**Title:** A Self-Evolving AI Auditing Algorithm for Adaptive Reliability Diagnosis and Recovery in Multimodal AI Systems (SEAA)

**Workspace:** `C:\\Users\\bharat kumar\\OneDrive\\Desktop\\SEAA`

## Critical execution rule
Implement **Phase 4 only**. Work in small, verifiable steps. After completing Phase 4, run the relevant tests, produce a concise implementation and results report, and **STOP**. Do not begin Phase 5, adaptive recovery, policy learning, or any self-evolving action-selection implementation until the user explicitly approves the next phase.

Do not claim a feature works unless it is implemented and supported by test results or measured experiments. If a command fails, investigate and report the actual error; do not silently skip it.

---

## 1. Begin with an audit of the current workspace

1. Read `PROJECT_STATUS.md` and all relevant Phase 3 reports before changing code.
2. Inspect the existing source tree, configuration, dataset files, saved checkpoints, tests, and output conventions.
3. Identify the actual APIs and prediction formats exposed by the trained text-only, image-only, and multimodal baseline models.
4. Confirm that the dataset split and test set remain unchanged. Do not retrain or overwrite Phase 1–3 checkpoints, metrics, or presentation artifacts.
5. Write a brief implementation plan to `docs/phase4/phase4_plan.md` (or the existing documentation convention) before implementing.
6. If a required file, checkpoint, package, or prediction interface is missing, document the blocker and stop to ask the user rather than fabricating a substitute.

## 2. Implement cross-modal disagreement measurements

Build reusable, tested functions that compare text and image model outputs for the same samples.

Include, where supported by the existing model outputs:

- **Probability disagreement:** Jensen–Shannon divergence between the text and image genre-probability distributions. Since MM-IMDb is multi-label, carefully define how the per-genre Bernoulli probabilities are converted into a valid comparison; document the chosen formula and numerical safeguards.
- **Prediction disagreement:** compare thresholded genre sets from text-only and image-only models, using a clearly defined metric such as Jaccard distance or normalized symmetric difference.
- **Per-genre disagreement:** record which genres disagree, rather than storing only one aggregate number.

Requirements:
- Clip probabilities safely to avoid logarithms of zero.
- Validate array dimensions, finite values, and probability ranges.
- Handle edge cases such as both prediction sets being empty.
- Document the mathematical definitions and limitations. Do not casually treat 23 independent sigmoid outputs as a 23-class softmax distribution.

## 3. Implement uncertainty estimates

Add practical uncertainty features for both modalities and, if supported, the fused prediction.

At minimum consider:
- Mean binary entropy across the 23 genre probabilities.
- Distance to the active, validation-tuned decision thresholds, with a documented definition.
- Number or proportion of predictions close to their thresholds.

Use the project's existing threshold files/configuration if available. Do not tune thresholds on the test set. If thresholds are missing, report that and follow the project's established threshold convention rather than inventing undocumented values.

Requirements:
- Explain what each uncertainty score means and its limitations.
- Include tests for probabilities near 0, 0.5, and 1; threshold-boundary cases; and invalid input.
- Do not present a heuristic score as a calibrated probability of error.

## 4. Simulate controlled modality failures

Implement reproducible, configurable corruption functions for evaluation. Preserve original files and do not overwrite source images or text.

Include only failure types compatible with the actual dataset and model pipeline:

### Text failures
- Truncation to a configurable fraction or token limit.
- Deterministic token masking or token deletion/noise, using a configurable seed.
- Handle empty or very short text safely.

### Image failures
- Missing image / unavailable image representation, where the pipeline can represent it safely.
- Gaussian blur.
- Gaussian noise, with configurable strength and seed.
- Validate image shape, channel order, dtype, and numeric range.

Requirements:
- Every corruption must have a clean input path and a reproducible configuration.
- Use deterministic seeds and preserve the original sample identifiers.
- Keep clean and corrupted predictions separately identifiable.
- Never silently substitute a zero vector or arbitrary prediction for a missing modality unless that fallback is explicitly documented as a tested experimental condition.
- If the saved pipeline only accepts cached ResNet50 features, investigate how a corruption can be applied before feature extraction. Do not claim image corruption was tested if it was applied after feature extraction in a way that does not simulate the intended failure.

## 5. Implement failure diagnosis

Create an auditable diagnostic layer that combines the measurements above and labels likely failure conditions.

Potential diagnostic signals:
- High text uncertainty.
- High image uncertainty.
- Large text/image disagreement.
- Missing modality.
- Prediction instability between clean and corrupted inputs, when both predictions are available.

Requirements:
- Keep raw numeric signals alongside any diagnosis label.
- Make thresholds/configuration explicit and configurable.
- Do not choose diagnostic cutoffs by looking at the test labels. Use validation data or a documented predeclared rule.
- Do not claim the diagnosis is correct merely because code returns a label. Evaluate diagnosis quality against known synthetic corruption labels where feasible.
- Distinguish **corruption detection** from **corruption-type classification**. Report the metrics actually measured (e.g., precision, recall, F1, confusion matrix), and note limitations if only synthetic failures are evaluated.

## 6. Evaluation protocol

Use the existing fixed train/validation/test split. Do not change the split or use test labels to tune the detector.

Create a repeatable Phase 4 evaluation script that:

1. Loads the existing baseline checkpoints and validation-tuned thresholds.
2. Establishes clean-input predictions.
3. Creates controlled corrupted versions without modifying source data.
4. Computes disagreement and uncertainty signals.
5. Produces diagnosis outputs and compares them with known synthetic corruption labels.
6. Reports clean-versus-corrupted changes in relevant multi-label metrics, such as Micro-F1, Macro-F1, and Hamming loss, where labels and predictions are available.
7. Measures runtime only with a documented protocol and clearly distinguishes cached-feature timing from end-to-end image preprocessing.
8. Uses fixed seeds and records the configuration.

Do not overstate the results. If a corruption type cannot be tested correctly with the current architecture, record it as unsupported and explain what is needed.

## 7. Required outputs

Follow existing repository conventions where possible. Create or update the following (use equivalent existing paths if the repository has an established structure):

- `src/auditing/disagreement.py`
- `src/auditing/uncertainty.py`
- `src/failures/text_corruptions.py`
- `src/failures/image_corruptions.py`
- `src/auditing/diagnosis.py`
- `scripts/run_phase4_audit.py`
- `tests/test_disagreement.py`
- `tests/test_uncertainty.py`
- `tests/test_corruptions.py`
- `tests/test_diagnosis.py`
- `docs/phase4/phase4_plan.md`
- `docs/phase4/phase4_results.md`
- `outputs/phase4/` with machine-readable CSV/JSON results and, if supported by the project's existing tooling, clear graphs/tables.

Do not create duplicate modules if equivalent implementations already exist; inspect and extend the existing files instead.

The results report must include:
- What was implemented.
- Exact commands executed and their actual outcomes.
- Test count and pass/fail status.
- Corruption configurations and seeds.
- Disagreement and uncertainty definitions.
- Detection/diagnosis metrics actually measured.
- Clean-versus-corrupted model metrics.
- Runtime protocol and limitations.
- Unsupported tests or blockers.
- Any deviation from this plan.
- A clear statement that Phase 5 has not been started.

## 8. Testing and quality requirements

- Add unit tests for normal inputs, edge cases, malformed shapes, invalid values, determinism, and preservation of source data.
- Run the relevant existing tests as well as new Phase 4 tests.
- Run the full test suite if feasible; report exact command, number of tests, and actual result.
- Check that Phase 1–3 artifacts and reported metrics were not modified.
- Ensure all new scripts work from the project root and have clear help/usage instructions.
- Avoid unnecessary dependencies; check installed packages before adding any.
- Keep the implementation CPU-compatible and mindful of approximately 16 GB RAM.
- Do not load every poster or materialize large arrays unnecessarily; use batching/streaming where possible.
- Use meaningful logging and actionable errors. No silent exception swallowing.

## 9. Final response format

When done, summarize:

1. Phase 4 completion status (complete, partial, or blocked).
2. Files created/modified.
3. Exact commands and test results.
4. Actual measured findings, with no invented numbers.
5. Known limitations and unsupported failure modes.
6. Confirmation that Phase 1–3 artifacts were preserved.
7. Confirmation that Phase 5 was **not** started.

Then **STOP and wait for the user's explicit approval** before doing any further phase.
