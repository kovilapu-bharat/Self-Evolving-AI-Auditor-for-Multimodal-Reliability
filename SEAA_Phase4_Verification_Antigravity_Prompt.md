# SEAA Phase 4 Verification — Antigravity Prompt

## Project details
**Project:** A Self-Evolving AI Auditing Algorithm for Adaptive Reliability Diagnosis and Recovery in Multimodal AI Systems (SEAA)  
**Workspace:** `C:\\Users\\bharat kumar\\OneDrive\\Desktop\\SEAA`

## Objective
Independently verify whether Phase 4 is implemented correctly, runs without errors, and is supported by real tests and experimental evidence. Do not assume it is complete just because files exist or a previous status document says it is complete.

## Strict rules
1. **Audit and verify first. Do not start Phase 5.**
2. Do not overwrite or alter Phase 1–3 checkpoints, dataset splits, baseline metrics, or presentation artifacts.
3. Do not fabricate test results, metrics, files, or successful command executions.
4. Do not change code just to make tests pass. First report the failures and their causes. Make fixes only when the user explicitly approves them.
5. If a command fails, capture its full output and exit code.
6. Use the existing project environment and dependencies where possible. Keep the workflow CPU-compatible and mindful of approximately 16 GB RAM.
7. Run commands from the project root unless the project requires otherwise.

## Step 1 — Inspect the project
- Read `PROJECT_STATUS.md`, the Phase 4 plan and results reports, and relevant Phase 3 reports.
- Inspect the repository structure, source code, tests, configuration, saved checkpoints, threshold files, and output folders.
- Identify the actual Phase 4 modules and evaluation entry point. Do not assume filenames from a plan exist.
- Check for placeholders, `TODO`/`pass` implementations, dummy outputs, hard-coded results, silent exception swallowing, and unimplemented branches.
- Verify that Phase 4 uses the actual trained baseline models and the existing validation-tuned thresholds.
- If required artifacts are missing, report the blocker rather than creating fake substitutes.

## Step 2 — Verify Python environment and imports
Run and record:
- `python --version`
- `python -m pytest --version` (if pytest is installed)
- Import checks for the actual Phase 4 modules and evaluation script.
- Any repository-supported lint/type checks, if already configured.

Record exact commands, exit codes, and relevant output. Do not install new dependencies without explaining why and asking first if installation would change the environment.

## Step 3 — Run automated tests
- Run all Phase 4-specific tests that actually exist.
- Run the existing full test suite, for example `python -m pytest -v`, if pytest is the project's test runner.
- If the project uses another test runner, use the repository's documented command.
- Report total, passed, failed, skipped, and errored tests.
- Inspect every failure and distinguish a test failure from an environment/setup problem.
- Do not claim the suite passed if the command returned a nonzero exit code.

## Step 4 — Verify core functionality
Use actual project APIs and real model outputs where feasible.

### A. Cross-modal disagreement
- Test examples where text/image predictions agree and disagree.
- Check probability ranges, array shapes, finite values, empty predicted-label sets, and per-genre disagreement.
- Inspect the mathematical definition. Do not treat 23 independent multi-label genre probabilities as a 23-class softmax distribution without a justified transformation.

### B. Uncertainty
- Test entropy and threshold-distance calculations on valid values and edge cases near 0, 0.5, and 1.
- Test threshold boundaries, malformed shapes, NaN/infinity, and out-of-range probabilities.
- Confirm thresholds come from validation or the project's saved configuration, not tuning on test labels.
- Do not describe heuristic uncertainty as a calibrated probability of error.

### C. Controlled failures
- Test every implemented text and image corruption.
- Confirm deterministic behavior when a fixed seed is used.
- Confirm original source text/images are not overwritten.
- Confirm sample IDs and labels remain aligned.
- Confirm missing-modality handling is explicit and tested; flag undocumented zero-vector or arbitrary fallbacks.
- If image corruption is intended to simulate blur/noise, confirm it is applied before image feature extraction. Do not count post-feature perturbation as a valid pixel-level corruption experiment unless specifically justified.

### D. Failure diagnosis
- Test diagnostic rules with known synthetic examples and ground-truth corruption labels.
- Measure appropriate metrics such as precision, recall, F1, and a confusion matrix where feasible.
- Distinguish detecting that a sample is corrupted from correctly identifying the corruption type.
- Do not claim diagnosis works solely because the code returns a label.

### E. End-to-end integration
- Identify the actual evaluation script and inspect its `--help` or documented usage before running it.
- Run the complete Phase 4 audit pipeline using the actual dataset, checkpoints, thresholds, and configuration.
- Do not guess command-line arguments.
- Confirm the script exits successfully and generates non-empty, readable outputs.
- Compare clean and corrupted Micro-F1, Macro-F1, and Hamming loss where supported.
- Record runtime protocol and distinguish cached-feature timing from end-to-end image preprocessing.

## Step 5 — Verify outputs and protect earlier work
- Inspect generated CSV/JSON/report files for missing rows, duplicate IDs, invalid values, NaNs, and mismatched metrics.
- Confirm the report agrees with the actual outputs and terminal logs.
- Compare Phase 1–3 artifacts and metric files against repository history, checksums, or a before/after file inventory when available.
- Do not claim artifacts were preserved unless there is evidence.
- Note any test that could not be performed and why.

## Step 6 — Create a verification report
Create `docs/phase4/phase4_verification_report.md` (or the existing equivalent report path) containing:

1. Overall verdict: **PASS**, **PASS WITH LIMITATIONS**, **FAIL**, or **BLOCKED**.
2. A requirement-by-requirement checklist with evidence.
3. Exact commands executed and actual exit codes.
4. Automated test counts: total, passed, failed, skipped, and errors.
5. Functional and integration test results.
6. Actual measured experimental metrics—never invented or copied as if newly measured.
7. Issues found, severity, likely cause, and recommended next action.
8. Tests that could not be run and the reason.
9. Evidence on whether Phase 1–3 artifacts were preserved.
10. Confirmation that Phase 5 was not started.

## Verdict criteria
- **PASS:** core implementation, automated tests, end-to-end integration, and measurable evaluation all work and support the claims.
- **PASS WITH LIMITATIONS:** verified features work, but documented limitations or unsupported cases remain.
- **FAIL:** substantive implementation, test, or integration failures exist.
- **BLOCKED:** missing artifacts, data, dependencies, or environment problems prevent a meaningful verification.

## Final response to the user
Summarize the verdict, key evidence, test results, pipeline outcome, actual findings, blockers/limitations, and report path. Be truthful and concise.

**STOP after the audit and report. Do not fix code or start Phase 5 without the user's explicit approval.**
