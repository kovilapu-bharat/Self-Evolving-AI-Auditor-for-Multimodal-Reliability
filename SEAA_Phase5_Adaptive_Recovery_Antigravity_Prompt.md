# SEAA Phase 5 — Controlled Implementation Prompt for Antigravity

## Project
**Title:** A Self-Evolving AI Auditing Algorithm for Adaptive Reliability Diagnosis and Recovery in Multimodal AI Systems (SEAA)

**Workspace:** `C:\Users\bharat kumar\OneDrive\Desktop\SEAA`

The project uses MM-IMDb multi-label genre classification (23 genres) with plot text and poster images. The intended SEAA loop is:

**Monitor → Diagnose → Recover → Evaluate → Learn**

Phase 5 must implement and evaluate an adaptive recovery policy that learns from recovery outcomes and improves future action selection. A fixed rule-based router alone is not evidence of self-evolution.

---

## Critical operating rules

1. Start by inspecting the repository as it currently exists. Previous reports claim that Phase 5 components such as `src/recovery/modality_gating.py` and `src/recovery/confidence_recovery.py` may already exist. Treat those statements as claims to verify.
2. Do not overwrite, delete, rename, or silently regenerate existing outputs, checkpoints, dataset files, Phase 1–3 artifacts, or earlier Phase 4 results.
3. Do not change the Phase 1–3 baseline pipeline or frozen artifacts. If an interface change is necessary, explain it first and prefer a new Phase 5 adapter or module.
4. Preserve existing work. Do not remove partially implemented Phase 5 files. Inspect and reuse them if sound; report defects before changing them.
5. Keep Phase 5 outputs separate. Use a fresh directory such as `outputs/phase5/<unique_run_id>/` so nothing is overwritten.
6. Work in small, testable steps. Do not claim a feature works unless tests and empirical measurements support it.
7. Do not fabricate metrics, training history, data provenance, runtime measurements, or successful tests. Distinguish **verified**, **failed**, and **not yet tested**.
8. Do not silently change dataset splits, sample IDs, targets, label order, model checkpoints, or evaluation protocol.
9. Run the full existing test suite before and after changes:
   `python -m unittest discover -s tests -p "test_*.py" -v`
10. Stop after Phase 5. Do not begin Phase 6 or unrelated work without my explicit approval.

---

## Step 0 — Establish the real current state (mandatory)

Before implementation:

- Record the current Git branch, `git status --short`, and `git diff`.
- Identify the baseline commit, including `6ff6058` if it exists.
- List tracked and untracked files added or changed since the baseline.
- Record file hashes and timestamps for relevant Phase 1–3 models, configs, scripts, and outputs where feasible.
- Read `PROJECT_STATUS.md`, the latest `phase4_critical_review(2).md` if present, and current Phase 4/5 documentation.
- Inspect the actual source of all recovery, diagnosis, policy, and evaluation modules.
- Determine whether `DynamicModalityGating`, `ConfidenceRecoveryRouter`, learned diagnostics, or other Phase 5 components exist. State what each actually does and whether the main experiment pipeline uses it.
- Run the full test suite. Do not treat a report's claim of passing tests as evidence that tests pass now.
- Write a starting-state report to a new file without overwriting existing documentation.

### Data leakage and integrity gate

Inspect existing learned diagnostic/recovery components and confirm:

- Training features and labels use only permitted training/validation data.
- The test set is not used to train, fit thresholds, select actions, tune hyperparameters, or choose checkpoints.
- Validation data used for calibration is not described as test data.
- Synthetic failures are generated without using test outcomes to decide the recovery policy.
- Sample IDs, targets, and the fixed 23-genre label order remain aligned.
- The code distinguishes explicit failure metadata from signals inferred from the input itself.

If leakage or alignment problems are found, do not train or report final performance until corrected and tested. Explain the issue and minimum safe correction.

---

## Step 1 — Define the research question and protocol

Research question:

> Does an outcome-learning recovery policy select recovery actions more effectively than a fixed recovery policy under multimodal input failures, while preserving performance on clean inputs?

Document before implementation:

- Available recovery actions and when each is valid.
- Observable signals available to the policy at decision time.
- What constitutes a recovery outcome and how reward/utility is calculated.
- How the policy updates from an outcome and how that update changes a future decision.
- How exploration is controlled so it cannot contaminate evaluation.
- The fixed-policy baseline and no-recovery baseline.
- Train/validation/test separation and exact evaluation protocol.

The policy must not use ground-truth genre labels, future outcomes, or hidden corruption labels at inference time. Ground-truth labels may be used after prediction to evaluate an outcome, but not to choose the action for that same test example.

---

## Step 2 — Implement or complete adaptive recovery

First determine whether existing Phase 5 modules can be safely reused. Do not create duplicates.

Implement a small, interpretable, reproducible policy. A contextual bandit or a carefully designed outcome-updated action-value table is acceptable. It must:

1. Select among supported actions such as text-only, image-only, static fusion, or reliability-weighted fusion, based on modalities and signals actually available.
2. Record context, selected action, confidence/reliability signals, prediction, action outcome, reward, and policy update.
3. Update action values or policy parameters from observed outcomes using a documented update rule.
4. Demonstrate that learning changes a later action selection in a controlled test; logging outcomes alone is insufficient.
5. Be deterministic under a fixed seed when exploration is disabled.
6. Use a safe fallback for missing, invalid, or out-of-range signals.
7. Not assume high prediction confidence alone means a modality is reliable. Document proxy signals and limitations.
8. Support an ablation mode with learning disabled.
9. Keep policy training/calibration separate from final test evaluation.
10. Save policy configuration, random seed, update history, and evaluation metadata in the new Phase 5 run directory.

Prefer modular additions. Avoid unnecessary retraining of Phase 1–3 models. If existing model outputs cannot support a valid experiment, explain the gap before changing the baseline pipeline.

---

## Step 3 — Tests required

Add focused tests for:

- Valid action selection for text-only, image-only, and both-modalities-available cases.
- Missing/malformed signals trigger a safe fallback.
- No action requires an unavailable modality.
- Policy update changes action value or policy state as specified.
- A controlled reward sequence changes a subsequent action selection.
- Fixed seed gives reproducible decisions when exploration is enabled.
- Evaluation mode does not update the policy.
- Test labels and corruption ground truth are never passed into action selection.
- Sample IDs and 23-label targets remain aligned after routing.
- Clean examples are not routed to recovery without a stated reason.
- Existing Phase 1–4 tests do not regress.

Run the full test suite, not only the new tests. Report the exact command and test count.

---

## Step 4 — Evaluation protocol

Use identical example IDs and targets across all policy comparisons. Keep the same underlying model predictions/checkpoints for a fair comparison unless a separate experiment explicitly studies a different model.

Evaluate at minimum, where supported by the actual code:
- Clean inputs.
- Missing/empty text.
- Missing/blackout image.
- Text corruption/noise.
- Image blur/noise.
- Dual-modality corruption.
- Cross-modal conflict, only if the injected condition is precisely defined and genuinely generated.

Compare:
1. No recovery.
2. Fixed recovery policy.
3. Adaptive/outcome-learning recovery policy.
4. Optional oracle policy, clearly labelled as an upper bound and never presented as deployable.

Report:
- Micro-F1 and Macro-F1.
- Per-condition delta relative to the **same-ID** no-recovery baseline.
- Per-condition results, not just an overall average.
- Number of examples per condition and exact IDs or a reproducible ID-list file.
- Confidence intervals or repeated-seed variability when feasible.
- Clean-input performance retention.
- Action-selection frequencies and policy update behaviour.
- Runtime overhead with timing method and units.
- Conditions where recovery reduces performance. Keep negative results visible.

Do not compare a new 1,000-sample result directly with a historical 500-sample or full-test result as if they were the same experiment. If clean baseline differs from earlier reports, identify whether model, split, IDs, checkpoint, label mapping, or fusion method changed and explain the discrepancy.

Write all outputs under a fresh run directory such as `outputs/phase5/<unique_run_id>/`. Never overwrite earlier benchmark CSVs or figures.

---

## Step 5 — Verify the “self-evolving” claim

Before claiming self-evolution, show evidence that:

- An earlier action's outcome is recorded.
- That outcome updates policy state.
- The updated state affects a later action choice in a reproducible controlled test.
- The learned policy is compared with a fixed policy on the same examples.
- Test labels do not leak into action selection.
- Improvement is measured, including negative outcomes and uncertainty where feasible.

If these criteria are not met, describe it as a prototype adaptive recovery policy and list the missing evidence.

---

## Step 6 — Final report and stop

Create a new Phase 5 report containing:

1. Executive summary.
2. Verified starting repository state.
3. Files created or modified.
4. Existing Phase 5 code reused, corrected, or left unchanged.
5. Policy design and update equation.
6. Data leakage and split-integrity checks.
7. Tests run and exact results.
8. Evaluation configuration and reproducibility details.
9. Results table with same-ID comparisons.
10. Failure cases and negative recovery deltas.
11. Runtime measurements and methodology.
12. Limitations and unresolved risks.
13. Confirmation that Phase 1–3 artifacts remained unchanged.
14. Recommended next step, without implementing it.

List exact new files and output locations. Explicitly state which claims are verified, failed, or untested. **Stop after Phase 5 and wait for my approval before starting another phase.**

## Expected behaviour

Be conservative and research-oriented. Preserve previous work, verify before trusting reports, implement only what is missing, measure honestly, and make no unsupported claims.
