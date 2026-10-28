# SEAA Research Project — Complete Technical Explanation and IEEE Publication Plan

## Project title

**A Self-Evolving AI Auditing Algorithm for Adaptive Reliability Diagnosis and Recovery in Multimodal AI Systems**

**Acronym:** SEAA — Self-Evolving AI Auditing Algorithm

This document explains the research problem, algorithm, dataset, implementation, reported results, mathematics, experimental design, remaining work, and publication plan. It distinguishes reported implementation/results from work that still needs to be demonstrated, so the paper does not claim more than the evidence supports.

> **Evidence note:** Project-specific details and metrics below are based on the SEAA project information and Phase 5 reports available in this conversation. Reported results have not all been independently rerun here. Recommended experiments are plans, not completed results.

---

## 1. What exactly is this project?

### The main idea

Modern AI systems often combine different types of information to make a prediction. Such systems are called **multimodal AI systems**.

For example, a movie-genre prediction system can use:

- **Text modality:** The movie's plot description, which contains information about the story, characters, events, and themes.
- **Image modality:** The movie poster, which can contain visual cues related to the genre, such as scenery, characters, and visual style.
- **Combined prediction:** A text model and an image model contribute to one multi-label genre prediction.

Using both modalities can be useful because one source may contain information that the other lacks. However, combining modalities does not guarantee reliability.

- A poster may be blurred or corrupted by noise.
- A plot description may be truncated or have important words masked.
- An entire modality may be missing.
- Both modalities may be corrupted simultaneously.
- The two models may disagree, or a model may be uncertain even when its input is available.

A conventional multimodal model may continue making predictions despite these failures. A fixed recovery rule might respond incorrectly by reducing the contribution of a modality that is actually useful.

**SEAA addresses this problem by introducing an adaptive auditing-and-recovery loop around a multimodal prediction system.**

Instead of always applying the same recovery rule, SEAA observes available signals, diagnoses the likely failure, selects a recovery action, evaluates the outcome during permitted training, and updates its policy.

## 2. The research problem

The central research problem can be stated as follows:

> How can a multimodal AI system adaptively diagnose input reliability failures and select recovery actions that improve predictive performance, while learning from recovery outcomes without using test ground-truth labels during inference?

This problem has five parts.

1. **Monitoring:** Determine whether available text and image signals appear reliable.
2. **Diagnosis:** Identify the likely failure type, such as missing text, image corruption, or cross-modal disagreement.
3. **Recovery:** Choose how the system should combine or select its modalities.
4. **Evaluation:** Measure whether the recovery policy improves performance compared with suitable baselines.
5. **Learning:** Update the policy from permitted training outcomes so later decisions can improve.

The research is not simply about improving movie-genre classification. The classification task is the experimental environment used to test whether adaptive reliability auditing and recovery works.

### Why this matters

The broader motivation applies to systems that use text and images, medical reports and scans, documents and figures, or other combinations of information. In those settings, an unreliable input source can lead to incorrect downstream predictions.

However, the current experiments use MM-IMDb. Therefore, the paper can demonstrate the algorithm on a text-plus-image benchmark, but it should not claim that reliability has already been established for healthcare, autonomous vehicles, or other deployment domains.

## 3. What is new about SEAA?

The intended contribution is the combination of **reliability diagnosis, adaptive recovery, and outcome-based policy learning**.

A normal multimodal classifier produces a prediction. A reliability auditor may detect suspicious behavior. A recovery system may change the modality weights. SEAA combines these operations into a repeated decision process.

### SEAA cycle

1. **Monitor:** Collect modality quality, uncertainty, availability, and agreement signals.
2. **Diagnose:** Estimate the likely reliability failure from observable signals.
3. **Recover:** Choose a fusion, reweighting, or single-modality action.
4. **Evaluate:** Measure prediction quality and recovery improvement in a permitted evaluation setting.
5. **Learn:** Update the recovery policy using the observed reward, then repeat.

The learned policy influences subsequent decisions. During held-out test evaluation, the policy must remain frozen.

### Intended research contributions

A defensible paper could claim the following contributions, provided the implementation and experiments support them:

1. **An adaptive auditing algorithm:** A common pipeline connecting reliability signals, diagnosis, recovery actions, and outcome learning.
2. **A reliability-aware action-selection mechanism:** A policy that chooses among multiple recovery actions rather than relying only on a fixed heuristic.
3. **An outcome-learning mechanism:** A tabular contextual policy whose action values change based on training rewards.
4. **A controlled robustness evaluation:** Experiments across clean inputs, individual modality corruptions, missing modalities, and dual corruption.
5. **A statistically evaluated recovery benefit:** Comparisons with no-recovery and fixed-policy baselines, using movie-level clustering because the same movie appears under multiple conditions.

These are intended contributions, not proof of global novelty. To establish novelty for IEEE publication, compare SEAA against closely related work on multimodal robustness, failure diagnosis, adaptive modality selection, contextual bandits, and test-time recovery.

## 4. Dataset: MM-IMDb

- **Dataset:** MM-IMDb
- **Source:** `pranavmr/MM-IMDb` on Hugging Face
- **Task:** Multi-label movie-genre classification
- **Modalities:** Text and image
- **Genre labels:** 23
- **Records:** 25,959

Each record contains a movie plot description, a poster image, and one or more genre labels. A movie can belong to several genres at the same time, so this is a **multi-label classification** problem rather than a single-class classification problem.

For example, a movie might have both `Drama` and `Romance` labels. The model must predict which genres apply, not just choose one category.

### Dataset preparation reported as completed

| Item | Reported result |
|---|---:|
| Total records | 25,959 |
| Training set | 15,564 |
| Validation set | 2,607 |
| Test set | 7,788 |
| Split proportions | 60% / 10% / 30% |
| Random seed | 42 |
| Genre classes | 23 |
| Missing plot descriptions | 0 |
| Missing or corrupted poster images | 0 |
| Automated dataset checks | 12 passed |

The original Hugging Face repository reportedly provided all records in one partition rather than official train/validation/test partitions. Therefore, the project created its own split using grouped multi-label iterative stratification, adapted from the method of Sechidis and colleagues.

Duplicate plot descriptions were grouped to prevent copies of the same text from leaking across partitions. The project records report 11 duplicate text strings affecting 25 records, with duplicate groups kept together and all three split index sets disjoint.

### Why the split matters

- **Training set:** Fits the baseline predictive models.
- **Validation set:** Supports model and policy development, including policy reward updates and tuning.
- **Test set:** Provides held-out evaluation after decisions are finalized.

The test set must not be used to select recovery actions, tune thresholds, or update policy parameters. Test labels may be used after predictions are complete to calculate evaluation metrics, but not to influence the decisions being evaluated.

The reported split counts and validation results should be included in the paper's dataset section, along with a reproducible description of the splitting procedure and random seed.

## 5. How the prediction system works

SEAA builds on three baseline components.

### 5.1 Text model

- **Input:** Movie plot description.
- **Feature extraction:** TF-IDF.
- **Maximum features:** 3,000.
- **N-grams:** Unigrams and bigrams.
- **Classifier:** Logistic Regression using a one-vs-rest multi-label setup.

TF-IDF converts text into numerical features based on word and phrase importance. Logistic Regression then estimates the presence of each genre.

### 5.2 Image model

- **Input:** Movie poster.
- **Feature extractor:** ResNet-50.
- **Image embeddings:** Frozen ResNet-50 representations.
- **Classifier:** Logistic Regression.

The reported approach uses a pretrained ResNet-50 to extract image features rather than training a complete deep image network from scratch. A classical classifier predicts the genres from these representations.

### 5.3 Multimodal fusion

The text and image models each produce a score for each genre. The project uses late fusion, combining the two prediction-score vectors.

A simplified fusion equation is:

\[
p_{\text{fusion}} =
\alpha p_{\text{text}} +
(1-\alpha)p_{\text{image}}
\]

Here:

- \(p_{\text{text}}\) is the text model's genre-score vector.
- \(p_{\text{image}}\) is the image model's genre-score vector.
- \(\alpha\) controls the contribution of text.
- The initial static fusion uses \(\alpha=0.55\).

The fusion result is converted into binary genre predictions using the project's prediction thresholds.

### Baseline results reported

| Model | Micro-F1 | Macro-F1 |
|---|---:|---:|
| Text-only | 0.6056 | 0.4955 |
| Image-only | 0.4151 | 0.2847 |
| Static late fusion | 0.5161 | 0.4003 |

These are the reported baseline figures. They establish that plot text is more predictive than poster imagery on this dataset and that static fusion does not automatically outperform the text-only model.

This is important to the research: **the contribution is not the assumption that more modalities always improve accuracy. It is the ability to adapt when a modality is unreliable or unhelpful.**

The Phase 5 robustness experiment uses a separate evaluation setup with 500 held-out movie IDs repeated across eight conditions. Its aggregate figures should be reported separately from the original baseline evaluation because the evaluation sample and corruption conditions differ.

## 6. What does SEAA monitor?

SEAA does not directly know the true reliability of a modality at inference time. Instead, it observes measurable signals that can indicate a problem.

The current project describes context features such as:

- Modality presence: whether text, image, or both are available.
- Text and image uncertainty or confidence signals.
- Disagreement between the text and image predictions.
- Diagnostic output indicating a likely failure category.
- Other observable signal summaries used by the policy.

The reported state representation is:

\[
s_t=(P_t,D_t,U_t,G_t)
\]

where:

- \(P_t\): modality-presence state.
- \(D_t\): cross-modal disagreement state.
- \(U_t\): uncertainty asymmetry between modalities.
- \(G_t\): predicted diagnosis category.

The implementation discretizes these signals into categories. For example, disagreement is divided into low and high using a reported Jensen–Shannon divergence threshold of \(0.1139\).

This produces a compact state representation suitable for a tabular policy.

### A crucial distinction

A diagnosis is not necessarily the same as the true cause of failure. For example, high disagreement may be caused by image corruption, an unusual but valid poster, or a genuine difference in what text and images reveal.

Therefore, the paper should evaluate diagnosis quality against known injected corruption conditions, not assume that the diagnostic output is always correct.

For each synthetic test condition, the experimenter knows what corruption was applied. That knowledge can be used to evaluate the diagnosis after the fact, but must not be secretly supplied to the policy during test-time action selection.

## 7. Failure conditions used in the experiment

The Phase 5 evaluation reports eight conditions.

| Condition | What it simulates | Purpose |
|---|---|---|
| `Clean_Baseline` | No input corruption | Test normal-input behavior |
| `Text_Truncation_25%` | Removes part of the text | Test incomplete descriptions |
| `Text_Token_Masking_35%` | Masks text tokens | Test degraded text semantics |
| `Text_Empty` | Removes the text input | Test a missing modality |
| `Image_Gaussian_Blur_r5` | Blurs the poster | Test loss of visual detail |
| `Image_Gaussian_Noise_s25` | Adds Gaussian noise | Test noisy visual inputs |
| `Image_Missing_Blackout` | Blackouts the image input | Test a missing image |
| `Dual_Corruption_Trunc_Noise` | Corrupts both modalities | Test simultaneous failure |

There are 500 movie IDs in the test evaluation, and each is evaluated under all eight conditions. That yields 4,000 movie-condition rows per policy.

This repeated-measures design is useful because the policies face the same movies under the same corruption conditions. It also creates statistical dependence: eight rows from the same movie are not eight independent movies.

## 8. The adaptive recovery policy

This is the central mechanism of the current Phase 5 implementation.

The project reports a contextual-bandit-style policy named `AdaptiveRecoveryPolicy`. It selects among five recovery actions.

| Action | Meaning |
|---|---|
| `STATIC_FUSION` | Use the standard 55% text / 45% image fusion |
| `TEXT_ONLY` | Use the text model without the image model |
| `IMAGE_ONLY` | Use the image model without the text model |
| `RELIABILITY_WEIGHTED` | Adapt the fusion weight according to relative modality reliability signals |
| `CONSERVATIVE_FUSION` | Use a 50% text / 50% image combination |

These actions give the policy multiple options instead of forcing it to use one fixed fusion rule.

For example, when the image is missing, using image-only prediction is not a valid recovery action. The policy must respect the available-modality constraints.

### Reliability-weighted fusion

The documented reliability-weighted action uses:

\[
\alpha(x)=
\operatorname{clip}
\left(
0.55+0.40\tanh\left(3(u_{\text{img}}-u_{\text{text}})\right),
0.05,0.95
\right)
\]

where \(u_{\text{img}}\) and \(u_{\text{text}}\) are the uncertainty-related signals used by the implementation.

The resulting \(\alpha\) determines the contribution of text; the remaining weight goes to the image model.

**Important:** The paper must define exactly what each \(u\) represents. If a larger value means greater uncertainty, the sign of the expression should be checked against the intended behavior. If the signals instead measure reliability, their interpretation is different. Do not describe the formula as uncertainty-aware until the direction and semantics of both inputs have been verified in the code.

### How does the policy learn?

The current design uses a tabular action-value estimate \(Q(s,a)\). This value represents the estimated reward for taking action \(a\) in context \(s\).

The documented update is:

\[
N_{t+1}(s_t,a_t)=N_t(s_t,a_t)+1
\]

\[
\eta_t=
\max\left(
0.05,\frac{0.20}{N_{t+1}(s_t,a_t)^{0.5}}
\right)
\]

\[
Q_{t+1}(s_t,a_t)=
Q_t(s_t,a_t)+
\eta_t\left[R_t-Q_t(s_t,a_t)\right]
\]

Here:

- \(N(s,a)\) counts visits to a state-action pair.
- \(\eta_t\) is the learning rate.
- \(R_t\) is the observed reward.
- \(Q(s,a)\) is the estimated value of an action in a state.

If the observed reward is greater than the current estimate, the value increases. If the reward is lower, the value decreases.

Over time, the policy can prefer actions with better observed rewards for similar contexts.

### Reward function

The documented reward is:

\[
R_t =
F_1(\hat y_{a_t},y_t)
+\lambda\left[
F_1(\hat y_{a_t},y_t)-
F_1(\hat y_{\text{unrecovered}},y_t)
\right]
\]

with \(\lambda=0.5\).

This rewards both predictive quality and improvement relative to the unrecovered prediction.

However, the reward uses the true label vector \(y_t\). This is acceptable during policy training on the designated validation set, where labels are available. It is **not acceptable to use test ground truth to choose actions or update the policy during test evaluation**.

### What “self-evolving” means here

The current implementation's self-evolution claim is that recorded training outcomes update the action values, and those changes can alter subsequent decisions.

The project reports 2,500 validation policy updates, a replay of the update history matching the saved policy within a maximum floating-point difference of approximately \(1.26\times10^{-5}\), and a controlled unit test in which an observed reward sequence changes the chosen action for an identical context.

These results support a limited, precise claim: **the policy adapts its action values from permitted training rewards.**

They do not yet establish that the system safely learns online in a real deployment, generalizes to unseen corruption types, or continually improves under arbitrary distribution shifts.

## 9. What has been implemented so far?

The SEAA framework is implemented as an integrated, unified system with the following core subsystems:

### Subsystem 1 — System Foundation & Verification
Hardware verification, memory management, and library environment checks (`scripts/verify_environment.py`).

### Subsystem 2 — Zero-Leakage Dataset Pipeline
MM-IMDb ingestion, grouped iterative stratified splitting (60/10/30), duplicate plot isolation, and zero-leakage index mappings (`src/data/`).

### Subsystem 3 — Multimodal Baseline Models & Representation Caching
Text-only (TF-IDF + Logistic Regression), Image-only (Standardized ResNet-50 + Logistic Regression), and validation-tuned static late fusion ($\alpha_0=0.55$) with disk-backed memory-mapped visual feature cache (`src/models/`).

### Subsystem 4 — Inline Reliability Auditor & Failure Diagnoser
Multi-label Bernoulli Jensen--Shannon Divergence ($D_{JS}$), predictive binary entropy ($H(\hat{p})$), L2 feature norms, and calibrated failure diagnosis (`src/auditing/`).

### Subsystem 5 — Self-Evolving Contextual Bandit Recovery Engine
Contextual state signature $s_t = (P_t, D_t, U_t, G_t)$, 5 canonical recovery actions with physical feasibility masking, hybrid F1 outcome reward, and frozen zero-leakage test inference (`src/policy/`, `src/recovery/`).

### Subsystem 6 — Repeated-Measures Clustered Statistical Verification & Ablations
Repeated-measures clustered paired $t$-test ($df=499, p=1.34 \times 10^{-32}$), 5,000-iteration cluster bootstrap, full component ablations, and 0.0245 ms decision latency micro-benchmarks (`scripts/`, `outputs/`).

### Subsystem 7 — Master Unified Pipeline Runner
A single command-center entry point (`run_seaa.py` / `main.py`) executing all framework modes end-to-end. The codebase is supported by 73 passing automated unit tests.

Those are reported execution results from the project environment, not tests independently rerun here.

## 10. What results are we trying to achieve?

There are two different questions:

1. What should SEAA achieve scientifically?
2. What has the current experiment actually achieved?

These must not be confused in an IEEE paper.

### 10.1 Expected research outcomes

The project should test whether SEAA can:

- Maintain or improve performance on clean inputs.
- Recover from text corruption and image corruption.
- Handle missing modalities without choosing an impossible action.
- Improve robustness when both modalities are degraded.
- Outperform both the unrecovered baseline and a fixed recovery policy.
- Learn better action values from training outcomes.
- Make decisions without access to test labels.
- Keep its recovery overhead low enough to be practically useful.

These are hypotheses to test, not results to assume in advance.

### 10.2 Results currently reported for Phase 5

The final report uses condition-averaged Micro-F1 as its primary aggregate metric, with globally pooled Micro-F1 as a complementary measure.

**Primary metric — condition-averaged Micro-F1**

- No recovery: **0.4012**
- Fixed recovery policy: **0.3942**
- Adaptive recovery policy: **0.4709**

Reported gains:

- Versus no recovery: **+0.0697**, or +6.97 percentage points (about 17.4% relative).
- Versus fixed policy: **+0.0768**, or +7.68 percentage points (about 19.5% relative).

These are reported results, not independently rerun calculations.

### 10.3 Complete aggregate comparison

| Metric | No recovery | Fixed policy | Adaptive policy |
|---|---:|---:|---:|
| Condition-averaged Micro-F1 | 0.4012 | 0.3942 | **0.4709** |
| Condition-averaged Macro-F1 | 0.3165 | 0.2945 | **0.3593** |
| Globally pooled Micro-F1 | 0.3973 | 0.3905 | **0.4690** |
| Globally pooled Macro-F1 | 0.3030 | 0.2898 | **0.3523** |
| Mean instance-level F1 | 0.3940 | 0.3901 | **0.4671** |

The adaptive policy is reported to outperform both baselines on all five aggregate metrics. The fixed policy is reported to perform worse overall than the no-recovery baseline.

That negative fixed-policy result is scientifically useful: it illustrates why an inflexible recovery rule can damage predictions even when it is designed to improve reliability.

### 10.4 Results by corruption condition

The following table shows reported Micro-F1 results for each of the eight conditions.

| Test condition | No recovery | Fixed | Adaptive |
|---|---:|---:|---:|
| Clean baseline | 0.5187 | 0.4652 | **0.5528** |
| Text truncation | 0.4635 | 0.4432 | **0.4700** |
| Text masking | 0.4619 | 0.4364 | **0.4690** |
| Empty text | 0.3836 | 0.3996 | **0.3996** |
| Image blur | 0.3466 | 0.2272 | **0.4016** |
| Image noise | 0.4121 | 0.3183 | **0.4988** |
| Image blackout | 0.3060 | **0.5820** | **0.5820** |
| Dual corruption | 0.3173 | 0.2813 | **0.3937** |

The reported findings suggest:

- **Clean inputs:** Adaptive recovery retains performance and improves over the unrecovered prediction.
- **Text degradation:** Gains are positive but relatively small.
- **Image corruption:** Recovery is particularly useful under blur and noise.
- **Missing modalities:** The policy respects modality-availability constraints in the reported conditions.
- **Dual corruption:** Adaptive recovery improves performance, but does not restore clean-input performance.

One nuance matters: the adaptive policy does not beat the no-recovery baseline in every condition. Under empty text and image blackout, it ties the fixed policy; the image-blackout fixed and adaptive policies both outperform no recovery. The strongest defensible claim is therefore that adaptive recovery improves the reported aggregate performance and performs positively across the tested conditions—not that it is uniquely best under every individual condition.

### 10.5 How to interpret the metrics

- **Micro-F1** aggregates true positives, false positives, and false negatives across labels. It gives more influence to commonly occurring label decisions.
- **Macro-F1** calculates F1 for each genre and averages across genres, giving rare genres more influence relative to their frequency.
- **Instance-level mean F1** calculates F1 for each movie and averages across movies or movie-condition rows, depending on the specified aggregation.

The paper must explicitly state which aggregation is being used. Condition-averaged Macro-F1 is not interchangeable with globally pooled Macro-F1.

## 11. Statistical significance: what can we claim?

The same 500 movies appear in eight conditions. Therefore, treating all 4,000 rows as independent observations would exaggerate the amount of independent evidence.

The project addresses this by aggregating instance-level F1 differences per movie and testing the resulting 500 movie-level differences.

The statistical report claims:

| Comparison | Mean instance-F1 difference | 95% confidence interval | Reported paired t-test |
|---|---:|---:|---:|
| Adaptive vs no recovery | +0.073120 | [0.061888, 0.084352] | \(p=1.34\times10^{-32}\) |
| Adaptive vs fixed policy | +0.076994 | [0.065194, 0.088795] | \(p=1.02\times10^{-32}\) |

The report also claims that a 5,000-iteration cluster bootstrap was run, resampling movie IDs rather than individual rows, and that all 10 tested metric comparisons had strictly positive differences in every resample.

These results are promising, but the exact statistical terminology matters.

### What needs to be checked before publication

1. **Reproduce the calculations:** Run the supplied scripts against the actual `sample_level_decisions.csv`.
2. **Confirm the cluster unit:** Verify that all eight condition rows for a movie are resampled together.
3. **Check the confidence intervals:** Confirm that the stated intervals use the documented method and the intended cluster-level resampling.
4. **Clarify the bootstrap p-value:** Zero non-positive differences in 5,000 bootstrap iterations means the observed empirical proportion is zero. It is not automatically a formal hypothesis-test p-value. Report the bootstrap procedure precisely, and use a justified test or interval for inferential claims.
5. **Address multiple comparisons:** There are 10 reported metric comparisons. Explain which is primary, which are secondary, and whether multiplicity adjustment is needed for the inferential claims.
6. **Avoid overstating generalization:** The 500 movies provide evidence on this evaluation sample and these synthetic corruption conditions, not proof that the algorithm will generalize to all multimodal systems.

A very small p-value does not establish practical importance, prove the mechanism is correct, or guarantee replication on a different dataset. The paper should report effect sizes and confidence intervals alongside statistical tests.

## 12. How do we demonstrate that the algorithm actually learns?

This is one of the most important questions an IEEE reviewer may ask.

A policy that always chooses `TEXT_ONLY` could perform well on MM-IMDb simply because text is more predictive than posters. Good benchmark performance alone does not prove that the policy is self-evolving.

The current project reports evidence of outcome learning:

- 2,500 training updates on validation trajectories.
- Recorded contexts, actions, rewards, old and new Q-values, and visit counts.
- A serialized learned policy.
- Replay of the update history.
- A controlled test in which reward updates change the selected action for the same context.

That demonstrates that the learning mechanism operates. However, stronger experiments are needed to establish that learning itself is responsible for the measured gains.

### Essential ablation experiments

An ablation removes or disables one component to measure how much it contributes.

| Experiment | What it tests |
|---|---|
| No recovery | Whether any recovery action is useful |
| Fixed heuristic policy | Whether the adaptive method beats a manually designed rule |
| Static fusion only | Whether switching actions helps beyond the initial fusion |
| Adaptive policy with learning disabled | Whether the learned updates add value |
| Adaptive policy with randomized actions | Whether intelligent selection matters |
| Adaptive policy with shuffled rewards | Whether meaningful outcome feedback matters |
| Adaptive policy without diagnosis features | Whether diagnosis signals improve decisions |
| Full SEAA | The complete algorithm |

The most important comparison is **full SEAA versus the same policy with learning disabled**. If they perform identically, the experiments may show that the action space is useful, but not that self-evolution adds value.

You should also test multiple random seeds and, if feasible, more than one held-out test split. A single seed can give a misleading impression of how stable the results are.

## 13. What is still missing or needs further verification?

The current report is a good starting point, but several issues should be addressed before the project is considered ready for a strong IEEE submission.

### 13.1 Independent reproducibility of Phase 5 statistics

The statistical report says its computations were reproduced, but the scripts and source CSV must be run independently to confirm the claims. The files uploaded so far include the policy comparison summary and benchmark JSON, not the complete sample-level decision data and verification scripts.

**Required:** Confirm that every reported result can be regenerated from the stored artifacts.

### 13.2 Oracle baseline

The report lists an oracle upper-bound Micro-F1 of 0.5704 in the condition-averaged summary. However, the sample-level decision CSV reportedly lacks the oracle prediction vectors.

An oracle baseline may be useful, but the paper must define it precisely. Does it select the best action using ground truth for each sample? If so, it is an idealized upper bound, not a deployable policy. Its use of ground truth must be isolated from all non-oracle methods.

**Required:** Verify how the oracle is constructed, how its predictions are generated, and whether the reported aggregate can be reproduced.

### 13.3 Latency claims

The report gives a mean policy decision latency of 0.0112 ms and throughput of approximately 89,385 decisions per second. It also reports an end-to-end overhead estimate of about 0.046 ms per sample.

The report notes that individual per-sample latency values were not saved. That means the current artifacts cannot establish latency variance, tail latency, or percentile behavior.

**Required:** Re-run latency measurements with a documented timing protocol, warm-up, repeat counts, and hardware/software details. Measure the actual end-to-end path and clearly distinguish cached feature inference from full image inference.

### 13.4 Reliability diagnosis accuracy

The paper needs to demonstrate whether the diagnostic module can distinguish text failure, image failure, missing modalities, and dual corruption.

**Required:** Produce a confusion matrix and class-wise precision, recall, and F1 for the diagnosis labels using the known synthetic corruption conditions. Do not assume a failure has been correctly diagnosed merely because the recovery action improved performance.

### 13.5 Threshold selection

The project reports a Jensen–Shannon divergence threshold of 0.1139 and validation-derived diagnostic thresholds.

**Required:** Document how each threshold was selected, whether it was tuned only on training/validation data, and whether it remains fixed during test evaluation. Include sensitivity analysis to show whether the result depends heavily on one threshold.

### 13.6 The “self-evolving” scope

The current policy is trained using validation outcomes and frozen for the test. This is appropriate for a controlled offline evaluation, but it does not yet prove safe online learning after deployment.

**Required:** Describe the current method as validation-trained, outcome-adaptive policy learning. Treat online adaptation under real distribution shift as future work unless you implement and evaluate it.

### 13.7 Generalization beyond MM-IMDb

The dataset has a strong text-versus-image information imbalance: reported text-only Micro-F1 is 0.6056, while image-only Micro-F1 is 0.4151.

That is an important limitation. The policy may learn to prefer text because it is generally stronger in this particular dataset.

**Required:** Test on a second multimodal dataset if feasible, or explicitly limit the current conclusions to MM-IMDb and its tested corruption conditions. A second dataset would materially strengthen the publication.

### 13.8 Bootstrap mean versus point estimate

The corrected report now appropriately avoids claiming that a simple Jensen's inequality argument explains the small differences between direct point estimates and bootstrap mean differences.

Keep that correction. The reported differences are small relative to the bootstrap standard errors, but their precise cause has not been established. Do not invent a mathematical explanation.

## 14. How to make the research suitable for IEEE publication

An IEEE paper must communicate more than a working implementation. It needs a clear research gap, a reproducible algorithm, fair experimental comparisons, statistical evidence, and limitations that are stated honestly.

### Recommended paper structure

**I. Introduction**

Define the multimodal reliability problem, explain why fixed recovery is insufficient, state the research question, and list the contributions.

**II. Related Work**

Compare multimodal fusion, uncertainty and shift detection, failure diagnosis, model recovery, contextual bandits, and adaptive AI auditing. Explain precisely how SEAA differs from each category.

**III. Problem Formulation and System Model**

Define modalities, predictions, reliability signals, states, actions, rewards, update equations, and the assumptions under which the method operates.

**IV. Proposed SEAA Algorithm**

Present the complete Monitor → Diagnose → Recover → Evaluate → Learn pipeline, pseudocode, action constraints, and test-time isolation.

**V. Experimental Setup**

Describe MM-IMDb, data splitting, text/image models, corruption operators, evaluation conditions, baselines, metrics, hardware, and reproducibility settings.

**VI. Results and Statistical Analysis**

Present baseline performance, per-condition results, aggregate metrics, diagnosis accuracy, ablations, latency, confidence intervals, and appropriate significance tests.

**VII. Discussion and Limitations**

Explain why recovery helps or fails, the text/image imbalance, the scope of self-evolution, possible bias, computational costs, and generalization limits.

**VIII. Conclusion and Future Work**

Summarize demonstrated contributions without overclaiming, and identify the next validated research steps.

Follow the target IEEE venue's current template and author instructions. The exact paper length, reference style details, supplementary-material rules, and reproducibility expectations depend on the conference or journal.

### Figures the paper should include

1. **SEAA architecture diagram:** Show the input modalities, predictive models, auditor, diagnosis, recovery policy, reward calculation, and learning loop.
2. **Data preparation and split diagram:** Show the dataset, duplicate grouping, stratification, and final partitions.
3. **Baseline comparison chart:** Compare text-only, image-only, and static fusion.
4. **Per-condition recovery chart:** Compare no recovery, fixed recovery, and adaptive recovery across the eight conditions.
5. **Policy action-distribution heatmap:** Show which actions are selected under each condition.
6. **Ablation chart:** Compare full SEAA against variants with learning, diagnosis, or adaptive action selection disabled.
7. **Statistical confidence-interval plot:** Display effect sizes and confidence intervals across the primary and secondary comparisons.
8. **Optional learning curve:** Plot policy reward or action value over training updates, with an appropriate interpretation.

Only include figures generated from the reproducible project outputs. Every figure should be referenced in the text and discussed, not merely inserted to make the paper look complete.

### Tables the paper should include

- Dataset statistics and split sizes.
- Model and training configuration.
- Main baseline comparison.
- Per-condition performance.
- Overall Micro-F1 and Macro-F1 under clearly defined aggregation methods.
- Diagnosis metrics.
- Ablation results.
- Statistical significance and confidence intervals.
- Runtime and computational overhead.
- Comparison with related methods, where the protocols and metrics are sufficiently comparable.

## 15. What the IEEE reviewer is likely to ask

| Likely reviewer question | Evidence you should provide |
|---|---|
| What is genuinely novel? | A literature-based comparison and a precise statement of the new algorithmic contribution |
| Why is this self-evolving rather than just adaptive routing? | Controlled ablations showing that outcome learning changes decisions and improves held-out performance |
| How do you know diagnosis is correct? | Confusion matrix and class-wise diagnosis metrics |
| Does the method use test labels? | Code-level audit, frozen policy, and reproducible test protocol |
| Why does it outperform fixed recovery? | Same samples, same corruption conditions, and a fair baseline implementation |
| Is the improvement statistically reliable? | Movie-level tests, cluster bootstrap, confidence intervals, and clear p-value interpretation |
| Does it generalize? | Multiple seeds, robustness analysis, and ideally a second dataset |
| Is the runtime practical? | Repeated end-to-end measurements on documented hardware |
| What happens when both modalities fail? | Dual-corruption results and a clear discussion of the recovery ceiling |
| Can others reproduce the work? | Code, configuration, seeds, data-preparation instructions, metrics, and experiment artifacts |

The paper should answer these questions explicitly. A reviewer should not have to infer the algorithm from code or guess how an experiment was run.

## 16. The remaining work — in the correct order

This is a proposed plan; it does not mean these tasks have already been executed.

1. **Complete the Phase 5 evidence audit.** Reproduce the aggregate metrics, clustered tests, bootstrap confidence intervals, and reported p-values from `sample_level_decisions.csv` using the saved verification scripts. Resolve inconsistencies before finalizing the report.
2. **Verify the implementation against the paper.** Confirm that the documented state signals, reliability-weighted fusion formula, action feasibility rules, reward function, update rule, and training/test separation match the actual source code.
3. **Strengthen the experimental comparisons.** Run the learning-disabled and other important ablations. Add repeated seeds where practical and quantify the contribution of the adaptive learning mechanism itself.
4. **Validate diagnosis separately from recovery.** Measure diagnosis accuracy against known corruption conditions. Report whether incorrect diagnoses cause poor recovery decisions.
5. **Improve reproducibility and efficiency measurements.** Confirm the oracle baseline, repeat latency benchmarks, and document all settings required to reproduce the results.
6. **Conduct a novelty and related-work review.** Compare the finalized algorithm with relevant published approaches. Use the project's finalized 15-reference list as the starting point, preserve its reference-numbering constraint, and verify that each citation supports the claim attached to it.
7. **Prepare the IEEE manuscript.** Write the mathematical method, pseudocode, experiments, figures, tables, limitations, and conclusion from verified results. Use the selected venue's current template.
8. **Final reproducibility and submission audit.** Confirm that tables match scripts and data, figure captions match plotted results, citations are accurate, and all claims in the abstract are supported by the main results.

Keep Phase 6 on hold until you approve the Phase 5 evidence audit. That respects the existing phase-by-phase workflow.

## 17. How to explain SEAA to your supervisor

> My research project is titled “A Self-Evolving AI Auditing Algorithm for Adaptive Reliability Diagnosis and Recovery in Multimodal AI Systems,” abbreviated as SEAA.
>
> The main objective is to develop an algorithm that can monitor the reliability of a multimodal AI system, diagnose potential input failures, select an appropriate recovery action, evaluate the outcome, and learn from the observed reward.
>
> I am using the MM-IMDb dataset, which contains movie plot descriptions, poster images, and 23 genre labels. I prepared training, validation, and test partitions while controlling duplicate-text leakage. I then developed text-only, image-only, and static multimodal-fusion baselines.
>
> In the recovery stage, the algorithm uses observable modality signals and diagnostic information to select among five actions: static fusion, text-only, image-only, reliability-weighted fusion, and conservative fusion. The adaptive policy updates its action values using a reward based on prediction quality and improvement over the unrecovered prediction.
>
> The Phase 5 benchmark reports that the adaptive policy achieves a condition-averaged Micro-F1 of 0.4709, compared with 0.4012 for no recovery and 0.3942 for the fixed policy. It also reports improvements in Macro-F1 and across the tested corruption conditions.
>
> The current priority is to independently reproduce the statistical results, validate the diagnosis and learning contributions through ablation experiments, and ensure that the paper's claims match the implementation. The goal is to produce a reproducible IEEE research paper that demonstrates the benefits and limitations of adaptive multimodal reliability recovery without overstating generalization.

## 18. The most important conclusion

SEAA's central research claim is not simply that a new classifier achieves a higher F1 score. It is that **a multimodal system can use reliability-related signals and observed recovery outcomes to adapt its recovery decisions, potentially improving performance when its input modalities are degraded**.

The current Phase 5 report presents encouraging results for that claim. To make the work publication-ready, the strongest next step is to establish that the improvement genuinely comes from the adaptive learning mechanism, reproduce the statistical evidence, measure diagnosis performance, and demonstrate that the method remains reliable beyond one specific test configuration.

No one can guarantee IEEE acceptance in advance. A strong submission depends on demonstrated novelty, rigorous comparisons, reproducible results, and an honest account of limitations—not just a complete report or a large number of passing tests.
