# SEAA Phase 5 Research Protocol: Adaptive Recovery Policy & Evaluation

**Project:** A Self-Evolving AI Auditing Algorithm for Adaptive Reliability Diagnosis and Recovery in Multimodal AI Systems (SEAA)  
**Document:** Phase 5 Formal Research Specification & Protocol  
**Location:** [`docs/phase5/phase5_research_protocol.md`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/docs/phase5/phase5_research_protocol.md)  

---

## 1. Research Question

> **Primary Research Question:**  
> *Does an outcome-learning recovery policy select recovery actions more effectively than a fixed recovery policy under multimodal input failures, while preserving performance on clean inputs?*

### Specific Hypotheses:
1. **$H_1$ (Superiority under Asymmetric Failures):** An adaptive policy that updates its action values based on observed recovery outcomes will learn to route degraded inputs (e.g., corrupted images, corrupted text) to the uncorrupted modality, achieving higher recovery Micro-F1 and Macro-F1 than static late fusion.
2. **$H_2$ (Preservation of Clean Performance):** Under clean multimodal inputs, an outcome-learning policy will converge to selecting cooperative multimodal fusion ($\alpha = 0.55$), preserving performance without introducing unnecessary single-modality fallbacks.
3. **$H_3$ (Demonstrated Self-Evolution):** In controlled sequential decision-making, an observed recovery outcome will update the policy's internal state such that a subsequent identical context transitions its action selection from a sub-optimal choice to an optimal choice.

---

## 2. Decision Space & Recovery Actions

Let $\mathcal{A} = \{a_0, a_1, a_2, a_3, a_4\}$ denote the set of supported recovery actions:

| Action ID | Action Name | Mathematical Formulation | Description & Feasibility Condition |
|---|---|---|---|
| **$a_0$** | `STATIC_FUSION` | $P_{\text{rec}} = 0.55 \cdot P_{\text{txt}} + 0.45 \cdot P_{\text{img}}$ | Baseline Phase 3 late fusion. Feasible only when both text and image modalities are present. |
| **$a_1$** | `TEXT_ONLY` | $P_{\text{rec}} = P_{\text{txt}}$ ($\alpha = 1.0$) | Fallback routing 100% weight to text. Feasible only when text is available (`has_text=True`). Infeasible if text is missing. |
| **$a_2$** | `IMAGE_ONLY` | $P_{\text{rec}} = P_{\text{img}}$ ($\alpha = 0.0$) | Fallback routing 100% weight to image. Feasible only when image is available (`has_image=True`). Infeasible if image is missing/blank. |
| **$a_3$** | `RELIABILITY_WEIGHTED` | $P_{\text{rec}} = \alpha(x) \cdot P_{\text{txt}} + (1 - \alpha(x)) \cdot P_{\text{img}}$ | Dynamic gating weight $\alpha(x) \in [0.05, 0.95]$ computed from continuous uncertainty difference and norm ratio: $\alpha(x) = \text{clip}\left(0.55 + 0.5 \tanh(u_{\text{img}} - u_{\text{txt}}), 0.05, 0.95\right)$. Feasible when both modalities are present. |
| **$a_4$** | `CONSERVATIVE_FUSION` | $P_{\text{rec}} = 0.50 \cdot P_{\text{txt}} + 0.50 \cdot P_{\text{img}}$ | Equal-weight conservative fusion for ambiguous or high-entropy situations. Feasible when both modalities are present. |

### Admissibility Masking ($\mathcal{A}_{\text{valid}}(s)$):
An action requiring an unavailable modality is strictly inadmissible. If an action $a \notin \mathcal{A}_{\text{valid}}(s)$, its selection probability is masked to 0 ($Q(s, a) = -\infty$). If both modalities are unavailable, the policy safely falls back to global dataset genre priors.

---

## 3. Observable Signals at Decision Time (Context Space $\mathcal{S}$)

At decision time $t$, the policy observes **strictly inference-time signals** produced by the auditor and unimodal models. Ground-truth genre labels $y$ and ground-truth corruption flags are **never accessible** at decision time.

### Vector of Observable Features $\mathbf{x} \in \mathbb{R}^d$:
1. $u_{\text{txt}} \in [0, 1]$: Mean predictive binary entropy of the text model over 23 genres.
2. $u_{\text{img}} \in [0, 1]$: Mean predictive binary entropy of the image model over 23 genres.
3. $d_{\text{JSD}} \in [0, 1]$: Multi-label Bernoulli Jensen-Shannon Divergence between text and image predictions.
4. $d_{\text{Hamming}} \in [0, 1]$: Fraction of disagreeing thresholded genre predictions.
5. $n_{\text{ratio}} \in \mathbb{R}^+$: Ratio of text feature L2-norm to image feature L2-norm.
6. $\hat{y}_{\text{diag}} \in \{\text{CLEAN}, \text{TEXT\_FAILURE}, \text{IMAGE\_FAILURE}, \text{CROSS\_MODAL\_CONFLICT}, \text{HIGH\_UNCERTAINTY}, \dots\}$: Categorical diagnosis produced by the auditor / diagnostic classifier.
7. $b_{\text{text}}, b_{\text{image}} \in \{0, 1\}$: Modality presence indicators.

### Discretized Context State $s \in \mathcal{S}$:
To guarantee full interpretability, rigorous convergence, and transparent policy introspection, the context is partitioned into discrete reliability regimes:
$$s = \langle \text{presence}, \text{disagreement}, \text{uncertainty\_balance}, \text{diagnosis} \rangle$$
- **Presence:** `BOTH_AVAILABLE`, `TEXT_ONLY_AVAILABLE`, `IMAGE_ONLY_AVAILABLE`.
- **Disagreement:** `LOW_DISAGREEMENT` ($d_{\text{JSD}} \le \tau_{\text{JSD}}$) vs `HIGH_DISAGREEMENT` ($d_{\text{JSD}} > \tau_{\text{JSD}}$).
- **Uncertainty Balance:** `TEXT_CONFIDENT` ($u_{\text{img}} - u_{\text{txt}} > 0.10$), `IMAGE_CONFIDENT` ($u_{\text{txt}} - u_{\text{img}} > 0.10$), `BALANCED` ($|u_{\text{txt}} - u_{\text{img}}| \le 0.10$).
- **Diagnosis:** High-level auditor diagnosis string.

---

## 4. Outcome Definition & Utility Function (Reward $R$)

After an action $a$ is executed and produces recovered prediction $\hat{y}_a \in \{0, 1\}^C$:
During policy training or offline outcome-learning (on the **validation split**), the outcome is evaluated against ground truth labels $y$:

### Multi-Label Instance Utility Function:
$$F_1(\hat{y}_a, y) = \frac{2 \sum_{c=1}^C \hat{y}_{a, c} y_c}{2 \sum_{c=1}^C \hat{y}_{a, c} y_c + \sum_{c=1}^C \hat{y}_{a, c}(1 - y_c) + \sum_{c=1}^C (1 - \hat{y}_{a, c})y_c + \epsilon}$$

### Reward Formulation:
To encourage both absolute predictive accuracy and positive recovery improvement over the unrecovered baseline $\hat{y}_{\text{static}}$:
$$R_t = F_1(\hat{y}_{a_t}, y_t) + \lambda \cdot \left[ F_1(\hat{y}_{a_t}, y_t) - F_1(\hat{y}_{\text{static}}, y_t) \right]$$
Setting $\lambda = 0.5$ balances absolute performance retention on clean data with maximal recovery gain on corrupted data. Bounded within $[-0.5, 1.5]$.

---

## 5. Policy Architecture & Learning Rule

We implement an **Adaptive Contextual Bandit Policy with Exponential Moving Average Action-Value Updating** and **Upper Confidence Bound (UCB) / $\epsilon$-greedy exploration**.

### 5.1 Internal State Representation:
The policy maintains:
1. $Q(s, a) \in \mathbb{R}$: Estimated expected utility for taking action $a$ in context state $s$.
2. $N(s, a) \in \mathbb{N}$: Number of times action $a$ has been selected under context $s$.
3. $\mathcal{H}_t = \{(s_\tau, a_\tau, R_\tau, Q_\tau(s, a))\}_{\tau=1}^t$: Complete audit trajectory of context, action, reward, and parameter updates.

### 5.2 Action Selection Rule:
In **Training Mode** (exploration enabled, $\epsilon > 0$):
$$a_t = \begin{cases} \text{Uniform}(\mathcal{A}_{\text{valid}}(s_t)) & \text{with probability } \epsilon \\ \arg\max_{a \in \mathcal{A}_{\text{valid}}(s_t)} \left[ Q(s_t, a) + c \sqrt{\frac{\ln \sum_{a'} N(s_t, a')}{N(s_t, a) + 1e-5}} \right] & \text{with probability } 1 - \epsilon \end{cases}$$

In **Evaluation Mode** (zero exploration, deterministic, frozen policy):
$$\epsilon = 0, \quad c = 0 \implies a^* = \arg\max_{a \in \mathcal{A}_{\text{valid}}(s_t)} Q(s_t, a)$$
*(Ties are broken deterministically in favor of the safest action according to modality availability).*

### 5.3 Outcome-Learning Update Rule:
Upon observing reward $R_t$ for action $a_t$ in context $s_t$:
$$N_{t+1}(s_t, a_t) = N_t(s_t, a_t) + 1$$
$$\alpha_t = \max\left(\alpha_{\text{min}}, \frac{1}{N_{t+1}(s_t, a_t)^\beta}\right)$$
$$Q_{t+1}(s_t, a_t) = Q_t(s_t, a_t) + \alpha_t \cdot \left( R_t - Q_t(s_t, a_t) \right)$$
where $\beta \in [0.5, 1.0]$ is the decay rate (or constant step size $\alpha_{\text{lr}} = 0.10$).

---

## 6. Baselines for Comparison

Across all experimental conditions, exactly **four policies** will be benchmarked on identical test sample IDs:

1. **No Recovery Baseline (`NO_RECOVERY`):**
   - Applies static late fusion ($P_{\text{fused}} = 0.55 P_{\text{txt}} + 0.45 P_{\text{img}}$) uniformly across all inputs, regardless of corruptions or diagnostic flags.
2. **Fixed Recovery Policy Baseline (`FIXED_POLICY`):**
   - Employs the existing hardcoded heuristic router ([`DynamicModalityGating`](file:///c:/Users/bharat%20kumar/OneDrive/Desktop/SEAA/src/recovery/modality_gating.py)) without any learning or parameter updates.
3. **Adaptive / Outcome-Learning Recovery Policy (`ADAPTIVE_POLICY`):**
   - The proposed Contextual Bandit policy trained on validation failure/recovery trajectories and frozen during test evaluation.
4. **Oracle Upper Bound (`ORACLE`):**
   - Evaluates all 5 candidate actions against ground-truth labels for each test sample and selects the action achieving highest individual F1. Clearly documented as an unattainable theoretical upper bound.

---

## 7. Experimental Protocol & Split Separation

### Strict Split Isolation:
- **Training / Policy Calibration:**
  - Carried out strictly on the **Validation Split** (2,607 samples, `splits['val']`).
  - Synthetic corruptions (text truncation, token masking, Gaussian blur, Gaussian noise, missing text, missing image) are generated on validation samples to generate training trajectories for the contextual bandit.
  - Test set data and test ground truth labels are strictly inaccessible.
- **Evaluation:**
  - Carried out on the **Test Split** (500 representative test samples with identical IDs across all conditions).
  - The policy is frozen in evaluation mode (`learning_enabled=False`, $\epsilon=0$).
  - Results are measured condition-by-condition and compared sample-by-sample against identical sample IDs.
- **Output Isolation:**
  - All Phase 5 outputs will be written exclusively to a fresh directory: `outputs/phase5/<unique_run_id>/`.
  - Phase 1–3 and Phase 4 artifacts remain untouched.
