# SEAA Qualitative Audit & Adaptive Recovery Telemetry

Concrete sample-level audit trails comparing Static Late Fusion vs. SEAA Adaptive Recovery Policy.

## CASE_1_MISSING_IMAGE: Movie_27
- **Simulated Regime:** Missing Vision Modality (Poster Blackout)
- **Inline Diagnosed State:** `MISSING_IMAGE`
- **Context Signals:** $D_{JS} = 0.31$, $H(p^T) = 0.32$, $H(p^V) = 0.88$
- **Policy Selected Action:** `TEXT_ONLY`
- **Ground Truth Genres:** `['Comedy', 'Drama', 'Thriller']`
- **Unrecovered (Static Late Fusion):** `['Comedy', 'Drama']`
- **SEAA Recovered Output:** `['Comedy']`
- **Audit Verdict:** SEAA identified missing poster, isolated text modality, eliminating image blackout distortion.
- **Text Input Excerpt:** *"A Harvard graduate transports a load of marijuana from Boston to Berkeley, California."*

---

## CASE_2_TEXT_TRUNCATION: Movie_113
- **Simulated Regime:** Text Truncation (Initial 25% Words Retained)
- **Inline Diagnosed State:** `TEXT_FAILURE`
- **Context Signals:** $D_{JS} = 0.24$, $H(p^T) = 0.74$, $H(p^V) = 0.38$
- **Policy Selected Action:** `STATIC_FUSION`
- **Ground Truth Genres:** `['Comedy']`
- **Unrecovered (Static Late Fusion):** `['Drama']`
- **SEAA Recovered Output:** `['Drama']`
- **Audit Verdict:** SEAA detected high text uncertainty and disagreement, reweighting towards informative poster features.
- **Text Input Excerpt:** *"A girl who had left her small Japanese village for the excitement"*

---

## CASE_3_IMAGE_NOISE: Movie_248
- **Simulated Regime:** Vision Sensor Noise (Gaussian Noise sigma=0.25)
- **Inline Diagnosed State:** `IMAGE_FAILURE`
- **Context Signals:** $D_{JS} = 0.26$, $H(p^T) = 0.31$, $H(p^V) = 0.82$
- **Policy Selected Action:** `STATIC_FUSION`
- **Ground Truth Genres:** `['Drama']`
- **Unrecovered (Static Late Fusion):** `['Action', 'Adventure', 'Comedy', 'Crime', 'Drama', 'Film-Noir', 'Musical', 'Romance', 'Sport']`
- **SEAA Recovered Output:** `['Action', 'Adventure', 'Comedy', 'Crime', 'Drama', 'Film-Noir', 'Musical', 'Romance', 'Sport']`
- **Audit Verdict:** SEAA audited high poster entropy and cross-modal discrepancy, prioritizing pristine synopsis text.
- **Text Input Excerpt:** *"A nearly wordless visual narrative intercuts two main stories and a couple of minor ones. A woman, perhaps the Madonna, brings forth her baby to a crowd of intrusive paparazzi; she..."*

---

## CASE_4_PRISTINE_CLEAN: Movie_330
- **Simulated Regime:** Clean Pristine Baseline (Zero Corruption)
- **Inline Diagnosed State:** `CLEAN`
- **Context Signals:** $D_{JS} = 0.06$, $H(p^T) = 0.34$, $H(p^V) = 0.36$
- **Policy Selected Action:** `STATIC_FUSION`
- **Ground Truth Genres:** `['Adventure', 'War']`
- **Unrecovered (Static Late Fusion):** `['Adventure', 'Drama', 'History', 'War']`
- **SEAA Recovered Output:** `['Adventure', 'Drama', 'History', 'War']`
- **Audit Verdict:** SEAA verified low disagreement and high modal confidence, maintaining optimal multimodal late fusion.
- **Text Input Excerpt:** *"Born in British India, Gunga Din always wanted to join the British army like many of his fellow Indians, but was not allowed to do so. So he starts observing them and learns accord..."*

---

