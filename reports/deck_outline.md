---
title: "SmokeFreeLab"
subtitle: "A commercial analytics playbook for a smoke-free product funnel"
author: "Ahmad Zulfan (Az)"
date: "April 2026"
---

# Slide 1 — Cover

**Title:** SmokeFreeLab
**Subtitle:** Commercial analytics for a smoke-free product funnel
**Author:** Ahmad Zulfan (Az) · Jakarta
**Footer:** github.com/azulcoder/smokefreelab · April 2026

**Speaker note (30s):** "Nine notebooks, one repository, one question — how does a data scientist defend an IDR budget reallocation. The deck walks through funnel, experimentation, attribution, MMM, elasticity, and CLV, with every number framed in rupiah."

---

# Slide 2 — The question this repo answers

**Headline:** *Should we spend the next Rp 50 B per quarter the same way we spent the last one?*

**Three sub-questions:**
1. Is the funnel working end-to-end?
2. Which channel carries the incremental conversion?
3. Is the price elastic enough that a cukai hike will drop volume?

**Speaker note (45s):** "Every slide that follows answers one of these three. If a slide does not answer one of them, it is cut."

---

# Slide 3 — The stack at a glance

**Visual:** single-page architecture diagram from `README.md` — GA4 BigQuery → dbt → `src/smokefreelab` → Streamlit + Looker + one-pager PDF.

**Key callouts:**
- Nine executable notebooks, `01_` through `09_`.
- Six library modules under `src/smokefreelab/`.
- One Streamlit app (Experiment Designer, three tabs).
- Quality gate: ruff, black, mypy strict, pytest with ≥ 60% coverage, CI green.

**Speaker note (30s):** "Portfolio quality bar: typed, tested, linted. The bar is the same as production; the domain is synthetic."

---

# Slide 4 — Funnel: where the money leaks

**Visual:** 5-stage GA4 funnel with drop-off (view_item → add_to_cart → …). Worst proportional leak highlighted.

**Numbers from `notebooks/01_eda_ga4_sample.ipynb`:**
- 354,857 sessions (GA4 public sample)
- 20.0% view_item → add_to_cart
- Worst proportional leak approximately 80% at one stage

**Key insight:** stage-level CVR lift economics are asymmetric. A 1 pp lift at stage two is worth roughly 4× a 1 pp lift at stage five — which is where experiments should be deployed first.

**Speaker note (45s):** "Where does the funnel leak, and where do we deploy experiments for maximum IDR return per experiment."

---

# Slide 5 — Experimentation: what to ship when p > 0.05

**Visual:** side-by-side frequentist CI vs Bayesian posterior density from `notebooks/02_ab_framework.ipynb`.

**Three points:**
1. **SRM gate first.** Any experiment failing SRM at α = 0.01 is discarded. No lift can be trusted when assignment is broken.
2. **Frequentist CI plus Bayesian P(T > C).** Stakeholders who have never written a t-test can act on `P(treatment beats control) = 96%`. They often cannot on `p = 0.04`.
3. **Peeking is the real enemy.** Empirical Type-I inflation from 5% to approximately 15% at 10 peeks. Bayesian posteriors are always-valid.

**Screenshot placeholder:** Experiment Designer Readout tab (SHIP banner).

**Speaker note (60s):** "Reframe the conversation from `is p < 0.05` to `how confident are we and at what cost if we are wrong`."

---

# Slide 6 — Attribution: where did the conversion come from?

**Visual:** 2 × 3 grid — six methods (last-click, first-click, linear, time-decay, Markov, Shapley) compared on the same simulated IDR 20 B budget.

**Key insight:** last-click over-credits bottom-funnel channels by approximately 30–40% relative to Shapley. Compensation schemes pegged to last-click systematically underpay brand-building channels.

**Numbers from `notebooks/05_mtattribution.ipynb`:**
- 10,000 simulated user journeys × 5 channels
- Heuristic-vs-Shapley share gap per channel

**Speaker note (45s):** "Attribution is a coalitional game-theory problem. Shapley values are the only credit-allocation scheme satisfying efficiency, symmetry, and null-player axioms simultaneously."

---

# Slide 7 — MMM: how do we reallocate the next Rp 50 B?

**Visual:** three response curves (TV, digital, trade) with current-spend markers from `notebooks/09_mmm.ipynb`.

**Derivation:**
- Current allocation: TV 32% · digital 12% · trade 52% · other 4%.
- Posterior-mean marginal ROI: TV ≈ 2.4× · digital ≈ 1.9× · **trade ≈ 2.7×**.
- **Recommendation:** shift Rp 5 B from TV to trade this quarter.
- **Expected gain:** approximately **Rp 1.4 B** incremental revenue at constant total spend.

**Risk flag:** trade is near its Hill-saturation elbow. Beyond Rp 3 B per week, the marginal return drops sharply. Monitor weekly.

**Speaker note (60s):** "Custom adstock and Hill saturation in PyMC — no black-box package. Every prior is defensible in this room."

---

# Slide 8 — Elasticity: what does the next cukai hike do to volume?

**Visual:** log-log elasticity fit and cukai scenario bar from `notebooks/07_price_elasticity.ipynb`.

**Headline:** Rp 3,000 → Rp 3,800 per stick (+26.7%) is an approximate **−14% volume** response category-wide under hierarchical pooling. The elastic-SKU bucket alone drops about 33%.

**Why hierarchical, not OLS?**
- OLS under-covers on thin SKU panels.
- Category-level pooling stabilises the thin SKU estimates.
- The fixed-effect + bootstrap CI protocol runs in the notebook.

**Speaker note (60s):** "Elasticity is a core number the finance team asks for every time cukai moves. Hierarchical partial pooling is the defensible default."

---

# Slide 9 — CLV: who are the top-decile customers?

**Visual:** Lorenz curve from `notebooks/08_clv_rfm.ipynb`, top decile shaded.

**Headline:** the top 10% of customers by predicted 12-month CLV capture **48%** of aggregate CLV. Gini ≈ 0.62.

**Eleven RFM segments** map one-to-one to retention tactics:
- Champions → loyalty / VIP tier.
- At Risk → win-back campaign, short window.
- Hibernating → reactivation email with bundle discount.
- Lost → stop spending.

**Retention budget:** Rp 2 B per quarter at a 3:1 ROAS cap allocates approximately **73%** of spend to the top two segments — which is also where roughly 80% of the returnable CLV sits.

**Speaker note (45s):** "Flat retention programs waste roughly 85% of spend. This slide justifies cutting the long tail."

---

# Slide 10 — Impact and close

**Headline:** approximately **Rp 27 B per year** incremental CLV on a 250K-monthly-registrant funnel. Full derivation in `reports/_derivation.ipynb`.

**First 90 days in a commercial analytics role:**
1. Wire POS + GA4 + commerce data into the modules above (weeks 1 – 3).
2. Kick off three A/B experiments through the Experiment Designer workflow (weeks 4 – 8).
3. Ship the first rupiah-denominated MMM readout with HDI to Brand Directors (weeks 9 – 12).

**Repo:** `github.com/azulcoder/smokefreelab`.

**Speaker note (60s):** "The Rp 27 B number is conservative. The full sensitivity matrix — 1 pp to 3 pp lift × Rp 225K to Rp 675K LTV — spans Rp 6.75 B to Rp 60 B. The point is not the point estimate; the point is I can tell you where the number falls on that grid given an input I trust."
