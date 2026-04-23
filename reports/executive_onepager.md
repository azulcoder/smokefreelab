---
title: "SmokeFreeLab — Executive one-pager"
author: "Ahmad Zulfan (Az)"
date: "April 2026"
geometry: margin=1in
fontsize: 11pt
---

# SmokeFreeLab

**One line.** An end-to-end product-analytics and experimentation stack for a smoke-free-product e-commerce funnel, built on the GA4 obfuscated sample in BigQuery.

**Why it matters.** A disciplined experimentation plus attribution plus MMM stack is worth approximately **Rp 27 B per year** on a 250K-monthly-registrant funnel. Derivation below.

---

## Five headline numbers

| # | Number | Source |
|---|---|---|
| 1 | **Rp 27 B / year** incremental CLV | 250,000 monthly registrants × 12 months × 2 pp activation lift × Rp 450,000 LTV / activated user. `notebooks/06_business_case.ipynb`. |
| 2 | **Rp 50 B / quarter** media budget | TV Rp 1.25 B/wk + digital Rp 450 M/wk + trade Rp 2 B/wk × 13 weeks. `notebooks/09_mmm.ipynb`. |
| 3 | **Rp 1.4 B / quarter** from TV → trade reallocation | Posterior-mean marginal ROI gap × Rp 5 B reallocation, saturation-adjusted. `notebooks/09_mmm.ipynb`. |
| 4 | **48% of CLV** in the top decile | Lorenz curve on a 1,500-customer BG/NBD + Gamma-Gamma simulation. `notebooks/08_clv_rfm.ipynb`. |
| 5 | **~ −14% volume** on a cukai hike | Hierarchical price elasticity × 26.7% price rise (Rp 3,000 → Rp 3,800 / stick). `notebooks/07_price_elasticity.ipynb`. |

Every number reproduces from `reports/_derivation.ipynb`.

---

## What the repo demonstrates

1. **A/B testing** — frequentist and Bayesian side by side, plus SRM gate, power curves, and a peeking demonstration. 55 tests at 92% coverage. Streamlit Experiment Designer (Planner / Readout / Peeking lab).
2. **Multi-touch attribution** — Markov removal-effect, exact Shapley, and four heuristics. `attribution/{heuristics,markov,shapley}.py`, 24 tests.
3. **Propensity with SHAP and calibration** — XGBoost + SHAP + reliability curve. `features/propensity.py`.
4. **Price elasticity** — log-log OLS and Bayesian hierarchical-by-category, with bootstrap CIs and a cukai scenario. `analytics/elasticity.py`.
5. **CLV + RFM** — BG/NBD + Gamma-Gamma, eleven canonical RFM segments, Lorenz curve and Gini. `analytics/clv.py`.
6. **Marketing Mix Modeling** — Bayesian custom adstock + Hill saturation in PyMC, per-channel response curves, budget reallocation. `attribution/mmm.py`.

Nine executable notebooks (`01_` through `09_`) cover the full narrative. Every notebook closes with a rupiah-framed business impact paragraph.

---

## How to read this repo in five minutes

1. `README.md` — hero table and quickstart.
2. `notebooks/06_business_case.ipynb` — the Rp 27 B derivation.
3. `notebooks/09_mmm.ipynb` — the MMM with adstock and Hill.
4. `make run-app` — the Experiment Designer in your browser.
5. `docs/ringkasan_eksekutif.md` — the same narrative in Bahasa Indonesia.

---

## Contact

Ahmad Zulfan (Az) · Jakarta · bilingual (Indonesian / English)
GitHub: [github.com/azulcoder](https://github.com/azulcoder) · Repo: [github.com/azulcoder/smokefreelab](https://github.com/azulcoder/smokefreelab)

---

*Simulation disclosure: the GA4 public sample is real; price-elasticity panels, CLV cohorts, and MMM media spend are synthetic but calibrated to resemble the Indonesian SFP market. The artifact is the methodology, not a claim of ground-truth recovery.*
