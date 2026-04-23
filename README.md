# SmokeFreeLab

A Bayesian and frequentist A/B testing framework for a simulated smoke-free product (SFP) e-commerce funnel, built on the Google Analytics 4 obfuscated sample dataset in BigQuery.

[![CI](https://github.com/azulcoder/smokefreelab/actions/workflows/ci.yml/badge.svg)](https://github.com/azulcoder/smokefreelab/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

SmokeFreeLab is an end-to-end product-analytics and experimentation playbook for a smoke-free product funnel. Nine executable notebooks cover funnel decomposition, frequentist and Bayesian A/B testing, multi-touch attribution, price elasticity, customer lifetime value, and marketing mix modeling. Every notebook closes with a rupiah-denominated business impact section.

The live demo is an Experiment Designer built in Streamlit with three tabs: Planner, Readout, and Peeking lab. Sizing, SRM gating, frequentist confidence intervals, and Bayesian posteriors sit side by side.

## Executive artifacts

| Artifact | What it is |
|---|---|
| [Executive one-pager (PDF)](reports/executive_onepager.pdf) | Single-page rupiah-framed summary. |
| [10-slide deck (PPTX)](reports/smokefreelab_deck.pptx) | Narrative deck with speaker notes. |
| [Derivation notebook](reports/_derivation.ipynb) | Source of truth for every headline IDR number. |
| [Ringkasan eksekutif (Bahasa)](docs/ringkasan_eksekutif.md) | One-page Indonesian summary. |

## Streamlit Experiment Designer

Run locally with `make run-app`. Screenshots below are captured against the live app; the capture guide lives at [docs/screenshots/README.md](docs/screenshots/README.md).

| Planner | Readout | Peeking lab |
|---|---|---|
| ![Planner — sample size + power curve](docs/screenshots/01_planner_hero.png) | ![Readout — Bayesian posteriors](docs/screenshots/04_readout_bayesian.png) | ![Peeking lab — Type-I inflation](docs/screenshots/06_peeking_after.png) |
| Sample sizing on (baseline, MDE, alpha, power) with a sample-size vs MDE curve at 80 / 90 / 95 percent power. | Two-proportion z-test plus Beta-Binomial posterior with SHIP / HOLD / ITERATE verdict, gated on SRM. | Empirical Type-I rate under peek-and-stop. |

Deployment is scripted at `deploy/hf_space/` ([see `DEPLOY.md`](deploy/hf_space/DEPLOY.md)).

## Business impact

| Metric | Before | After |
|---|---|---|
| Experiment cycle time | 6 – 8 weeks | 2 – 3 weeks |
| False-positive rate under peeking | ~25% unadjusted | <5% with sequential correction |
| Stakeholder decision | Post-hoc only | Pre-registered, real-time posterior |
| MDE transparency | Implicit | Explicit per-experiment with power curves |

Extrapolated to a 250K monthly-registrant funnel at Rp 450K LTV per activated user, a 2 pp activation lift is worth approximately **Rp 27 B per year** in incremental CLV. The derivation and sensitivity grid live in `reports/executive_onepager.pdf` and `reports/_derivation.ipynb`.

## Featured deliverables

| # | Deliverable | Where |
|---|---|---|
| 1 | Bayesian + frequentist A/B framework | `src/smokefreelab/experiment/`, notebooks `03` and `04` |
| 2 | Multi-touch attribution (Markov + Shapley) | `src/smokefreelab/attribution/`, notebook `05` |
| 3 | Price elasticity (log-log + hierarchical Bayes) | `notebooks/07_price_elasticity.ipynb` |
| 4 | CLV + RFM segmentation (BG/NBD + Gamma-Gamma) | `notebooks/08_clv_rfm.ipynb` |
| 5 | Marketing Mix Modeling (Bayesian adstock + Hill) | `notebooks/09_mmm.ipynb` |

Each notebook ends with a rupiah-framed business impact section. See `reports/executive_onepager.pdf` and `reports/smokefreelab_deck.pptx` for the consolidated executive narrative, and `docs/ringkasan_eksekutif.md` for the Bahasa Indonesia summary.

## Questions this project answers

1. Where in the funnel are we losing users? (funnel decomposition, drop-off, cohort retention)
2. Should we launch this feature? (frequentist A/B, Bayesian A/B, MVT, power analysis, pre-registration)
3. How do we attribute conversions across channels? (Markov chains, Shapley values, data-driven attribution)
4. Who will convert and who will churn? (propensity and survival models via XGBoost + lifelines)
5. What is the plain-language business story? (executive one-pager, Looker Studio dashboard)

## Architecture

```
GA4 BigQuery sample (public)
        │
        ▼
   dbt staging & marts  ──►  SQL layer (CTEs, windows)
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                   src/smokefreelab/                           │
│                                                                │
│   analytics/     ─► funnel, cohort, RFM, stickiness           │
│   experiment/    ─► frequentist + Bayesian + MVT + designer   │
│   attribution/   ─► Markov + Shapley multi-touch               │
│   features/      ─► feature engineering for propensity models │
└───────────────────────────────────────────────────────────────┘
        │                          │
        ▼                          ▼
 Streamlit app              Looker Studio dashboard
 (Experiment Designer)      (executive KPIs)
        │
        ▼
 Executive one-pager PDF
```

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/azulcoder/smokefreelab.git
cd smokefreelab
uv sync

# 2. Set up BigQuery Sandbox (free tier, no credit card)
gcloud auth application-default login
cp .env.example .env  # add your GCP project id

# 3. Run the smoke test
make test

# 4. Launch the Experiment Designer
streamlit run app/experiment_designer.py

# 5. Open the notebooks
jupyter lab notebooks/
```

## Repo structure

```
smokefreelab/
├── src/smokefreelab/       # importable library
│   ├── analytics/          # funnel, cohort, RFM, stickiness
│   ├── experiment/         # frequentist + Bayesian A/B + MVT
│   └── attribution/        # Markov + Shapley
├── sql/                    # standalone BigQuery queries
├── notebooks/              # narrative analysis, numbered 01 – 09
├── app/                    # Streamlit Experiment Designer
├── dashboards/             # Looker Studio spec + screenshots
├── reports/                # executive one-pager + deck
├── docs/                   # mkdocs-material site
└── tests/                  # pytest suite
```

## Skills demonstrated

| Skill | Where |
|---|---|
| Advanced SQL (CTEs, window functions) | `sql/01_funnel_decomposition.sql`, `sql/02_cohort_retention.sql` |
| Python (Pandas, Scikit-learn, Statsmodels) | `src/smokefreelab/` |
| A/B testing (frequentist + Bayesian) | `experiment/frequentist.py`, `experiment/bayesian.py`, notebook `03` |
| Multivariate testing | `experiment/designer.py` with Bonferroni / Holm correction |
| Sample size and power calculation | `experiment/power.py`, Streamlit Experiment Designer |
| GA4 proficiency | BigQuery queries against `ga4_obfuscated_sample_ecommerce` |
| Looker & data storytelling | `dashboards/looker_studio_spec.md` |
| Product analytics | `analytics/funnel.py`, `analytics/cohort.py` |
| Marketing attribution | `attribution/markov.py`, `attribution/shapley.py` |
| Executive communication | `reports/executive_onepager.pdf` |

## Further reading

- `docs/frequentist_vs_bayesian.md` — when to use which, with simulated examples
- `docs/experiment_protocol.md` — pre-registration template used across notebooks

## Author

**Ahmad Zulfan (Az)** · Jakarta · bilingual (Indonesian / English)

[LinkedIn](https://www.linkedin.com/in/ahmadzulfan) · [GitHub](https://github.com/azulcoder) · [Azul Analysis](https://azulanalysis.com)
