---
title: "Cover letter — PT HM Sampoerna Tbk · Data Science & Analytics"
author: "Ahmad Zulfan (Az)"
date: "April 2026"
requisition: "24383"
---

**Dear Sampoerna Data Science & Analytics Hiring Team,**

I am writing to apply for the Data Science & Analytics position at PT HM Sampoerna Tbk (requisition 24383). I have spent the past several months building a portfolio that maps directly to the skills your JD lists, and I would welcome the chance to walk a hiring manager through it.

My recent research on real-time fraud detection — **StreamRing: A Cascading Architecture for Real-Time Detection of Blockchain Fraud Rings** — was accepted into the **FinDS Workshop at ACM SIGMOD 2026** (Bengaluru, camera-ready in preparation). The system pairs a streaming XGBoost filter with cascading graph-neural-network inference under tight latency budgets (<5 / <50 / <500 ms tiers) and reaches **97.4% precision** with **99.2% Ring Detection Timeliness** across 166 fraud rings on 11.6M transactions. The same statistical-rigour habit — paired t-tests, effect sizes, multiple baselines, ablation studies — runs through the FMCG portfolio described below.

The project is called **SmokeFreeLab** (`github.com/azulcoder/smokefreelab`). It is an end-to-end product-analytics and experimentation playbook for a smoke-free-product e-commerce funnel, built on the GA4 obfuscated sample in BigQuery. Nine executable notebooks cover funnel decomposition, frequentist and Bayesian A/B testing, multi-touch attribution (Markov and Shapley), price elasticity with a cukai scenario, customer lifetime value (BG/NBD plus Gamma-Gamma), and a Bayesian Marketing Mix Model with custom adstock and Hill saturation in PyMC. A Streamlit Experiment Designer — three tabs: Planner, Readout, and Peeking lab — sits on top of the A/B engine, which has 55 tests at 92% coverage. Every notebook closes with a rupiah-framed business impact section.

Mapping the JD against what the repository already ships:

- **A/B testing (frequentist and Bayesian).** Both are implemented side by side. Stakeholders who understand `P(treatment beats control) = 96%` do not always parse `p = 0.04`; the repo defends both.
- **GA4 and Looker.** Notebook 01 pulls directly from the GA4 BigQuery public sample — 354,857 sessions, five-stage funnel decomposition. A Looker Studio spec lives in `dashboards/`.
- **Advanced SQL (CTEs, window functions).** Production-grade BigQuery and Postgres for six years; `sql/01_funnel_decomposition.sql` is representative of the style.
- **Python (Pandas, Scikit-learn, Statsmodels).** The `src/` tree is typed with `mypy --strict`, linted with ruff, CI-green on every commit.
- **Advanced Excel (financial modeling, complex formulas).** Long history of building economic models with Monte Carlo overlays and multi-sheet scenario stacks in Excel.
- **Time-series analysis (Prophet, ARIMA) and growth modeling (S-curves).** Hyperbolic decline fits and Gompertz / logistic family models are part of my daily toolkit; Prophet and ARIMA are within that family.

Your JD also emphasises translating complex findings for senior leadership. The repository ships three artifacts built for that audience: a single-page rupiah-framed one-pager (`reports/executive_onepager.pdf`), a 10-slide deck with speaker notes (`reports/smokefreelab_deck.pptx`), and a Bahasa Indonesia executive summary (`docs/ringkasan_eksekutif.md`). The headline numbers — including an approximate **Rp 27 B per year** incremental CLV on a Sampoerna-scale funnel, and an approximate **−14% volume** response to a 26.7% cukai-driven price rise — all derive from a single arithmetic notebook (`reports/_derivation.ipynb`), so every figure is reproducible.

I am based in Jakarta, bilingual, and available to start within four weeks. I would welcome a conversation about how the toolkit applies to Sampoerna's category economics — the cukai-sensitivity and MMM reallocation work are the two pieces I expect would land first in a Brand Analytics review.

*Hormat saya,*

**Ahmad Zulfan (Az)**
`github.com/azulcoder` · `infoman.xyz123@gmail.com` · Jakarta
Portfolio: `github.com/azulcoder/smokefreelab`
