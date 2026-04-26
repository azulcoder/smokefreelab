---
title: SmokeFreeLab Experiment Designer
colorFrom: indigo
colorTo: gray
sdk: streamlit
sdk_version: 1.56.0
app_file: app.py
pinned: true
license: mit
short_description: Frequentist + Bayesian A/B workbench for an FMCG smoke-free funnel.
---

# SmokeFreeLab — Experiment Designer

Live Streamlit front-end to the [SmokeFreeLab](https://github.com/) A/B framework.

Three tabs:

1. **Planner** — sample size, duration, and power curves for a chosen
   (baseline, MDE, alpha, power, daily traffic).
2. **Readout** — two-proportion z-test, Bayesian posterior, SRM gate, and a
   SHIP / HOLD / ITERATE verdict.
3. **Peeking lab** — empirical demonstration that naive peek-and-stop inflates
   Type-I error, plus the two industry remedies (Bayesian / sequential).

Every number in the app is produced by `smokefreelab.experiment` — the same
library the notebooks use.

## Local dev

```bash
uv sync --all-extras
make run-app
```

## Built on

- The [GA4 obfuscated e-commerce sample](https://developers.google.com/analytics/bigquery/web-ecommerce-demo-dataset).
- A Beta-Binomial conjugate engine (no PyMC — tractable enough to derive on a whiteboard).
- Fleiss (1981) normal-approximation sizing, agreeing with `statsmodels.NormalIndPower`
  to within one unit across the power grid.
