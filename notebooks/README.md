# notebooks/

Notebooks are numbered by session. Each notebook:

- Is reproducible top-to-bottom from a clean clone (`make install-dev`).
- Loads data via the library in `src/smokefreelab/` — never inline SQL.
- Ends with a **Business Impact** section framed in rupiah or percentage points.
- Calls out at least two caveats (sampling, bot traffic, survivorship, etc.).

## Planned notebooks

| # | Notebook | Session | Status |
|---|---|---|---|
| 01 | `01_eda_ga4_sample.ipynb` — GA4 sample EDA + funnel analysis | 1 | ⏳ |
| 02 | `02_funnel_analysis.ipynb` — cohort retention, DAU/MAU stickiness | 1 | ⏳ |
| 03 | `03_ab_testing_framework.ipynb` — frequentist A/B walkthrough | 2 | ⏳ |
| 04 | `04_bayesian_experimentation.ipynb` — Bayesian A/B walkthrough | 2 | ⏳ |
| 05 | `05_mtattribution.ipynb` — Markov + Shapley attribution | 4 | ⏳ |
| 06 | `06_business_case.ipynb` — executive synthesis, the money notebook | 4 | ⏳ |

## Convention

When a notebook requires a new library not in `pyproject.toml`, add it to `pyproject.toml` and run `uv sync` — never `!pip install` inside a notebook.

Strip outputs before committing (`nbstripout` handles this automatically via pre-commit).
