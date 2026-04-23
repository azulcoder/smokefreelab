# Experiment pre-registration protocol

> Every experiment run in SmokeFreeLab is pre-registered using the template below. Deviations between the plan and the execution must be disclosed in the post-hoc analysis.

---

## Template

### 1. Experiment metadata

| Field | Value |
|---|---|
| Experiment ID | EXP-YYYY-NNN |
| Name | |
| Owner | |
| Date registered | |
| Planned start | |
| Planned end | |
| Status | Draft / Active / Complete / Cancelled |

### 2. Business context (2–4 sentences)

What business problem prompted this experiment? Reference the funnel stage, revenue line, or strategic initiative it supports. End with the decision that will be made from the result.

### 3. Hypothesis

State in the form: **"We believe that [change] will cause [metric] to [direction] by [magnitude] for [segment], because [mechanism]."**

### 4. Variants

| Arm | Description | Expected traffic share |
|---|---|---|
| Control | Current experience | 50% |
| Treatment | | 50% |

### 5. Metrics

| Role | Metric | Definition | Direction |
|---|---|---|---|
| Primary | | | ↑ / ↓ |
| Guardrail | | Must not degrade by > X% | neutral |
| Secondary | | | ↑ / ↓ |

At most one primary metric. Pre-registering two primary metrics is a red flag.

### 6. Design parameters

| Parameter | Value |
|---|---|
| Paradigm | Frequentist / Bayesian |
| Baseline conversion rate | X% |
| Minimum detectable effect (MDE) | Xpp (absolute) or X% (relative) |
| Alpha | 0.05 (default) |
| Power | 0.80 (default) |
| Sample size per arm | n = |
| Expected duration | D days |
| Randomization unit | user_id / session_id / device |
| Traffic allocation | % of eligible users |

### 7. Bayesian specifics (if applicable)

| Parameter | Value |
|---|---|
| Prior on control CVR | Beta(α, β) — with justification |
| Prior on lift | Normal(μ, σ) — with justification |
| Stopping rule | Expected loss < threshold |
| MCMC sampler | NUTS (default) |
| Target posterior ESS | ≥ 1000 |

### 8. Peeking policy

Select one:
- **No peeking until n_planned reached** (strict; use for high-stakes decisions)
- **Alpha-spending with Pocock/O'Brien-Fleming boundaries** (frequentist sequential)
- **Bayesian continuous monitoring with expected-loss stopping**

Peeking without a pre-registered rule inflates false positives and invalidates the experiment.

### 9. Guardrail tripwires

Automatic stop conditions. Example:
- Page load time increases by > 200ms at p95
- Revenue/user drops > 5% with 95% confidence
- Crash rate exceeds 0.5%

### 10. Exclusions and segments

| Segment | Include / Exclude | Reason |
|---|---|---|
| Internal users | Exclude | Not representative |
| Bots (as flagged by GA4) | Exclude | Non-human |
| < 18 y.o. (age-gated SFP) | Exclude | Compliance |

### 11. Analysis plan

Bullet the exact analysis to be run. No post-hoc slicing without disclosure.

- [ ] Primary hypothesis test (frequentist or Bayesian)
- [ ] Guardrail checks
- [ ] CUPED variance reduction using pre-experiment covariate X
- [ ] Sensitivity analysis: redo with only users who had ≥ N sessions
- [ ] Segment analysis: mobile vs desktop, new vs returning

### 12. Decision rules

State explicitly what result leads to what action. Example:

| Outcome | Decision |
|---|---|
| Primary metric lifts ≥ MDE, all guardrails pass | Launch to 100% |
| Primary metric lifts < MDE, all guardrails pass | Do not launch; iterate |
| Any guardrail tripped | Do not launch; diagnose |
| Inconclusive (wide CI spanning zero) | Extend duration OR rerun with larger traffic |

### 13. Deviations log

Any change to this pre-registration after experiment start must be logged here with timestamp and reason. An empty deviations log is a credibility signal.

---

## Why pre-register?

Because experiments without pre-registration are indistinguishable from storytelling. Anyone can find a p<0.05 result somewhere in a dataset if they slice enough ways. Pre-registration is how we separate "we tested our hypothesis" from "we found something that looked significant."

This discipline is the single clearest signal of statistical maturity in a portfolio and what commercial leadership needs to trust experiment-driven decisions at scale.
