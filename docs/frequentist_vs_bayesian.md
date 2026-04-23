# Frequentist vs Bayesian A/B testing: when to use which

> Stub. Kept here so the repo structure is visible from day one.

This document compares the two paradigms across six dimensions, with runnable examples from `notebooks/02_ab_framework.ipynb`. Intended length: ~2,500 words, skimmable in 5 minutes.

## Planned outline

### 1. The paradigms in one paragraph each

- **Frequentist:** the data is random, the parameter is fixed. We ask "how unusual is this data if the null is true?" and reject at a pre-registered alpha. Output: p-value, confidence interval.
- **Bayesian:** the data is observed, the parameter is random. We ask "what do we believe about the parameter given the data and our prior?" Output: posterior distribution, credible interval, probability of improvement.

### 2. When each excels

- Frequentist excels at: regulatory or auditable contexts, large sample sizes, well-understood baseline metrics, organizations new to experimentation (fewer moving parts).
- Bayesian excels at: small-sample or early-stopping contexts, hierarchical MVT (pool information across arms), communication to non-statisticians ("87% probability of winning" reads better than "p=0.04"), incorporating domain priors.

### 3. Common pitfalls each paradigm invites

- Frequentist: p-hacking, peeking, multiple-comparison inflation, misinterpreting p-values as posterior probabilities.
- Bayesian: prior hacking, computational non-convergence, overstated credible intervals when the prior is too tight, MCMC mis-diagnosis.

### 4. Worked example

Simulate an SFP onboarding funnel with a 2pp true lift from 18% to 20% baseline. Run both paradigms at n=500, n=5,000, and n=50,000 per arm. Compare:
- Time to decision
- False-positive rate under peeking
- Expected loss under early stopping
- Interpretation readability for a business stakeholder

### 5. The recommendation

For SmokeFreeLab, default to Bayesian with informative priors; fall back to frequentist with alpha-spending sequential testing for regulated or high-stakes decisions. Pre-register both the prior and the alpha-spending curve; deviations must be disclosed.

## References

- Kohavi, Tang, Xu — *Trustworthy Online Controlled Experiments* (2020)
- Gelman et al. — *Bayesian Data Analysis* (3rd ed)
- Deng et al. (Microsoft) — "Data-Driven Metric Development for Online Controlled Experiments" (2016)
