"""SmokeFreeLab Experiment Designer — Streamlit front-end to the A/B engine.

Three tabs wrap the library in a PM-friendly surface:

1. **Planner** — sample size, duration, power curve for a chosen (baseline,
   MDE, alpha, power, daily traffic).
2. **Readout** — two-proportion frequentist + Bayesian readout with SRM
   gate and a SHIP / HOLD / ITERATE verdict.
3. **Peeking Lab** — interactive demonstration that fixed-horizon frequentist
   tests cannot be monitored without inflating Type-I error.

No new statistics live here. Every calculation is delegated to
``smokefreelab.experiment`` so the notebook and the app stay in lockstep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from smokefreelab.analytics.viz import (
    COLOR_ACCENT,
    COLOR_GRID,
    COLOR_MUTED,
    COLOR_NEGATIVE,
    COLOR_POSITIVE,
    COLOR_PRIMARY,
    COLOR_TEXT,
    FONT_FAMILY,
    apply_sfl_theme,
)
from smokefreelab.experiment import (
    ArmStats,
    bayesian_test,
    check_srm,
    experiment_duration_days,
    frequentist_test,
    sample_size_per_arm,
    simulate_peeking_inflation,
)

# =====================================================================
# Page config & global styling
# =====================================================================


st.set_page_config(
    page_title="SmokeFreeLab Experiment Designer",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "About": (
            "SmokeFreeLab — a frequentist + Bayesian A/B workbench for an FMCG "
            "smoke-free product funnel. Built on the GA4 obfuscated sample."
        ),
    },
)


CUSTOM_CSS = f"""
<style>
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 1200px;
    }}
    h1, h2, h3 {{
        color: {COLOR_TEXT};
        font-family: {FONT_FAMILY};
        letter-spacing: -0.01em;
    }}
    h1 {{ font-weight: 600; }}
    .sfl-caption {{
        color: {COLOR_MUTED};
        font-size: 0.95rem;
        margin-top: -0.5rem;
    }}
    .stMetric {{
        background: white;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        border: 1px solid {COLOR_GRID};
    }}
    [data-testid="stMetricLabel"] {{
        color: {COLOR_MUTED};
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    [data-testid="stMetricValue"] {{
        color: {COLOR_PRIMARY};
        font-weight: 600;
    }}
    .sfl-banner {{
        padding: 1rem 1.25rem;
        border-radius: 8px;
        font-weight: 500;
    }}
    .sfl-banner-ship {{
        background: rgba(79, 121, 66, 0.12);
        border-left: 4px solid {COLOR_POSITIVE};
        color: {COLOR_TEXT};
    }}
    .sfl-banner-iterate {{
        background: rgba(155, 34, 38, 0.10);
        border-left: 4px solid {COLOR_NEGATIVE};
        color: {COLOR_TEXT};
    }}
    .sfl-banner-hold {{
        background: rgba(200, 85, 61, 0.10);
        border-left: 4px solid {COLOR_ACCENT};
        color: {COLOR_TEXT};
    }}
    .sfl-banner-block {{
        background: rgba(155, 34, 38, 0.15);
        border-left: 4px solid {COLOR_NEGATIVE};
        color: {COLOR_TEXT};
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =====================================================================
# Helpers
# =====================================================================


@dataclass(frozen=True)
class DecisionVerdict:
    """Plain-English verdict with a CSS class for banner styling."""

    label: str
    css_class: str
    explanation: str


def classify_decision(
    p_t_beats_c: float,
    expected_loss_t: float,
    ci_low: float,
    srm_passed: bool,
) -> DecisionVerdict:
    """Map three readout numbers to a ship / hold / iterate verdict.

    Thresholds mirror notebooks/02_ab_framework.ipynb §8 so the app and the
    narrative stay in sync.
    """
    if not srm_passed:
        return DecisionVerdict(
            "BLOCKED (SRM fail)",
            "sfl-banner-block",
            "Sample ratio mismatch — do not read the primary metric. Investigate randomisation.",
        )
    if p_t_beats_c >= 0.95 and expected_loss_t <= 0.0005 and ci_low > 0:
        return DecisionVerdict(
            "SHIP",
            "sfl-banner-ship",
            "Posterior probability high, expected loss tight, CI excludes zero. Launch.",
        )
    if p_t_beats_c < 0.80 or ci_low < -0.005:
        return DecisionVerdict(
            "ITERATE",
            "sfl-banner-iterate",
            "Treatment is probably flat or worse. Drop it and iterate.",
        )
    return DecisionVerdict(
        "HOLD",
        "sfl-banner-hold",
        "Inconclusive. Extend horizon or check for novelty / primacy effects.",
    )


def render_banner(verdict: DecisionVerdict) -> None:
    """Render a coloured verdict banner using raw HTML."""
    st.markdown(
        f"<div class='sfl-banner {verdict.css_class}'>"
        f"<strong>Decision: {verdict.label}</strong><br>"
        f"<span style='font-size:0.92rem;'>{verdict.explanation}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


# =====================================================================
# Tab 1 — Planner
# =====================================================================


def render_planner() -> None:
    """Sample-size + duration planner with a power curve."""
    st.subheader("1 · Plan — sizing a new experiment")
    st.markdown(
        "<p class='sfl-caption'>Enter the baseline CVR you expect at the target stage, "
        "the smallest lift worth shipping, and your daily eligible traffic. The planner "
        "returns the per-arm sample size, total n, and minimum calendar duration.</p>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1.4], gap="large")

    with left:
        st.markdown("**Inputs**")
        baseline = st.number_input(
            "Baseline CVR",
            min_value=0.001,
            max_value=0.999,
            value=0.20,
            step=0.005,
            format="%.3f",
            help="Control-arm conversion rate at the target stage (0.20 = 20%).",
        )
        mde = st.number_input(
            "Minimum detectable effect (absolute pp)",
            min_value=0.001,
            max_value=0.50,
            value=0.01,
            step=0.001,
            format="%.3f",
            help="Smallest absolute lift you want powered. 0.01 = +1pp.",
        )
        alpha = st.select_slider(
            "Significance level (alpha)",
            options=[0.01, 0.025, 0.05, 0.10],
            value=0.05,
        )
        power = st.select_slider(
            "Statistical power",
            options=[0.70, 0.80, 0.85, 0.90, 0.95],
            value=0.80,
        )
        daily_traffic = st.number_input(
            "Daily eligible traffic per arm",
            min_value=1,
            max_value=10_000_000,
            value=5_000,
            step=500,
            help="Users per day flowing into each arm after allocation.",
        )

    plan = sample_size_per_arm(baseline, mde, alpha=alpha, power=power)
    duration = experiment_duration_days(plan.sample_size_per_arm, daily_traffic)

    with right:
        st.markdown("**Outputs**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Per arm", f"{plan.sample_size_per_arm:,}")
        m2.metric("Total n", f"{plan.total_sample_size:,}")
        m3.metric("Duration", f"{duration} day{'s' if duration != 1 else ''}")

        st.caption(
            f"Sizing assumes a two-sided test at alpha = {alpha}, power = {power:.0%}. "
            f"Daily traffic is constant; a production plan should pad 20-30% for "
            f"weekday/weekend cohort drift."
        )

        fig = _power_curve_figure(baseline, mde, alpha)
        st.plotly_chart(fig, config={"displayModeBar": False})


def _power_curve_figure(baseline: float, mde_chosen: float, alpha: float) -> go.Figure:
    """Power-curve: sample size vs MDE at 80 / 90 / 95% power."""
    mde_axis = np.linspace(max(0.001, mde_chosen * 0.3), max(0.03, mde_chosen * 3.0), 60)
    powers = (0.80, 0.90, 0.95)
    palette = {0.80: COLOR_PRIMARY, 0.90: COLOR_ACCENT, 0.95: COLOR_NEGATIVE}
    fig = go.Figure()
    for p in powers:
        ys = [
            sample_size_per_arm(baseline, float(m), alpha=alpha, power=p).sample_size_per_arm
            for m in mde_axis
        ]
        fig.add_trace(
            go.Scatter(
                x=mde_axis * 100,
                y=ys,
                mode="lines",
                name=f"Power = {p:.0%}",
                line={"color": palette[p], "width": 3},
                hovertemplate="MDE %{x:.2f}pp<br>n per arm %{y:,.0f}<extra></extra>",
            )
        )
    fig.add_vline(
        x=mde_chosen * 100,
        line={"color": COLOR_MUTED, "width": 1, "dash": "dot"},
        annotation_text=f"Chosen MDE {mde_chosen * 100:.2f}pp",
        annotation_position="top right",
        annotation_font={"color": COLOR_MUTED, "size": 11},
    )
    fig.update_yaxes(title_text="Sample size per arm", type="log", tickformat=",")
    fig.update_xaxes(title_text="MDE (pp)")
    fig.update_layout(title="Sample size vs MDE at fixed power")
    apply_sfl_theme(fig, height=360, subtitle=f"Baseline CVR {baseline:.1%}, alpha = {alpha}.")
    return fig


# =====================================================================
# Tab 2 — Readout
# =====================================================================


def render_readout() -> None:
    """Paste counts, read out frequentist + Bayesian, get a verdict."""
    st.subheader("2 · Read out — did the treatment win?")
    st.markdown(
        "<p class='sfl-caption'>Paste the raw assignment and conversion counts. The "
        "readout gates on SRM first, then runs both the pooled-variance z-test and "
        "the Beta-Binomial posterior. The verdict combines the three numbers into "
        "a ship / hold / iterate call.</p>",
        unsafe_allow_html=True,
    )

    col_c, col_t, col_conf = st.columns(3, gap="large")
    with col_c:
        st.markdown("**Control**")
        c_n = st.number_input("Users", min_value=1, value=25_000, step=100, key="c_n")
        c_conv = st.number_input(
            "Conversions",
            min_value=0,
            max_value=int(c_n),
            value=min(5_000, int(c_n)),
            step=10,
            key="c_conv",
        )
    with col_t:
        st.markdown("**Treatment**")
        t_n = st.number_input("Users", min_value=1, value=25_000, step=100, key="t_n")
        t_conv = st.number_input(
            "Conversions",
            min_value=0,
            max_value=int(t_n),
            value=min(5_250, int(t_n)),
            step=10,
            key="t_conv",
        )
    with col_conf:
        st.markdown("**Settings**")
        alpha = st.select_slider(
            "Alpha",
            options=[0.01, 0.025, 0.05, 0.10],
            value=0.05,
            key="readout_alpha",
        )
        prior_strength = st.select_slider(
            "Bayesian prior",
            options=["Uniform", "Weak (baseline=0.20, s=50)", "Medium (s=200)"],
            value="Uniform",
            help="Uniform is Beta(1,1). Non-uniform shrinks toward baseline 0.20.",
        )
        n_draws = st.select_slider(
            "Posterior draws",
            options=[20_000, 50_000, 200_000],
            value=50_000,
        )

    prior_alpha, prior_beta = _resolve_prior(prior_strength)

    control = ArmStats("Control", n=int(c_n), conversions=int(c_conv))
    treatment = ArmStats("Treatment", n=int(t_n), conversions=int(t_conv))

    srm = check_srm([control.n, treatment.n])
    freq = frequentist_test(control, treatment, alpha=alpha)
    bayes = bayesian_test(
        control,
        treatment,
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
        n_draws=int(n_draws),
        rng=np.random.default_rng(2026),
    )

    verdict = classify_decision(
        bayes.prob_treatment_beats_control,
        bayes.expected_loss_choose_treatment,
        freq.ci_low_abs,
        srm.passed,
    )

    st.markdown("### Decision")
    render_banner(verdict)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Observed lift (pp)",
        f"{freq.lift_abs * 100:+.3f}",
        delta=f"{freq.lift_rel * 100:+.1f}% relative",
        delta_color="normal",
    )
    m2.metric("P(T > C)", f"{bayes.prob_treatment_beats_control:.3f}")
    m3.metric("Expected loss choose T", f"{bayes.expected_loss_choose_treatment * 100:.3f}pp")
    m4.metric(
        "SRM p-value",
        f"{srm.p_value:.3f}",
        delta="PASS" if srm.passed else "FAIL",
        delta_color="normal" if srm.passed else "inverse",
    )

    tab_freq, tab_bayes, tab_tables = st.tabs(["Frequentist", "Bayesian", "Raw numbers"])
    with tab_freq:
        st.plotly_chart(_lift_ci_figure(freq), config={"displayModeBar": False})
        st.markdown(
            f"- **z-statistic** (pooled SE): `{freq.z_stat:+.3f}`  \n"
            f"- **p-value** (two-sided): `{freq.p_value:.4f}`  \n"
            f"- **{int((1 - freq.alpha) * 100)}% CI on the absolute lift:** "
            f"`[{freq.ci_low_abs * 100:+.3f}pp, {freq.ci_high_abs * 100:+.3f}pp]`"
        )
    with tab_bayes:
        st.plotly_chart(
            _posterior_figure(bayes, control, treatment),
            config={"displayModeBar": False},
        )
        lo, hi = bayes.credible_interval_abs
        st.markdown(
            f"- **Prior:** `Beta({bayes.prior_alpha:g}, {bayes.prior_beta:g})`  \n"
            f"- **Posterior draws:** `{bayes.n_draws:,}`  \n"
            f"- **{int(bayes.credible_level * 100)}% credible interval on lift_abs:** "
            f"`[{lo * 100:+.3f}pp, {hi * 100:+.3f}pp]`  \n"
            f"- **Expected loss if we choose Control:** "
            f"`{bayes.expected_loss_choose_control * 100:.4f}pp`"
        )
    with tab_tables:
        st.dataframe(
            pd.DataFrame(
                {
                    "Arm": [control.name, treatment.name],
                    "Users": [control.n, treatment.n],
                    "Conversions": [control.conversions, treatment.conversions],
                    "Observed CVR": [control.rate, treatment.rate],
                }
            ),
            hide_index=True,
        )


def _resolve_prior(label: str) -> tuple[float, float]:
    """Map the dropdown label to a (prior_alpha, prior_beta) pair."""
    if label == "Uniform":
        return 1.0, 1.0
    if label.startswith("Weak"):
        return 50 * 0.20, 50 * 0.80
    return 200 * 0.20, 200 * 0.80


def _lift_ci_figure(freq) -> go.Figure:  # type: ignore[no-untyped-def]
    """Forest-style CI on the absolute lift."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[freq.lift_abs * 100],
            y=["Absolute lift"],
            mode="markers",
            marker={
                "color": COLOR_PRIMARY,
                "size": 14,
                "line": {"color": "white", "width": 2},
            },
            error_x={
                "type": "data",
                "symmetric": False,
                "array": [(freq.ci_high_abs - freq.lift_abs) * 100],
                "arrayminus": [(freq.lift_abs - freq.ci_low_abs) * 100],
                "color": COLOR_PRIMARY,
                "thickness": 2,
                "width": 12,
            },
            hovertemplate="Lift %{x:+.3f}pp<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_vline(
        x=0.0,
        line={"color": COLOR_ACCENT, "width": 1.5, "dash": "dash"},
    )
    fig.update_xaxes(title_text="Absolute lift (pp)", zeroline=False)
    fig.update_yaxes(title_text="", showticklabels=True)
    fig.update_layout(title=f"{int((1 - freq.alpha) * 100)}% CI on the absolute lift")
    apply_sfl_theme(
        fig,
        height=220,
        subtitle=f"z = {freq.z_stat:+.2f}, p = {freq.p_value:.4f}.",
    )
    return fig


def _posterior_figure(bayes, control: ArmStats, treatment: ArmStats) -> go.Figure:  # type: ignore[no-untyped-def]
    """Two posterior densities, side by side, with shaded fills."""
    lo = max(0.0, min(control.rate, treatment.rate) - 0.04)
    hi = min(1.0, max(control.rate, treatment.rate) + 0.04)
    x_grid = np.linspace(lo, hi, 500)
    post_c = stats.beta.pdf(
        x_grid,
        bayes.prior_alpha + control.conversions,
        bayes.prior_beta + control.n - control.conversions,
    )
    post_t = stats.beta.pdf(
        x_grid,
        bayes.prior_alpha + treatment.conversions,
        bayes.prior_beta + treatment.n - treatment.conversions,
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_grid * 100,
            y=post_c,
            fill="tozeroy",
            name="Control",
            line={"color": COLOR_PRIMARY, "width": 2.5},
            fillcolor="rgba(31, 58, 95, 0.35)",
            hovertemplate="CVR %{x:.2f}%<extra>Control</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_grid * 100,
            y=post_t,
            fill="tozeroy",
            name="Treatment",
            line={"color": COLOR_ACCENT, "width": 2.5},
            fillcolor="rgba(200, 85, 61, 0.35)",
            hovertemplate="CVR %{x:.2f}%<extra>Treatment</extra>",
        )
    )
    fig.update_xaxes(title_text="CVR (%)")
    fig.update_yaxes(title_text="Posterior density", showticklabels=False)
    fig.update_layout(title="Posterior distributions of the two CVRs")
    apply_sfl_theme(
        fig,
        height=340,
        subtitle=(
            f"Beta({bayes.prior_alpha:g}, {bayes.prior_beta:g}) prior, " f"{bayes.n_draws:,} draws."
        ),
    )
    return fig


# =====================================================================
# Tab 3 — Peeking Lab
# =====================================================================


@st.cache_data(show_spinner=False)
def _cached_peek_sim(
    baseline: float,
    n_total: int,
    n_peeks: int,
    alpha: float,
    n_sims: int,
    seed: int,
) -> float:
    """Wrap ``simulate_peeking_inflation`` with Streamlit's args-keyed cache."""
    return simulate_peeking_inflation(
        baseline_rate=baseline,
        n_total_per_arm=n_total,
        n_peeks=n_peeks,
        alpha=alpha,
        n_sims=n_sims,
        rng=np.random.default_rng(seed),
    )


def render_peeking() -> None:
    """Empirical Type-I inflation under peek-and-stop."""
    st.subheader("3 · Peeking lab — why monitoring a frequentist test is dangerous")
    st.markdown(
        "<p class='sfl-caption'>Every experiment simulated here is an <b>A/A</b> "
        "(same true rate for both arms). Under a correctly-sized fixed-horizon test "
        "exactly alpha of them should reject H0. Under peek-and-stop, the observed "
        "Type-I rate inflates with the number of looks.</p>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")
    with col1:
        baseline = st.number_input(
            "True rate (both arms)",
            min_value=0.01,
            max_value=0.99,
            value=0.20,
            step=0.01,
            key="peek_base",
        )
        n_total = st.number_input(
            "Horizon (per arm)",
            min_value=1_000,
            max_value=200_000,
            value=25_000,
            step=1_000,
            key="peek_n",
        )
    with col2:
        alpha = st.select_slider(
            "Alpha",
            options=[0.01, 0.025, 0.05, 0.10],
            value=0.05,
            key="peek_alpha",
        )
        n_sims = st.select_slider(
            "Simulations per peek count",
            options=[500, 1_000, 2_000],
            value=1_000,
            help=(
                "Each cell runs this many A/A experiments. More sims = "
                "tighter estimate, longer runtime."
            ),
        )

    if st.button("Run peeking simulation", type="primary"):
        schedule = [1, 2, 5, 10, 20]
        observed = {
            k: _cached_peek_sim(baseline, int(n_total), k, alpha, int(n_sims), seed=100 + k)
            for k in schedule
        }
        table = pd.DataFrame(
            {
                "Peeks": schedule,
                "Observed Type-I rate": [observed[k] for k in schedule],
                "Inflation vs nominal": [observed[k] / alpha for k in schedule],
            }
        )
        st.dataframe(
            table.style.format(
                {"Observed Type-I rate": "{:.1%}", "Inflation vs nominal": "{:.2f}x"}
            ),
            hide_index=True,
        )

        fig = _peek_figure(schedule, observed, alpha)
        st.plotly_chart(fig, config={"displayModeBar": False})

        st.markdown(
            "**Takeaway.** The single pre-declared horizon (green bar) is the only "
            "honest frequentist reading. Everything else is alpha-inflated. Two "
            "industry-grade remedies: (a) switch to Bayesian posterior inference — "
            "always valid, does not depend on horizon or stopping rule; (b) adopt a "
            "sequential frequentist design with pre-declared peek schedule and "
            "alpha-spending (O'Brien-Fleming, Pocock, or mSPRT)."
        )
    else:
        st.info("Click **Run peeking simulation** to populate the chart (~2-5 seconds).")


def _peek_figure(
    schedule: list[int],
    observed: dict[int, float],
    alpha: float,
) -> go.Figure:
    """Bar chart of observed alpha per peek count with a nominal-alpha line."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[f"{k} peek{'s' if k > 1 else ''}" for k in schedule],
            y=[observed[k] * 100 for k in schedule],
            marker={
                "color": [COLOR_POSITIVE if k == 1 else COLOR_NEGATIVE for k in schedule],
                "line": {"color": "white", "width": 1.5},
            },
            text=[f"{observed[k] * 100:.1f}%" for k in schedule],
            textposition="outside",
            textfont={"size": 13, "color": COLOR_TEXT},
            hovertemplate="%{x}<br>Observed alpha %{y:.2f}%<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_hline(
        y=alpha * 100,
        line={"color": COLOR_ACCENT, "width": 1.5, "dash": "dash"},
        annotation_text=f"Nominal alpha = {alpha * 100:.1f}%",
        annotation_position="top right",
        annotation_font={"color": COLOR_ACCENT, "size": 11},
    )
    fig.update_yaxes(title_text="Observed Type-I rate (%)")
    fig.update_xaxes(title_text="Interim analyses per experiment")
    fig.update_layout(title="Peeking inflates the false-positive rate")
    apply_sfl_theme(fig, height=380, subtitle="Green bar is the one honest reading.")
    return fig


# =====================================================================
# App entry
# =====================================================================


def main() -> None:
    """Render the app."""
    st.title("SmokeFreeLab — Experiment Designer")
    st.markdown(
        "<p class='sfl-caption'>Plan, read out, and audit A/B tests for a simulated "
        "smoke-free product funnel. Every number on this page is computed from the "
        "same library the notebooks use — "
        "<code>smokefreelab.experiment</code>.</p>",
        unsafe_allow_html=True,
    )

    tab_planner, tab_readout, tab_peeking = st.tabs(["Planner", "Readout", "Peeking lab"])

    with tab_planner:
        render_planner()
    with tab_readout:
        render_readout()
    with tab_peeking:
        render_peeking()

    st.markdown("---")
    st.caption(
        "Built on the GA4 obfuscated e-commerce sample. See the notebooks in the repo "
        "for the full narrative, data pipeline, and derivation of the IDR baseline."
    )


if __name__ == "__main__":
    main()
