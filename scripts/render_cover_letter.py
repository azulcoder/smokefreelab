"""Render reports/cover_letter_sampoerna.pdf — tailored to Sampoerna req 24383.

Standalone reportlab renderer, matches the executive-onepager typography.
Produces a two-page A4 PDF with justified body text, a header block, and a
clean sign-off.

Usage:
    python3 scripts/render_cover_letter.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "reports" / "cover_letter_sampoerna.pdf"

INK = colors.HexColor("#111111")
SUB = colors.HexColor("#444444")
ACCENT = colors.HexColor("#8B0000")


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()["BodyText"]
    header = ParagraphStyle(
        "header",
        parent=base,
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=INK,
        alignment=TA_LEFT,
    )
    subheader = ParagraphStyle(
        "subheader",
        parent=base,
        fontName="Helvetica",
        fontSize=10,
        leading=12,
        textColor=SUB,
    )
    salutation = ParagraphStyle(
        "salutation",
        parent=base,
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=INK,
        spaceAfter=8,
    )
    body = ParagraphStyle(
        "body",
        parent=base,
        fontName="Helvetica",
        fontSize=10.5,
        leading=14,
        textColor=INK,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
    )
    bullet = ParagraphStyle(
        "bullet",
        parent=body,
        leftIndent=14,
        bulletIndent=0,
        spaceAfter=4,
    )
    sign_off = ParagraphStyle(
        "signoff",
        parent=body,
        fontName="Helvetica-Oblique",
        alignment=TA_LEFT,
        spaceBefore=6,
        spaceAfter=4,
    )
    name = ParagraphStyle(
        "name",
        parent=body,
        fontName="Helvetica-Bold",
        alignment=TA_LEFT,
        spaceAfter=2,
    )
    return {
        "header": header,
        "subheader": subheader,
        "salutation": salutation,
        "body": body,
        "bullet": bullet,
        "sign_off": sign_off,
        "name": name,
    }


def build_story(styles: dict[str, ParagraphStyle]) -> list[object]:
    s = styles
    story: list[object] = []

    story.append(Paragraph("Ahmad Zulfan (Az)", s["header"]))
    story.append(
        Paragraph(
            "Jakarta &middot; bilingual (Indonesian / English) &middot; "
            "<font color='#8B0000'>github.com/azulcoder</font> &middot; "
            "infoman.xyz123@gmail.com",
            s["subheader"],
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "<b>To:</b> PT HM Sampoerna Tbk &mdash; Data Science &amp; "
            "Analytics Hiring Team<br/>"
            "<b>Re:</b> Data Science &amp; Analytics role &mdash; "
            "requisition 24383<br/>"
            "<b>Date:</b> April 2026",
            s["subheader"],
        )
    )
    story.append(Spacer(1, 12))

    story.append(Paragraph("Dear Sampoerna Data Science &amp; Analytics Hiring Team,", s["salutation"]))

    story.append(
        Paragraph(
            "I am writing to apply for the Data Science &amp; Analytics "
            "position at PT HM Sampoerna Tbk (requisition 24383). I have "
            "spent the past several months building a portfolio that maps "
            "directly to the skills your JD lists, and I would welcome the "
            "chance to walk a hiring manager through it.",
            s["body"],
        )
    )

    story.append(
        Paragraph(
            "The project is called <b>SmokeFreeLab</b> "
            "(<font color='#8B0000'>github.com/azulcoder/smokefreelab</font>). "
            "It is an end-to-end product-analytics and experimentation "
            "playbook for a smoke-free-product e-commerce funnel, built on "
            "the GA4 obfuscated sample in BigQuery. Nine executable "
            "notebooks cover funnel decomposition, frequentist and "
            "Bayesian A/B testing, multi-touch attribution (Markov and "
            "Shapley), price elasticity with a cukai scenario, customer "
            "lifetime value (BG/NBD plus Gamma-Gamma), and a Bayesian "
            "Marketing Mix Model with custom adstock and Hill saturation "
            "in PyMC. A Streamlit Experiment Designer &mdash; three tabs: "
            "Planner, Readout, and Peeking lab &mdash; sits on top of the "
            "A/B engine, which has 55 tests at 92% coverage. Every "
            "notebook closes with a rupiah-framed business impact section.",
            s["body"],
        )
    )

    story.append(
        Paragraph(
            "Mapping your JD against what the repository already ships:",
            s["body"],
        )
    )

    bullets = [
        (
            "<b>A/B testing (frequentist and Bayesian).</b> Both are "
            "implemented side by side. Stakeholders who understand "
            "<i>P(treatment beats control) = 96%</i> do not always parse "
            "<i>p = 0.04</i>; the repo defends both."
        ),
        (
            "<b>GA4 and Looker.</b> Notebook 01 pulls directly from the "
            "GA4 BigQuery public sample &mdash; 354,857 sessions, "
            "five-stage funnel decomposition. A Looker Studio spec lives "
            "in <i>dashboards/</i>."
        ),
        (
            "<b>Advanced SQL (CTEs, window functions).</b> Production-grade "
            "BigQuery and Postgres for six years; "
            "<i>sql/01_funnel_decomposition.sql</i> is representative of "
            "the style."
        ),
        (
            "<b>Python (Pandas, Scikit-learn, Statsmodels).</b> The "
            "<i>src/</i> tree is typed with "
            "<font face='Courier'>mypy --strict</font>, linted with ruff, "
            "CI-green on every commit."
        ),
        (
            "<b>Advanced Excel (financial modeling, complex formulas).</b> "
            "Long history of building economic models with Monte Carlo "
            "overlays and multi-sheet scenario stacks in Excel."
        ),
        (
            "<b>Time-series analysis (Prophet, ARIMA) and growth modeling "
            "(S-curves).</b> Hyperbolic decline fits and Gompertz / "
            "logistic family models are part of my daily toolkit; Prophet "
            "and ARIMA are within that family."
        ),
    ]
    for b in bullets:
        story.append(Paragraph("&bull;&nbsp;&nbsp; " + b, s["bullet"]))

    story.append(Spacer(1, 4))

    story.append(
        Paragraph(
            "Your JD also emphasises translating complex findings for "
            "senior leadership. The repository ships three artifacts built "
            "for that audience: a single-page rupiah-framed one-pager "
            "(<i>reports/executive_onepager.pdf</i>); a 10-slide deck "
            "with speaker notes "
            "(<i>reports/smokefreelab_deck.pptx</i>); and a Bahasa "
            "Indonesia executive summary "
            "(<i>docs/ringkasan_eksekutif.md</i>). The headline numbers "
            "&mdash; including an approximate <b>Rp 27 B per year</b> "
            "incremental CLV on a Sampoerna-scale funnel and an "
            "approximate <b>&minus;14% volume</b> response to a 26.7% "
            "cukai-driven price rise &mdash; all derive from a single "
            "arithmetic notebook (<i>reports/_derivation.ipynb</i>), so "
            "every figure is reproducible.",
            s["body"],
        )
    )

    story.append(
        Paragraph(
            "I am based in Jakarta, bilingual, and available to start "
            "within four weeks. I would welcome a conversation about how "
            "the toolkit applies to Sampoerna's category economics &mdash; "
            "the cukai-sensitivity and MMM reallocation work are the two "
            "pieces I expect would land first in a Brand Analytics review.",
            s["body"],
        )
    )

    story.append(Paragraph("<i>Hormat saya,</i>", s["sign_off"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>Ahmad Zulfan (Az)</b>", s["name"]))
    story.append(
        Paragraph(
            "<font color='#8B0000'>github.com/azulcoder</font> &middot; "
            "infoman.xyz123@gmail.com &middot; Jakarta<br/>"
            "Portfolio: <font color='#8B0000'>"
            "github.com/azulcoder/smokefreelab</font>",
            s["subheader"],
        )
    )

    return story


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = build_styles()
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=2.0 * cm,
        rightMargin=2.0 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title="Cover letter — PT HM Sampoerna Tbk (req 24383)",
        author="Ahmad Zulfan (Az)",
    )
    doc.build(build_story(styles))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
