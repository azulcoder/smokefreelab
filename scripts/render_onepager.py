"""Render reports/executive_onepager.pdf from reports/executive_onepager.md.

Standalone reportlab renderer — no pandoc dependency. Produces a single-page
A4 PDF with the five headline numbers as a rupiah-denominated table, followed
by the repo summary, the "why me" bridge, and contact info.

Usage:
    python3 scripts/render_onepager.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "reports" / "executive_onepager.pdf"

INK = colors.HexColor("#111111")
SUB = colors.HexColor("#444444")
ACCENT = colors.HexColor("#8B0000")
RULE = colors.HexColor("#999999")
BAND = colors.HexColor("#F3F1EE")


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()["BodyText"]
    title = ParagraphStyle(
        "title",
        parent=base,
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=INK,
        alignment=TA_LEFT,
    )
    subtitle = ParagraphStyle(
        "subtitle",
        parent=base,
        fontName="Helvetica",
        fontSize=10,
        leading=12,
        textColor=SUB,
    )
    section = ParagraphStyle(
        "section",
        parent=base,
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=ACCENT,
        spaceBefore=8,
        spaceAfter=3,
    )
    body = ParagraphStyle(
        "body",
        parent=base,
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=INK,
        spaceAfter=4,
    )
    small = ParagraphStyle(
        "small",
        parent=base,
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=10,
        textColor=SUB,
    )
    return {
        "title": title,
        "subtitle": subtitle,
        "section": section,
        "body": body,
        "small": small,
    }


def headline_table(styles: dict[str, ParagraphStyle]) -> Table:
    cell = ParagraphStyle(
        "cell", parent=styles["body"], fontSize=9, leading=11, spaceAfter=0
    )
    cell_bold = ParagraphStyle(
        "cell_bold",
        parent=cell,
        fontName="Helvetica-Bold",
        textColor=ACCENT,
    )
    rows: list[list[Paragraph]] = [
        [
            Paragraph("<b>#</b>", cell_bold),
            Paragraph("<b>Number</b>", cell_bold),
            Paragraph("<b>Where it comes from</b>", cell_bold),
        ],
        [
            Paragraph("1", cell),
            Paragraph("<b>Rp 27 B / year</b><br/>incremental CLV", cell_bold),
            Paragraph(
                "250,000 monthly registrants &times; 12 months &times; 2 pp "
                "activation lift &times; Rp 450,000 LTV/activated user. "
                "<i>notebooks/06_business_case.ipynb</i>.",
                cell,
            ),
        ],
        [
            Paragraph("2", cell),
            Paragraph("<b>Rp 50 B / quarter</b><br/>media budget", cell_bold),
            Paragraph(
                "TV Rp 1.25 B/wk + digital Rp 450 M/wk + trade Rp 2 B/wk "
                "&times; 13 weeks. <i>notebooks/09_mmm.ipynb</i>.",
                cell,
            ),
        ],
        [
            Paragraph("3", cell),
            Paragraph(
                "<b>Rp 1.4 B / quarter</b><br/>TV &rarr; trade reallocation",
                cell_bold,
            ),
            Paragraph(
                "Posterior-mean marginal ROI gap &times; Rp 5 B reallocation, "
                "saturation-adjusted. <i>notebooks/09_mmm.ipynb</i>.",
                cell,
            ),
        ],
        [
            Paragraph("4", cell),
            Paragraph(
                "<b>48% of CLV</b><br/>in the top decile", cell_bold
            ),
            Paragraph(
                "Lorenz curve on 1,500-customer BG/NBD + Gamma-Gamma "
                "simulation. <i>notebooks/08_clv_rfm.ipynb</i>.",
                cell,
            ),
        ],
        [
            Paragraph("5", cell),
            Paragraph("<b>&asymp; &minus;14% volume</b><br/>on a cukai hike", cell_bold),
            Paragraph(
                "Hierarchical price elasticity &times; 26.7% price rise "
                "(Rp 3,000 &rarr; Rp 3,800/stick). "
                "<i>notebooks/07_price_elasticity.ipynb</i>.",
                cell,
            ),
        ],
    ]

    table = Table(rows, colWidths=[0.7 * cm, 4.4 * cm, 11.9 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BAND),
                ("LINEBELOW", (0, 0), (-1, 0), 0.5, RULE),
                ("LINEBELOW", (0, -1), (-1, -1), 0.5, RULE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def build_story(styles: dict[str, ParagraphStyle]) -> list[object]:
    story: list[object] = []

    story.append(Paragraph("SmokeFreeLab &mdash; Executive one-pager", styles["title"]))
    story.append(
        Paragraph(
            "Ahmad Zulfan (Az) &middot; Jakarta &middot; April 2026",
            styles["subtitle"],
        )
    )
    story.append(Spacer(1, 6))

    story.append(
        Paragraph(
            "<b>One line.</b> An end-to-end product-analytics and "
            "experimentation stack for a smoke-free-product (SFP) e-commerce "
            "funnel, built on the GA4 obfuscated sample in BigQuery.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "<b>Why it matters.</b> A disciplined experimentation plus "
            "attribution plus MMM stack is worth approximately "
            "<b>Rp 27 B / year</b> on a 250K-monthly-registrant funnel. "
            "Full derivation reproducible from "
            "<i>reports/_derivation.ipynb</i>.",
            styles["body"],
        )
    )

    story.append(Paragraph("The five headline numbers", styles["section"]))
    story.append(headline_table(styles))

    story.append(Paragraph("What the repo demonstrates", styles["section"]))
    bullets = [
        "<b>A/B testing</b> &mdash; frequentist + Bayesian + SRM + power + "
        "peeking proof. 55 tests @ 92% cov. Streamlit three-tab designer.",
        "<b>Multi-touch attribution</b> &mdash; Markov removal-effect + exact "
        "Shapley + four heuristics.",
        "<b>Propensity</b> &mdash; XGBoost + SHAP + calibration curve.",
        "<b>Price elasticity</b> &mdash; log-log OLS + Bayesian hierarchical "
        "pooling, bootstrap CIs, cukai scenario.",
        "<b>CLV + RFM</b> &mdash; BG/NBD + Gamma-Gamma + 11 canonical "
        "segments + Lorenz / Gini.",
        "<b>MMM</b> &mdash; Bayesian custom adstock + Hill saturation in "
        "PyMC, per-channel response curves + budget reallocation.",
    ]
    for item in bullets:
        story.append(Paragraph("&bull;&nbsp; " + item, styles["body"]))

    story.append(Paragraph("How to read this repo in five minutes", styles["section"]))
    steps = [
        "<b>README.md</b> &mdash; hero table and quickstart.",
        "<b>notebooks/06_business_case.ipynb</b> &mdash; Rp 27 B derivation.",
        "<b>notebooks/09_mmm.ipynb</b> &mdash; MMM with adstock and Hill.",
        "<b>make run-app</b> &mdash; Experiment Designer in the browser.",
        "<b>docs/ringkasan_eksekutif.md</b> &mdash; the same story in Bahasa.",
    ]
    for idx, item in enumerate(steps, start=1):
        story.append(Paragraph(f"{idx}.&nbsp; {item}", styles["body"]))

    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "<b>Contact.</b> Ahmad Zulfan (Az) &middot; Jakarta &middot; "
            "bilingual (Indonesian / English) &middot; "
            "<font color='#8B0000'>github.com/azulcoder/smokefreelab</font>",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "Simulation disclosure: the GA4 public sample is real; "
            "price-elasticity panels, CLV cohorts, and MMM media spend are "
            "synthetic but calibrated to resemble the Indonesian SFP market. "
            "What ships here is the methodology, not ground-truth recovery.",
            styles["small"],
        )
    )

    return story


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = build_styles()
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.4 * cm,
        title="SmokeFreeLab — Executive one-pager",
        author="Ahmad Zulfan (Az)",
    )
    story = build_story(styles)
    doc.build(story)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
