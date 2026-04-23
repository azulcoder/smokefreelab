"""Editorial-grade visualization helpers for SmokeFreeLab notebooks.

Every notebook in the project renders through this module so that figures
inherit the same palette, typography, and layout conventions. Inconsistent
styling across notebooks is a tell that the author ships visuals as an
afterthought; a single shared theme is the cheapest way to read as
portfolio-grade.

Business context
----------------
Hiring managers spend seconds per chart. The palette below is deliberately
narrow (one primary, one accent, one positive, one negative, one muted)
because a five-colour vocabulary is more legible than a rainbow. The IDR
formatter auto-scales because rupiah figures span ``IDR 450K`` per user to
``IDR 27B`` annualized — readers should never have to count zeros.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import plotly.graph_objects as go

COLOR_PRIMARY = "#1F3A5F"
"""Deep navy — primary series, headline bars, funnel entry."""

COLOR_ACCENT = "#C8553D"
"""Terra cotta — accent series, emphasis, worst-leak annotation."""

COLOR_POSITIVE = "#4F7942"
"""Olive green — lift, positive delta, retained users."""

COLOR_NEGATIVE = "#9B2226"
"""Deep red — drop-off, negative delta, lost users."""

COLOR_MUTED = "#A8B4C0"
"""Cool gray — secondary text, grid accents, context bands."""

COLOR_BACKGROUND = "#FAFAF5"
"""Cream — figure background. Warmer than pure white, easier on eyes."""

COLOR_TEXT = "#2B2B2B"
"""Near-black — headings and axis labels."""

COLOR_GRID = "#E4E4E0"
"""Off-white — subtle grid that recedes behind data."""

FUNNEL_PALETTE = [
    "#1F3A5F",
    "#3E5F85",
    "#6B8CAE",
    "#A88576",
    "#C8553D",
]
"""5-colour gradient (navy → terra cotta) for the 5-stage funnel.

The colour walks from cold (top-of-funnel, neutral) to warm (purchase,
commercial outcome). This reinforces the narrative that value concentrates
at the bottom of the funnel.
"""

FONT_FAMILY = "Helvetica Neue, Helvetica, Arial, sans-serif"


def apply_sfl_theme(
    fig: go.Figure,
    *,
    height: int = 520,
    title_size: int = 20,
    subtitle: str | None = None,
) -> go.Figure:
    """Apply the SmokeFreeLab editorial theme to a Plotly figure.

    The theme fixes background, typography, grid, and margins so figures
    read as a family rather than a collection. Call this on every figure
    before ``fig.show()`` or ``fig.write_image()``.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Figure to style, mutated in place and returned.
    height : int, default 520
        Figure height in pixels. 520 is tuned for a single-column notebook
        render; bump to 640 for dense heatmaps or Sankey diagrams.
    title_size : int, default 20
        Title font size. Subtitle is rendered two points smaller.
    subtitle : str, optional
        Grey subtitle beneath the main title, useful for the analytic
        question or window ("Nov 2020 - Jan 2021, GA4 obfuscated sample").

    Returns
    -------
    plotly.graph_objects.Figure
        The same figure, themed.
    """
    title_text: str | None = None
    if fig.layout.title is not None and fig.layout.title.text:
        main = fig.layout.title.text
        if subtitle:
            title_text = (
                f"<b>{main}</b>"
                f"<br><span style='font-size:{title_size - 6}px;color:{COLOR_MUTED}'>"
                f"{subtitle}</span>"
            )
        else:
            title_text = f"<b>{main}</b>"

    fig.update_layout(
        height=height,
        paper_bgcolor=COLOR_BACKGROUND,
        plot_bgcolor=COLOR_BACKGROUND,
        font={"family": FONT_FAMILY, "color": COLOR_TEXT, "size": 13},
        title={
            "text": title_text if title_text is not None else fig.layout.title.text,
            "font": {"size": title_size, "color": COLOR_TEXT},
            "x": 0.02,
            "xanchor": "left",
            "y": 0.95,
            "yanchor": "top",
        },
        margin={"l": 70, "r": 40, "t": 90 if subtitle else 70, "b": 60},
        legend={
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": COLOR_GRID,
            "borderwidth": 0,
            "font": {"size": 12, "color": COLOR_TEXT},
        },
        hoverlabel={
            "bgcolor": "white",
            "bordercolor": COLOR_PRIMARY,
            "font": {"family": FONT_FAMILY, "size": 12, "color": COLOR_TEXT},
        },
    )
    fig.update_xaxes(
        showgrid=False,
        showline=True,
        linecolor=COLOR_GRID,
        linewidth=1,
        ticks="outside",
        tickcolor=COLOR_GRID,
        tickfont={"size": 11, "color": COLOR_TEXT},
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=COLOR_GRID,
        gridwidth=1,
        zeroline=False,
        showline=False,
        ticks="outside",
        tickcolor=COLOR_GRID,
        tickfont={"size": 11, "color": COLOR_TEXT},
    )
    return fig


def format_rupiah(value: float, *, scale: str = "auto") -> str:
    """Format a numeric rupiah value with auto-scaled suffix.

    Parameters
    ----------
    value : float
        Raw IDR amount (not already scaled).
    scale : {"auto", "T", "B", "M", "K", "none"}, default "auto"
        Force a specific unit, or let the function pick based on magnitude.
        ``"T"`` triliun (1e12), ``"B"`` miliar (1e9), ``"M"`` juta (1e6),
        ``"K"`` ribu (1e3), ``"none"`` no suffix.

    Returns
    -------
    str
        e.g. ``"IDR 27.0B"``, ``"IDR 450K"``, ``"IDR 1.13T"``.

    Examples
    --------
    >>> format_rupiah(27_000_000_000)
    'IDR 27.0B'
    >>> format_rupiah(450_000)
    'IDR 450K'
    >>> format_rupiah(1_125_000_000, scale="B")
    'IDR 1.1B'
    """
    suffixes = {"T": 1e12, "B": 1e9, "M": 1e6, "K": 1e3, "none": 1.0}
    if scale == "auto":
        abs_value = abs(value)
        if abs_value >= 1e12:
            unit = "T"
        elif abs_value >= 1e9:
            unit = "B"
        elif abs_value >= 1e6:
            unit = "M"
        elif abs_value >= 1e3:
            unit = "K"
        else:
            unit = "none"
    else:
        unit = scale

    divisor = suffixes[unit]
    scaled = value / divisor
    suffix = "" if unit == "none" else unit
    if unit in {"T", "B"}:
        return f"IDR {scaled:,.1f}{suffix}"
    if unit == "M":
        return f"IDR {scaled:,.1f}{suffix}"
    if unit == "K":
        return f"IDR {scaled:,.0f}{suffix}"
    return f"IDR {scaled:,.0f}"


def add_insight_annotation(
    fig: go.Figure,
    *,
    text: str,
    x: float,
    y: float,
    xref: str = "paper",
    yref: str = "paper",
    arrow: bool = False,
) -> go.Figure:
    """Annotate a figure with an editorial-style callout.

    Use sparingly — one or two annotations per chart maximum. These are
    the "reader-friendly insight" captions that turn a chart into a
    narrative artefact rather than a wall of data.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Figure to annotate.
    text : str
        Annotation copy. Keep under 80 characters. Use ``<b>...</b>`` for
        the punchline word.
    x, y : float
        Position. By default in paper coordinates (0-1).
    xref, yref : str
        Plotly reference frames. Use ``"x"`` / ``"y"`` to anchor on the
        data coordinates instead.
    arrow : bool
        If True, draw a short arrow from the annotation toward (x, y).

    Returns
    -------
    plotly.graph_objects.Figure
        The same figure, annotated.
    """
    fig.add_annotation(
        text=text,
        x=x,
        y=y,
        xref=xref,
        yref=yref,
        showarrow=arrow,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1.2,
        arrowcolor=COLOR_ACCENT,
        ax=0 if not arrow else 40,
        ay=0 if not arrow else -40,
        align="left",
        font={"family": FONT_FAMILY, "size": 12, "color": COLOR_TEXT},
        bgcolor="rgba(250, 250, 245, 0.92)",
        bordercolor=COLOR_MUTED,
        borderwidth=0,
        borderpad=6,
    )
    return fig
