"""Unit tests for the editorial-theme visualisation helpers."""

from __future__ import annotations

import plotly.graph_objects as go
import pytest

from smokefreelab.analytics.viz import (
    COLOR_BACKGROUND,
    COLOR_PRIMARY,
    FUNNEL_PALETTE,
    add_insight_annotation,
    apply_sfl_theme,
    format_rupiah,
)


class TestFormatRupiah:
    """Currency auto-scaling must round-trip for each magnitude band."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (1_125_000_000_000, "IDR 1.1T"),
            (27_000_000_000, "IDR 27.0B"),
            (3_600_000_000, "IDR 3.6B"),
            (450_000_000, "IDR 450.0M"),
            (450_000, "IDR 450K"),
            (500, "IDR 500"),
            (0, "IDR 0"),
        ],
    )
    def test_auto_scale(self, value: float, expected: str) -> None:
        """Each input lands in the expected unit with the expected precision."""
        assert format_rupiah(value) == expected

    def test_forced_scale(self) -> None:
        """Explicit scale overrides the auto-picker."""
        assert format_rupiah(1_125_000_000, scale="B") == "IDR 1.1B"
        assert format_rupiah(27_000_000_000, scale="T") == "IDR 0.0T"

    def test_negative_value(self) -> None:
        """Negative values keep their sign and pick scale from magnitude."""
        assert format_rupiah(-3_600_000_000) == "IDR -3.6B"


class TestApplyTheme:
    """Theme application is a smoke test — it must not mangle data."""

    def test_preserves_data(self) -> None:
        """Applying the theme does not touch the trace data."""
        fig = go.Figure(go.Bar(x=["a", "b", "c"], y=[1, 2, 3]))
        apply_sfl_theme(fig)
        assert list(fig.data[0].x) == ["a", "b", "c"]
        assert list(fig.data[0].y) == [1, 2, 3]

    def test_sets_background(self) -> None:
        """Theme paints the configured cream background."""
        fig = go.Figure(go.Bar(x=[1], y=[1]))
        apply_sfl_theme(fig)
        assert fig.layout.paper_bgcolor == COLOR_BACKGROUND
        assert fig.layout.plot_bgcolor == COLOR_BACKGROUND

    def test_subtitle_renders(self) -> None:
        """A subtitle is injected into the title HTML."""
        fig = go.Figure(go.Bar(x=[1], y=[1]))
        fig.update_layout(title="Main title")
        apply_sfl_theme(fig, subtitle="Nov 2020 - Jan 2021")
        assert fig.layout.title.text is not None
        assert "Nov 2020" in fig.layout.title.text
        assert "<b>Main title</b>" in fig.layout.title.text


class TestPalette:
    """Palette must be a narrow, deterministic vocabulary."""

    def test_funnel_palette_length(self) -> None:
        """Funnel palette has exactly 5 colours — one per stage."""
        assert len(FUNNEL_PALETTE) == 5

    def test_primary_color_is_hex(self) -> None:
        """Primary colour is a valid 7-char hex string."""
        assert COLOR_PRIMARY.startswith("#")
        assert len(COLOR_PRIMARY) == 7


class TestAnnotation:
    """Insight annotations attach to the figure without disturbing data."""

    def test_adds_annotation(self) -> None:
        """An insight annotation lands in layout.annotations."""
        fig = go.Figure(go.Bar(x=[1], y=[1]))
        add_insight_annotation(fig, text="<b>Key insight</b>", x=0.5, y=0.9)
        assert len(fig.layout.annotations) == 1
        assert fig.layout.annotations[0].text == "<b>Key insight</b>"
