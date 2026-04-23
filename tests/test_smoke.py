"""Smoke test to confirm the package is importable and `make test` passes on day one."""

from __future__ import annotations


def test_package_importable() -> None:
    """SmokeFreeLab package imports and exposes a version string."""
    import smokefreelab

    assert hasattr(smokefreelab, "__version__")
    assert isinstance(smokefreelab.__version__, str)
    assert smokefreelab.__version__ != ""


def test_version_is_semver_like() -> None:
    """Version string is parseable as MAJOR.MINOR.PATCH."""
    import smokefreelab

    parts = smokefreelab.__version__.split(".")
    assert len(parts) == 3
    for part in parts:
        assert part.isdigit(), f"Version part '{part}' is not numeric"
