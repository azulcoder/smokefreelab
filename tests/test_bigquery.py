"""Unit tests for the BigQuery client wrapper.

Tests use pytest-mock and never hit real BigQuery. Integration tests that
require live BQ are marked with ``@pytest.mark.bigquery`` and excluded from
default CI runs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from smokefreelab.data.bigquery import (
    BQConfig,
    _to_query_params,
    estimate_query_bytes,
    load_sql,
    run_query,
)


def test_bqconfig_loads_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """BQConfig reads env vars case-insensitively and applies defaults."""
    monkeypatch.setenv("GCP_PROJECT_ID", "my-sandbox-project")
    monkeypatch.delenv("BQ_LOCATION", raising=False)
    monkeypatch.delenv("GA4_DATASET", raising=False)
    monkeypatch.delenv("GA4_PROJECT", raising=False)

    cfg = BQConfig(_env_file=None)
    assert cfg.gcp_project_id == "my-sandbox-project"
    assert cfg.bq_location == "US"
    assert cfg.ga4_project == "bigquery-public-data"
    assert cfg.ga4_dataset == "ga4_obfuscated_sample_ecommerce"


def test_run_query_returns_dataframe(mocker: MockerFixture) -> None:
    """run_query returns the DataFrame produced by the underlying BQ client."""
    expected = pd.DataFrame({"event_date": ["2020-11-01"], "users": [42]})

    mock_client = mocker.MagicMock()
    mock_client.query.return_value.result.return_value.to_dataframe.return_value = expected
    mocker.patch(
        "smokefreelab.data.bigquery.bigquery.Client",
        return_value=mock_client,
    )

    cfg = BQConfig(gcp_project_id="test-project", _env_file=None)
    result = run_query("SELECT 1", config=cfg)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 2)
    assert list(result.columns) == ["event_date", "users"]
    mock_client.query.assert_called_once()


def test_run_query_passes_parameters(mocker: MockerFixture) -> None:
    """run_query forwards scalar parameters to the BQ job config."""
    mock_client = mocker.MagicMock()
    mock_client.query.return_value.result.return_value.to_dataframe.return_value = pd.DataFrame(
        {"x": [1]}
    )
    mocker.patch(
        "smokefreelab.data.bigquery.bigquery.Client",
        return_value=mock_client,
    )

    cfg = BQConfig(gcp_project_id="test-project", _env_file=None)
    run_query(
        "SELECT @n AS x",
        config=cfg,
        params={"n": 1, "flag": True, "label": "hello"},
    )

    job_config = mock_client.query.call_args.kwargs["job_config"]
    params_by_name = {p.name: (p.type_, p.value) for p in job_config.query_parameters}
    assert params_by_name["n"] == ("INT64", 1)
    assert params_by_name["flag"] == ("BOOL", True)
    assert params_by_name["label"] == ("STRING", "hello")


def test_estimate_query_bytes_returns_int(mocker: MockerFixture) -> None:
    """estimate_query_bytes returns total_bytes_processed without executing."""
    mock_job = mocker.MagicMock()
    mock_job.total_bytes_processed = 1_234_567_890
    mock_client = mocker.MagicMock()
    mock_client.query.return_value = mock_job
    mocker.patch(
        "smokefreelab.data.bigquery.bigquery.Client",
        return_value=mock_client,
    )

    cfg = BQConfig(gcp_project_id="test-project", _env_file=None)
    bytes_scanned = estimate_query_bytes("SELECT 1", config=cfg)

    assert bytes_scanned == 1_234_567_890
    mock_job.result.assert_not_called()


def test_to_query_params_rejects_unsupported_type() -> None:
    """_to_query_params raises TypeError on unsupported Python types."""
    with pytest.raises(TypeError, match="Unsupported parameter type"):
        _to_query_params({"when": object()})  # type: ignore[dict-item]


def test_load_sql_round_trips_file(tmp_path: Path) -> None:
    """load_sql reads the file matching `{name}.sql` under the given directory."""
    sql_dir = tmp_path / "sql"
    sql_dir.mkdir()
    (sql_dir / "demo.sql").write_text("SELECT 1 AS answer;\n", encoding="utf-8")

    result = load_sql("demo", sql_dir=sql_dir)

    assert result == "SELECT 1 AS answer;\n"


def test_load_sql_raises_when_missing(tmp_path: Path) -> None:
    """load_sql raises FileNotFoundError for a missing stem."""
    with pytest.raises(FileNotFoundError):
        load_sql("does_not_exist", sql_dir=tmp_path)
