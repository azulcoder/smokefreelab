"""BigQuery client wrapper and data-access utilities for SmokeFreeLab.

This module is the single entry point for every BigQuery interaction in the
project. Notebooks and modules must call ``run_query`` — never instantiate
``google.cloud.bigquery.Client`` directly — so that authentication,
parameterization, and cost-guarding are consistent.

Business context
----------------
All queries bill against the user's GCP Sandbox project (1 TB scanned per
month, free tier). Every query that hits the GA4 public sample must use
``_TABLE_SUFFIX`` filtering so that scans stay well within that budget — the
full five-stage funnel query is engineered to scan under 2 GB per invocation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from google.cloud import bigquery
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from collections.abc import Mapping

ScalarParam = str | int | float | bool
"""Types accepted as BigQuery scalar query parameters."""


def _find_env_file() -> str | None:
    """Locate ``.env`` by walking up from this module.

    Makes ``BQConfig`` robust to the caller's working directory — notebooks,
    scripts, and nbconvert all end up in different CWDs, and pydantic-settings
    otherwise resolves ``env_file`` relative to CWD.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / ".env"
        if candidate.is_file():
            return str(candidate)
    return None


class BQConfig(BaseSettings):
    """BigQuery configuration, populated from environment variables or ``.env``.

    Attributes
    ----------
    gcp_project_id : str
        The GCP project that owns the compute for each query (your BigQuery
        Sandbox project). Queries are billed here, not against the public
        dataset's host project.
    bq_location : str
        Dataset location. Defaults to ``US`` because the public GA4 sample
        lives in US; do not change unless you are pointing at a private export.
    ga4_project : str
        The GCP project that hosts the GA4 sample (``bigquery-public-data``).
    ga4_dataset : str
        The GA4 sample dataset name.
    """

    gcp_project_id: str
    bq_location: str = "US"
    ga4_project: str = "bigquery-public-data"
    ga4_dataset: str = "ga4_obfuscated_sample_ecommerce"

    model_config = SettingsConfigDict(
        env_file=_find_env_file(),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


def get_client(config: BQConfig | None = None) -> bigquery.Client:
    """Return an authenticated BigQuery client.

    Relies on Application Default Credentials; run
    ``gcloud auth application-default login`` once per workstation before the
    first call.

    Parameters
    ----------
    config : BQConfig, optional
        Configuration. If omitted, reads from environment / ``.env``.

    Returns
    -------
    google.cloud.bigquery.Client
        A client bound to the configured project and location.
    """
    cfg = config if config is not None else BQConfig()
    return bigquery.Client(project=cfg.gcp_project_id, location=cfg.bq_location)


def run_query(
    sql: str,
    *,
    config: BQConfig | None = None,
    params: Mapping[str, ScalarParam] | None = None,
) -> pd.DataFrame:
    """Execute a BigQuery SQL statement and return the result as a DataFrame.

    Parameters
    ----------
    sql : str
        The SQL text. Use ``@name`` placeholders for parameterization.
        Never interpolate caller-supplied values via f-strings — that is an
        injection surface.
    config : BQConfig, optional
        Configuration. Defaults to environment-driven.
    params : Mapping[str, ScalarParam], optional
        Named query parameters. Types are inferred from Python types
        (``str`` → STRING, ``int`` → INT64, ``float`` → FLOAT64,
        ``bool`` → BOOL).

    Returns
    -------
    pandas.DataFrame
        Query result.

    Business context
    ----------------
    Parameterization is non-negotiable for any query that takes user- or
    notebook-supplied values. The BQ Storage API is enabled for faster
    downloads of large result sets (common for the funnel + attribution
    queries).
    """
    client = get_client(config)
    job_config = bigquery.QueryJobConfig(
        query_parameters=_to_query_params(params or {}),
    )
    job = client.query(sql, job_config=job_config)
    return job.result().to_dataframe(create_bqstorage_client=True)


def estimate_query_bytes(
    sql: str,
    *,
    config: BQConfig | None = None,
    params: Mapping[str, ScalarParam] | None = None,
) -> int:
    """Return the bytes a query would scan, without executing it.

    Use this as a cost-guard before running an unfamiliar query against the
    GA4 sample: the Sandbox tier has a 1 TB/month scan budget, and a
    misfiltered ``_TABLE_SUFFIX`` can burn through it in a single run.

    Parameters
    ----------
    sql : str
        The SQL text.
    config : BQConfig, optional
        Configuration.
    params : Mapping[str, ScalarParam], optional
        Named query parameters, same semantics as ``run_query``.

    Returns
    -------
    int
        Estimated bytes scanned.
    """
    client = get_client(config)
    job_config = bigquery.QueryJobConfig(
        dry_run=True,
        use_query_cache=False,
        query_parameters=_to_query_params(params or {}),
    )
    job = client.query(sql, job_config=job_config)
    return int(job.total_bytes_processed)


def load_sql(name: str, sql_dir: Path | None = None) -> str:
    """Load a ``.sql`` file from the project's ``sql/`` directory by stem.

    Parameters
    ----------
    name : str
        File stem without extension (e.g. ``"01_funnel_decomposition"``).
    sql_dir : pathlib.Path, optional
        Override the resolved ``sql/`` directory (useful in tests).

    Returns
    -------
    str
        Raw SQL text.

    Raises
    ------
    FileNotFoundError
        If the file does not exist under the resolved directory.
    """
    directory = sql_dir if sql_dir is not None else _default_sql_dir()
    path = directory / f"{name}.sql"
    if not path.exists():
        raise FileNotFoundError(f"SQL file not found: {path}")
    return path.read_text(encoding="utf-8")


def _to_query_params(
    params: Mapping[str, ScalarParam],
) -> list[bigquery.ScalarQueryParameter]:
    """Map Python values to BigQuery scalar parameters."""
    type_map: dict[type, str] = {
        bool: "BOOL",
        int: "INT64",
        float: "FLOAT64",
        str: "STRING",
    }
    out: list[bigquery.ScalarQueryParameter] = []
    for name, value in params.items():
        bq_type = type_map.get(type(value))
        if bq_type is None:
            raise TypeError(
                f"Unsupported parameter type for {name!r}: {type(value).__name__}. "
                "Supported: str, int, float, bool."
            )
        out.append(bigquery.ScalarQueryParameter(name, bq_type, value))
    return out


def _default_sql_dir() -> Path:
    """Locate the project's ``sql/`` directory by walking up from this module."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "sql"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate `sql/` directory relative to the package. " "Pass `sql_dir=` explicitly."
    )
