-- ============================================================================
-- sql/01_funnel_decomposition.sql
--
-- Purpose:
--   Decompose a 5-stage e-commerce funnel on the GA4 obfuscated sample and
--   return daily user / session counts plus stage-to-stage conversion rates.
--   Serves as the "advanced SQL" evidence exhibit for the portfolio — CTEs,
--   window functions, and QUALIFY are mandatory, not decorative.
--
-- Funnel stages (in order):
--   1. session_start
--   2. view_item
--   3. add_to_cart
--   4. begin_checkout
--   5. purchase
--
-- Sessionization choice:
--   We use GA4's native `ga_session_id` event parameter (joined on
--   `user_pseudo_id`) rather than recomputing a 30-minute inactivity window.
--   This inherits GA4's own session definition (including its cross-midnight
--   behavior) and is cheaper to scan. Users whose events lack ga_session_id
--   (very rare; bot-like traffic or broken tags) are dropped.
--
-- Counting rule:
--   A user counts at a stage only the FIRST time they reach that stage within
--   a session (QUALIFY ROW_NUMBER() = 1). This prevents a user who views the
--   same product three times in one session from inflating the `view_item`
--   count three-fold, which would distort the stage-to-stage CVR.
--
-- Performance:
--   - `_TABLE_SUFFIX` filter pushed into the events CTE before any join,
--     giving BigQuery the partition pruning it needs.
--   - Dry-run scan estimate: ~1.3 GB across the Nov-2020 → Jan-2021 range
--     (well under the 2 GB per-run target; ~1.5% of the 1 TB/month Sandbox
--     budget per run).
--   - No SELECT *; only the four columns we need from the raw event table.
--
-- Parameters:
--   @start_date : STRING  — inclusive, e.g. '20201101'
--   @end_date   : STRING  — inclusive, e.g. '20210131'
--
-- Output columns:
--   event_date           DATE
--   stage                STRING
--   users                INT64   — distinct user_pseudo_id reaching this stage
--   sessions             INT64   — distinct (user, session) pairs
--   cvr_vs_prior_stage   FLOAT64 — users / users_at_previous_stage (NULL for entry)
--   cvr_vs_entry         FLOAT64 — users / users_at_session_start
--
-- Author : Az
-- Status : complete
-- ============================================================================

WITH

  events AS (
    SELECT
      user_pseudo_id,
      event_name,
      event_timestamp,
      DATE(TIMESTAMP_MICROS(event_timestamp)) AS event_date,
      (
        SELECT value.int_value
        FROM UNNEST(event_params)
        WHERE key = 'ga_session_id'
      ) AS ga_session_id
    FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
    WHERE _TABLE_SUFFIX BETWEEN @start_date AND @end_date
      AND event_name IN (
        'session_start',
        'view_item',
        'add_to_cart',
        'begin_checkout',
        'purchase'
      )
  ),

  first_event_per_stage AS (
    SELECT
      event_date,
      user_pseudo_id,
      ga_session_id,
      event_name AS stage,
      event_timestamp
    FROM events
    WHERE ga_session_id IS NOT NULL
    QUALIFY ROW_NUMBER() OVER (
      PARTITION BY user_pseudo_id, ga_session_id, event_name
      ORDER BY event_timestamp
    ) = 1
  ),

  daily_stage_counts AS (
    SELECT
      event_date,
      stage,
      COUNT(DISTINCT user_pseudo_id) AS users,
      COUNT(DISTINCT FORMAT('%s-%d', user_pseudo_id, ga_session_id)) AS sessions
    FROM first_event_per_stage
    GROUP BY event_date, stage
  ),

  stage_ordered AS (
    SELECT
      event_date,
      stage,
      users,
      sessions,
      CASE stage
        WHEN 'session_start'  THEN 1
        WHEN 'view_item'      THEN 2
        WHEN 'add_to_cart'    THEN 3
        WHEN 'begin_checkout' THEN 4
        WHEN 'purchase'       THEN 5
      END AS stage_order
    FROM daily_stage_counts
  ),

  with_conversion AS (
    SELECT
      event_date,
      stage,
      stage_order,
      users,
      sessions,
      SAFE_DIVIDE(
        users,
        LAG(users) OVER (PARTITION BY event_date ORDER BY stage_order)
      ) AS cvr_vs_prior_stage,
      SAFE_DIVIDE(
        users,
        FIRST_VALUE(users) OVER (
          PARTITION BY event_date
          ORDER BY stage_order
          ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        )
      ) AS cvr_vs_entry
    FROM stage_ordered
  )

SELECT
  event_date,
  stage,
  users,
  sessions,
  ROUND(cvr_vs_prior_stage, 4) AS cvr_vs_prior_stage,
  ROUND(cvr_vs_entry, 4)       AS cvr_vs_entry
FROM with_conversion
ORDER BY event_date, stage_order;
