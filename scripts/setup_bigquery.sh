#!/usr/bin/env bash
# Setup script for BigQuery Sandbox access.
# Run once per machine. No credit card needed — sandbox is free up to 1TB scanned/month.
#
# Usage:  bash scripts/setup_bigquery.sh

set -euo pipefail

echo "────────────────────────────────────────────────────────────────"
echo "  SmokeFreeLab — BigQuery Sandbox setup"
echo "────────────────────────────────────────────────────────────────"

# ─── Step 1: verify gcloud CLI ────────────────────────────────────────────
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found."
    echo ""
    echo "Install it first:"
    echo "  macOS:   brew install google-cloud-sdk"
    echo "  Linux:   curl https://sdk.cloud.google.com | bash"
    echo "  Windows: https://cloud.google.com/sdk/docs/install"
    echo ""
    exit 1
fi
echo "✅ gcloud CLI found: $(gcloud --version | head -1)"

# ─── Step 2: authenticate ──────────────────────────────────────────────────
echo ""
echo "Step 2/4: authenticate to Google Cloud."
echo "This opens a browser window. Sign in with a Google account that has"
echo "BigQuery Sandbox enabled (free tier, no credit card required)."
echo ""
read -rp "Press ENTER to continue..."
gcloud auth application-default login

# ─── Step 3: get or create project ─────────────────────────────────────────
echo ""
echo "Step 3/4: Google Cloud project."
echo ""
CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null || echo "")
if [[ -n "$CURRENT_PROJECT" ]]; then
    echo "Current project: $CURRENT_PROJECT"
    read -rp "Use this project? [Y/n] " use_current
    if [[ "$use_current" != "n" && "$use_current" != "N" ]]; then
        PROJECT_ID="$CURRENT_PROJECT"
    fi
fi

if [[ -z "${PROJECT_ID:-}" ]]; then
    echo ""
    echo "Create a new project at: https://console.cloud.google.com/projectcreate"
    echo "Name it something memorable, e.g. 'smokefreelab-az'."
    read -rp "Enter your project ID: " PROJECT_ID
    gcloud config set project "$PROJECT_ID"
fi

# ─── Step 4: verify BigQuery access ────────────────────────────────────────
echo ""
echo "Step 4/4: smoke-test BigQuery connectivity to the public GA4 sample."
echo ""

bq query --use_legacy_sql=false --format=pretty --max_rows=3 \
'SELECT event_name, COUNT(*) AS n
 FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_20210131`
 GROUP BY event_name
 ORDER BY n DESC
 LIMIT 3'

# ─── Step 5: write .env ────────────────────────────────────────────────────
echo ""
echo "Writing .env with your project ID..."
if [[ -f .env ]]; then
    echo "⚠️  .env already exists. Not overwriting. Manually check GCP_PROJECT_ID matches: $PROJECT_ID"
else
    cp .env.example .env
    # Replace the placeholder (works on both GNU and BSD sed)
    sed -i.bak "s|your-project-id-here|$PROJECT_ID|" .env && rm -f .env.bak
    echo "✅ .env written."
fi

echo ""
echo "────────────────────────────────────────────────────────────────"
echo "  ✅ BigQuery setup complete."
echo "────────────────────────────────────────────────────────────────"
echo ""
echo "Free tier: 1TB scanned/month, 10GB storage/month. Write queries that"
echo "filter on _TABLE_SUFFIX for date range — a naive SELECT * can blow"
echo "through your quota in one query."
echo ""
echo "Next: open notebooks/01_eda_ga4_sample.ipynb and run it top-to-bottom."
