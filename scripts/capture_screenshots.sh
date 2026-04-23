#!/usr/bin/env bash
# Launch Streamlit headlessly, capture the 6 README screenshots, stop the server.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PORT=8513
LOG=$(mktemp)

echo "• launching streamlit on :${PORT}"
.venv/bin/streamlit run app/experiment_designer.py \
  --server.headless true \
  --server.port "${PORT}" \
  --server.runOnSave false \
  --browser.gatherUsageStats false \
  < /dev/null > "${LOG}" 2>&1 &
STREAMLIT_PID=$!

cleanup() {
  echo "• stopping streamlit (pid=${STREAMLIT_PID})"
  kill "${STREAMLIT_PID}" 2>/dev/null || true
  wait "${STREAMLIT_PID}" 2>/dev/null || true
  rm -f "${LOG}"
}
trap cleanup EXIT

echo "• waiting for server to come up"
ready=0
for _ in $(seq 1 80); do
  if curl -fs "http://localhost:${PORT}/_stcore/health" > /dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 0.5
done

if [ "${ready}" = "0" ]; then
  echo "✗ streamlit did not respond on :${PORT} within 40s"
  echo "  streamlit log:"
  sed 's/^/    /' "${LOG}"
  exit 1
fi

echo "• server ready, running capture"
.venv/bin/python scripts/capture_screenshots.py
