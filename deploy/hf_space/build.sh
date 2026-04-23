#!/usr/bin/env bash
# Assemble a self-contained, pushable Hugging Face Space bundle at
# deploy/hf_space/_bundle/. The bundle is the single source of truth for
# what gets pushed to the Space — see deploy/hf_space/DEPLOY.md.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
BUNDLE="${HERE}/_bundle"

echo "• cleaning ${BUNDLE}"
rm -rf "${BUNDLE}"
mkdir -p "${BUNDLE}/smokefreelab/experiment" "${BUNDLE}/smokefreelab/analytics"

echo "• copying app.py"
cp "${ROOT}/app/experiment_designer.py" "${BUNDLE}/app.py"

echo "• copying smokefreelab source subset"
cp "${ROOT}/src/smokefreelab/__init__.py"                   "${BUNDLE}/smokefreelab/__init__.py"
cp "${ROOT}/src/smokefreelab/experiment/__init__.py"        "${BUNDLE}/smokefreelab/experiment/__init__.py"
cp "${ROOT}/src/smokefreelab/experiment/ab_test.py"         "${BUNDLE}/smokefreelab/experiment/ab_test.py"
cp "${ROOT}/src/smokefreelab/analytics/__init__.py"         "${BUNDLE}/smokefreelab/analytics/__init__.py"
cp "${ROOT}/src/smokefreelab/analytics/viz.py"              "${BUNDLE}/smokefreelab/analytics/viz.py"

echo "• copying Space metadata"
cp "${HERE}/README.md"         "${BUNDLE}/README.md"
cp "${HERE}/requirements.txt"  "${BUNDLE}/requirements.txt"

echo "• bundle contents:"
(cd "${BUNDLE}" && find . -type f | sort | sed 's|^\./|  |')

echo ""
echo "✅ Bundle ready at ${BUNDLE}"
echo "   Push it to Hugging Face with the steps in deploy/hf_space/DEPLOY.md."
