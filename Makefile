.PHONY: help install install-dev test lint format typecheck clean docs serve-docs run-app deploy-hf screenshots

help:
	@echo "SmokeFreeLab — make targets"
	@echo ""
	@echo "  install        Install production dependencies via uv"
	@echo "  install-dev    Install dev dependencies (includes pytest, ruff, mypy)"
	@echo "  test           Run pytest with coverage"
	@echo "  lint           Run ruff linting"
	@echo "  format         Auto-format code with black + ruff"
	@echo "  typecheck      Run mypy strict mode"
	@echo "  docs           Build mkdocs site"
	@echo "  serve-docs     Serve mkdocs site locally on :8000"
	@echo "  run-app        Launch Streamlit Experiment Designer"
	@echo "  deploy-hf      Build Hugging Face Space bundle (see deploy/hf_space/DEPLOY.md)"
	@echo "  screenshots    Capture the 6 README screenshots via Playwright"
	@echo "  clean          Remove build/test/cache artifacts"

install:
	uv sync

install-dev:
	uv sync --extra dev --extra docs --extra attribution
	uv run pre-commit install

test:
	uv run pytest -v

test-fast:
	uv run pytest -v -m "not slow and not bigquery"

lint:
	uv run ruff check src tests app

format:
	uv run black src tests app
	uv run ruff check --fix src tests app

typecheck:
	uv run mypy src app

check: lint typecheck test

docs:
	uv run mkdocs build

serve-docs:
	uv run mkdocs serve

run-app:
	uv run streamlit run app/experiment_designer.py

deploy-hf:
	./deploy/hf_space/build.sh

screenshots:
	./scripts/capture_screenshots.sh

clean:
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	rm -rf site/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +
