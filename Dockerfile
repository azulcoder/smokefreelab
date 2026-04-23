FROM python:3.11-slim

LABEL maintainer="Ahmad Zulfan"
LABEL project="smokefreelab"
LABEL description="A/B testing and experimentation framework on the GA4 public sample"

# Install uv
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy lockfile and manifest first for better layer caching
COPY pyproject.toml uv.lock* ./

# Install dependencies
RUN uv sync --frozen --no-dev || uv sync --no-dev

# Copy source
COPY src/ ./src/
COPY app/ ./app/
COPY sql/ ./sql/

EXPOSE 8501

# Default: launch Streamlit Experiment Designer
CMD ["uv", "run", "streamlit", "run", "app/experiment_designer.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
