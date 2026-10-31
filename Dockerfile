# ==============================================================================
# BankGuard — Production Containerfile (Task 7.3)
# Multi-purpose container for FastAPI Serving Service and Streamlit Dashboard.
# ==============================================================================

FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install runtime dependencies (libgomp1 required for LightGBM OpenMP support)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root application user
RUN useradd -m -u 1000 -s /bin/bash bankguard

WORKDIR /app

# Layer cache optimization: install Python dependencies first
COPY requirements.txt pyproject.toml /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source, configurations, trained model artifacts, and reports
# (Raw Kaggle data is strictly excluded via .dockerignore)
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY artifacts/ /app/artifacts/
COPY report/ /app/report/
COPY .streamlit/ /app/.streamlit/
COPY dashboard.py README.md docker-entrypoint.sh /app/

# Install BankGuard package in editable mode without reinstalling dependencies
RUN pip install --no-deps -e .

# Set execution permissions and ownership
RUN chmod +x /app/docker-entrypoint.sh && \
    chown -R bankguard:bankguard /app

# Switch to non-root application user
USER bankguard

# Expose ports for FastAPI serving (8000) and Streamlit dashboard (8501)
EXPOSE 8000 8501

# Healthcheck probe using standard Python urllib (no external dependency)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["api"]

