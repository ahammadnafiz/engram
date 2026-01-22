# syntax=docker/dockerfile:1

# =============================================================================
# Engram - AI Memory Library for LLM Applications
# Multi-stage Dockerfile for development and production
# =============================================================================

# -----------------------------------------------------------------------------
# Base stage with Python and system dependencies
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS base

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd --gid 1000 engram \
    && useradd --uid 1000 --gid engram --shell /bin/bash --create-home engram

WORKDIR /app

# -----------------------------------------------------------------------------
# Builder stage for installing dependencies
# -----------------------------------------------------------------------------
FROM base AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy package files (pyproject.toml needs src for dynamic version)
COPY pyproject.toml ./
COPY src/ ./src/

# Install dependencies
RUN pip install --upgrade pip \
    && pip install build \
    && pip install ".[all,dev]"

# -----------------------------------------------------------------------------
# Development stage
# -----------------------------------------------------------------------------
FROM base AS development

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy source code (for editable install)
COPY --chown=engram:engram . .

# Re-install in editable mode for development
RUN pip install -e .

USER engram

# Default command for development
CMD ["python", "-m", "pytest", "tests/", "-v"]

# -----------------------------------------------------------------------------
# Production stage
# -----------------------------------------------------------------------------
FROM base AS production

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only necessary files
COPY --chown=engram:engram src/ ./src/
COPY --chown=engram:engram pyproject.toml ./

# Install package
RUN pip install .

USER engram

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import engram; print('OK')" || exit 1

# Default command
CMD ["python", "-c", "import engram; print(f'Engram v{engram.__version__} ready')"]
