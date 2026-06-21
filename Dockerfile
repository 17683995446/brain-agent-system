# ============================================================
# Multi-stage Dockerfile for BrainAgentSystem
# Stage 1: Builder - compile and install dependencies
# Stage 2: Runtime - minimal production image
# ============================================================

# ---------- Stage 1: Builder ----------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ ./src/

# Install dependencies into a target directory
RUN pip install --no-cache-dir --target=/install -e ".[dev]"

# ---------- Stage 2: Runtime ----------
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install runtime system dependencies (no gcc/g++)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local/lib/python3.11/site-packages

# Copy application code
COPY pyproject.toml .
COPY configs/ ./configs/
COPY src/ ./src/
COPY data/ ./data/

# Create non-root user
RUN groupadd -r brainuser && useradd -r -g brainuser -d /app -s /sbin/nologin brainuser && \
    chown -R brainuser:brainuser /app

USER brainuser

# Expose metrics port
EXPOSE 9090

# Health check: probe the /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:9090/health || exit 1

# Default command: run the simulation server
CMD ["brain-agent", "run", "--steps", "0"]
