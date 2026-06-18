FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY configs/ ./configs/
COPY src/ ./src/

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[dev]" --break-system-packages 2>/dev/null || pip install --no-cache-dir -e ".[dev]"

# Default command: run tests
CMD ["pytest", "-v", "--tb=short"]
