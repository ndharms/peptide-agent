# Multi-stage build for peptide-agent
# Stage 1: Builder
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better caching
COPY pyproject.toml ./
COPY src/ ./src/
COPY README.md LICENSE ./

# Install the package and dependencies in a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir .

# Stage 2: Runtime
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy data directory (if needed at runtime)
COPY data/ /app/data/

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PEPTIDE_DATA_DIR=/app/data \
    PEPTIDE_FAISS_DIR=/app/data/index/faiss

# Create cache directory
RUN mkdir -p /app/data/index/faiss

# Add healthcheck (for when running as a service)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD peptide-agent --help || exit 1

# Set entrypoint
ENTRYPOINT ["peptide-agent"]
CMD ["--help"]
