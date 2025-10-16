# Dockerfile for Homelab AI Services
FROM python:3.13-slim

# Set working directory
WORKDIR /haid

# Install system dependencies required by the AI libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Copy dependency files
COPY pyproject.toml ./

# Install Python dependencies
RUN uv pip install --system -r pyproject.toml

# Install playwright browsers for crawl4ai
RUN uv pip install --system playwright && \
    playwright install --with-deps chromium && \
    crawl4ai-setup

# Copy application code
COPY main.py ./
COPY src/ ./src/

# Create data directory for model cache
RUN mkdir -p /haid/data/embedding /haid/data/image-caption

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run the application
CMD ["python", "main.py"]
