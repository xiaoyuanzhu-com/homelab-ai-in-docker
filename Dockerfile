# Dockerfile for Homelab AI Services

# Stage 1: Build the UI
FROM node:lts AS ui-builder

WORKDIR /ui

# Copy package files and install dependencies
COPY ui/package.json ui/package-lock.json ./
RUN npm ci

# Copy UI source and build
COPY ui/ ./
RUN npm run build

# Stage 2: Final image
FROM python:3.13

# Set working directory
WORKDIR /haid

# Install system dependencies required by the AI libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    aria2 \
    wget \
    ccache \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
# Torch wheels expect their shared libraries to be on the runtime search path.
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.13/site-packages/torch/lib"

# Install hfd (HuggingFace downloader with aria2 support and mirror compatibility)
RUN curl -L https://gist.githubusercontent.com/padeoe/697678ab8e528b85a2a7bddafea1fa4f/raw/hfd.sh -o /usr/local/bin/hfd && \
    chmod +x /usr/local/bin/hfd

# Copy dependency files
COPY pyproject.toml ./

# Install Python dependencies (--no-cache to reduce image size)
RUN uv pip install --system --no-cache -r pyproject.toml

# Copy application code
COPY main.py ./
COPY src/ ./src/

# Install package in editable mode with GPU support (--no-cache to reduce image size)
# PyTorch 2.7.0 with CUDA 12.6, flash-attn 2.8.3 from pre-built wheels
# Index configuration is in pyproject.toml [tool.uv.index]
RUN uv pip install --system --no-cache -e .[gpu]

# Install Playwright browsers for crawl4ai at build time
# This installs browsers to default location inside the container
RUN crawl4ai-setup

# Copy built UI from builder stage
COPY --from=ui-builder /ui/dist ./ui/dist

# Create data directory for PaddleX cache (OCR models)
RUN mkdir -p /haid/data/paddlex

# Set PaddleX cache directory to persist OCR models
ENV PADDLE_PDX_CACHE_HOME=/haid/data/paddlex

# Expose port
EXPOSE 12310

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:12310/api/health || exit 1

# Run the application
CMD ["python", "main.py"]
