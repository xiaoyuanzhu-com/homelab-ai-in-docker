# Dockerfile for Homelab AI Services
#
# Lean build: Main env contains only API server + web crawling.
# ML dependencies are in worker environments, installed on-demand.

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

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    aria2 \
    wget \
    ccache \
    ffmpeg \
    # LibreOffice runtime dependencies
    libcairo2 \
    libcups2 \
    libdbus-1-3 \
    libglib2.0-0 \
    libgtk-3-0 \
    libpango-1.0-0 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install LibreOffice 25.8 for Office document conversion
RUN curl -L https://mirror-hk.koddos.net/tdf/libreoffice/stable/25.8.3/deb/x86_64/LibreOffice_25.8.3_Linux_x86-64_deb.tar.gz -o /tmp/libreoffice.tar.gz && \
    tar -xzf /tmp/libreoffice.tar.gz -C /tmp && \
    dpkg -i /tmp/LibreOffice_25.8.3*_Linux_x86-64_deb/DEBS/*.deb || apt-get install -f -y && \
    rm -rf /tmp/libreoffice.tar.gz /tmp/LibreOffice_25.8.3*

# Install uv for faster Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Install hfd (HuggingFace downloader with aria2 support and mirror compatibility)
RUN curl -L https://gist.githubusercontent.com/padeoe/697678ab8e528b85a2a7bddafea1fa4f/raw/hfd.sh -o /usr/local/bin/hfd && \
    chmod +x /usr/local/bin/hfd

# Copy main project dependency files
COPY pyproject.toml ./

# Install Python dependencies (lean - no ML libs)
RUN uv pip install --system --no-cache -r pyproject.toml

# Copy application code
COPY main.py ./
COPY src/ ./src/

# Install package in editable mode
RUN uv pip install --system --no-cache -e .

# Copy worker environment templates (pyproject.toml + lock files only, no .venv)
# These will be installed on-demand when workers are first spawned
COPY envs/ ./envs/

# Install Playwright browsers for crawl4ai at build time
RUN crawl4ai-setup

# Copy documentation files and build with MkDocs
COPY docs/ ./docs/
COPY mkdocs.yml ./
RUN mkdocs build

# Copy built UI from builder stage
COPY --from=ui-builder /ui/dist ./ui/dist

# Create data directories
# - /haid/data/models: HuggingFace model cache
# - /haid/data/paddlex: PaddleX OCR cache
# - /haid/data/envs: On-demand worker environments (optional external mount)
RUN mkdir -p /haid/data/models /haid/data/paddlex

# Environment configuration
# Worker environments are installed to persistent volume if HAID_ENVS_DIR is set
# Otherwise they're installed in-place in /haid/envs (ephemeral)
ENV HAID_DATA_DIR=/haid/data
ENV HF_HOME=/haid/data/models
ENV PADDLE_PDX_CACHE_HOME=/haid/data/paddlex
# Optional: Set HAID_ENVS_DIR=/haid/data/envs to persist worker envs across restarts

# Expose port
EXPOSE 12310

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:12310/api/health || exit 1

# Run the application
CMD ["python", "main.py"]
