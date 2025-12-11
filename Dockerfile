# Dockerfile for Homelab AI Services
#
# Lean build: Main env contains only API server, no ML dependencies.
# ML dependencies are in worker environments, installed on-demand.
#
# Worker environments require different Python versions:
# - Python 3.13: transformers, paddle, whisper, hunyuan, crawl4ai, markitdown, screenitshot
# - Python 3.12: deepseek (flash-attn wheel compatibility)
# uv manages multiple Python versions automatically.

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
# Use slim base - uv will manage Python versions for workers
FROM debian:bookworm-slim

# Set working directory
WORKDIR /haid

# Install system dependencies
# Note: Python comes from uv, not the base image
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    aria2 \
    wget \
    ccache \
    ffmpeg \
    ca-certificates \
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
    # Playwright dependencies (for crawl4ai and screenitshot workers)
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Install LibreOffice 25.8 for Office document conversion
RUN curl -L https://mirror-hk.koddos.net/tdf/libreoffice/stable/25.8.3/deb/x86_64/LibreOffice_25.8.3_Linux_x86-64_deb.tar.gz -o /tmp/libreoffice.tar.gz && \
    tar -xzf /tmp/libreoffice.tar.gz -C /tmp && \
    dpkg -i /tmp/LibreOffice_25.8.3*_Linux_x86-64_deb/DEBS/*.deb || apt-get install -f -y && \
    rm -rf /tmp/libreoffice.tar.gz /tmp/LibreOffice_25.8.3*

# Install uv - manages Python versions and packages
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Install Python versions needed by workers (uv manages these)
# - Python 3.13: main env + most workers
# - Python 3.12: deepseek worker (flash-attn wheel compatibility)
RUN uv python install 3.13 3.12

# Install hfd (HuggingFace downloader with aria2 support and mirror compatibility)
RUN curl -L https://gist.githubusercontent.com/padeoe/697678ab8e528b85a2a7bddafea1fa4f/raw/hfd.sh -o /usr/local/bin/hfd && \
    chmod +x /usr/local/bin/hfd

# Copy main project dependency files
COPY pyproject.toml uv.lock .python-version ./

# Install Python dependencies for main env (lean - no ML libs)
RUN uv sync --frozen --no-dev

# Copy application code
COPY main.py ./
COPY src/ ./src/

# Copy worker environment templates (pyproject.toml + lock files only)
# The .dockerignore excludes .venv directories from envs/
COPY envs/ ./envs/

# Copy documentation files and build with MkDocs
COPY docs/ ./docs/
COPY mkdocs.yml ./
RUN uv run mkdocs build

# Copy built UI from builder stage
COPY --from=ui-builder /ui/dist ./ui/dist

# Create data directories
# - /haid/data/models: HuggingFace model cache
# - /haid/data/paddlex: PaddleX OCR cache
# - /haid/data/envs: On-demand worker environments (optional external mount)
RUN mkdir -p /haid/data/models /haid/data/paddlex

# Environment configuration
ENV HAID_DATA_DIR=/haid/data
ENV HF_HOME=/haid/data/models
ENV PADDLE_PDX_CACHE_HOME=/haid/data/paddlex
# Worker environments installed to data volume for persistence across container restarts
# When /haid/data is a mounted volume, envs persist and don't need reinstalling
ENV HAID_ENVS_DIR=/haid/data/envs

# Expose port
EXPOSE 12310

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:12310/api/health || exit 1

# Run the application using uv (which activates the correct venv)
CMD ["uv", "run", "python", "main.py"]
