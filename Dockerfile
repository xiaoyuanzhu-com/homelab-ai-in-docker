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
# Ensure CUDA/NVIDIA shared libraries installed via pip are on the runtime search path
# This is required on Python 3.13 where pip's nvidia-* site hooks may not set paths early enough.
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.13/site-packages/torch/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/cuda_runtime/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/cublas/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/cudnn/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/cufft/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/cuda_nvrtc/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/nvjitlink/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/cuda_cupti/lib:${LD_LIBRARY_PATH}"

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

# Copy documentation files and build with MkDocs
# Note: mkdocs-material is already installed via pyproject.toml above
COPY docs/ ./docs/
COPY mkdocs.yml ./
RUN mkdocs build

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
