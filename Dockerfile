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
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

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

# Copy built UI from builder stage
COPY --from=ui-builder /ui/dist ./ui/dist

# Create data directories for model cache, crawl4ai, and playwright
RUN mkdir -p /haid/data/embedding /haid/data/image-caption /haid/data/crawl4ai /haid/data/playwright

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run the application
CMD ["python", "main.py"]
