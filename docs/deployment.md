# Deployment Guide

## GitHub Container Registry (GHCR)

The project is configured with GitHub Actions to automatically build and push Docker images to GitHub Container Registry.

### Automatic Builds

Images are automatically built and pushed on:

- **Push to `main` branch** → `ghcr.io/xiaoyuanzhu-com/homelab-ai-in-docker:latest`
- **Git tags (e.g., `v1.0.0`)** → `ghcr.io/xiaoyuanzhu-com/homelab-ai-in-docker:1.0.0`
- **Pull requests** → Build only (no push)

### Available Tags

- `latest` - Latest commit on main branch
- `v{version}` - Specific version (e.g., `v0.1.0`)
- `v{major}.{minor}` - Major.minor version (e.g., `v0.1`)
- `v{major}` - Major version (e.g., `v0`)
- `main-{sha}` - Specific commit on main branch

### Multi-Architecture Support

Images are built for:
- `linux/amd64` (x86_64)
- `linux/arm64` (ARM64, including Apple Silicon)

### Using Pre-built Images

```bash
# Pull latest
docker pull ghcr.io/xiaoyuanzhu-com/homelab-ai-in-docker:latest

# Pull specific version
docker pull ghcr.io/xiaoyuanzhu-com/homelab-ai-in-docker:v0.1.0

# Run
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/data:/haid/data \
  ghcr.io/xiaoyuanzhu-com/homelab-ai-in-docker:latest
```

### Making a Release

1. Create and push a version tag:
```bash
git tag v0.1.0
git push origin v0.1.0
```

2. GitHub Actions will automatically:
   - Build the Docker image
   - Push to GHCR with version tags
   - Create attestations for provenance

### Image Details

- **Base Image**: `python:3.13-slim`
- **Working Directory**: `/haid`
- **Exposed Port**: `8000`
- **Health Check**: `/api/health` every 30s
- **Size**: ~2-3GB (includes Python, ML libraries, Chromium)

### Permissions

The workflow requires:
- `contents: read` - Read repository code
- `packages: write` - Push to GHCR

These are automatically available via `GITHUB_TOKEN`.

## Local Development

For local development, build from source:

```bash
docker-compose up -d
```

Or manually:

```bash
docker build -t homelab-ai .
docker run -d -p 8000:8000 -v $(pwd)/data:/haid/data homelab-ai
```

## Production Deployment

### Docker Compose (Recommended)

1. Update `docker-compose.yml` to use GHCR image
2. Create volume directory:
```bash
mkdir -p data
```

3. Start the service:
```bash
docker-compose up -d
```

#### Remote Chrome Browser (Optional)

The API supports using a remote Chrome instance for web crawling via Chrome DevTools Protocol (CDP). This allows you to offload browser operations to a separate service for better resource management and scalability.

**Configuration Options**:

1. **Environment Variable (Default for all requests)**:

   Edit `docker-compose.yml` and uncomment the `CHROME_CDP_URL` line:
   ```yaml
   environment:
     - PYTHONUNBUFFERED=1
     - CHROME_CDP_URL=http://172.16.2.2:9223
   ```

2. **Per-Request Parameter** (overrides environment variable):
   ```bash
   curl -X POST http://localhost:8000/api/crawl \
     -H "Content-Type: application/json" \
     -d '{
       "url": "https://example.com",
       "chrome_cdp_url": "http://172.16.2.2:9223"
     }'
   ```

**Setting Up Remote Chrome**:

You'll need to run Chrome with remote debugging enabled on your network:

```bash
# Example: Run Chrome in Docker with remote debugging
docker run -d \
  --name chrome \
  -p 9223:3000 \
  --shm-size 2gb \
  browserless/chrome:latest
```

Or use an existing Chrome instance with `--remote-debugging-port`:
```bash
google-chrome --remote-debugging-port=9223 --headless
```

**Benefits of Remote Chrome**:
- Better resource isolation between crawling and API processes
- Horizontal scaling of browser instances
- Centralized browser management
- Reduced memory usage on API server
- Support for managed browser services (e.g., browserless, BlitzBrowser)

**How It Works**:

When a remote Chrome URL is provided, the API uses crawl4ai's `browser_mode="cdp"` to connect to the remote browser via Chrome DevTools Protocol. The remote browser handles all page rendering and JavaScript execution, while the API server only processes the extracted content.

### Kubernetes / Homelab

Example pod configuration:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: homelab-ai
spec:
  containers:
  - name: api
    image: ghcr.io/xiaoyuanzhu-com/homelab-ai-in-docker:latest
    ports:
    - containerPort: 8000
    volumeMounts:
    - name: data
      mountPath: /haid/data
    resources:
      limits:
        memory: "8Gi"
        cpu: "4"
      requests:
        memory: "4Gi"
        cpu: "2"
  volumes:
  - name: data
    hostPath:
      path: /path/to/data
```

## Storage Requirements

- **Model Cache** (`/haid/data`): 2-8GB depending on models used
  - `all-MiniLM-L6-v2`: ~90MB
  - `blip-image-captioning-base`: ~1-2GB
- **History Database** (`/haid/data/history.db`): Grows over time, typically <100MB (SQLite)
- **Container Size**: ~2-3GB

## Monitoring

Health check endpoint:
```bash
curl http://localhost:8000/api/health
```

Container logs:
```bash
docker-compose logs -f
```

## Troubleshooting

### Image Pull Issues

If the image is private, authenticate first:
```bash
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
```

### Build Failures

Check GitHub Actions logs at:
`https://github.com/xiaoyuanzhu-com/homelab-ai-in-docker/actions`

### Model Download Issues

First run will download models (2-8GB). Ensure:
- Sufficient disk space
- Internet connectivity
- Volume is properly mounted

## Security

- Images include attestations for build provenance
- Base image is official Python slim variant
- No secrets or credentials in image
- Model data stored in mounted volumes only
