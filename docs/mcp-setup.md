# MCP (Model Context Protocol) Setup Guide

This guide explains how to use the Homelab AI Services as a remote MCP server with Claude Code and other MCP clients.

## Overview

The Homelab AI Services exposes all AI capabilities via the **Model Context Protocol (MCP)**, allowing AI assistants like Claude Code to access your self-hosted AI tools remotely.

**Protocol:** Streamable HTTP (modern MCP standard, recommended for 2025+)
**Endpoint:** `/mcp`
**Transport:** HTTP/HTTPS (supports both local and remote connections)

## Available MCP Tools

The following tools are exposed via MCP:

### 1. `crawl_web`
Scrape web pages with JavaScript rendering and extract clean Markdown content.

**Parameters:**
- `url` (string, required): URL to crawl
- `screenshot` (boolean, optional): Whether to capture a screenshot (default: false)
- `wait_for_js` (boolean, optional): Whether to wait for JavaScript execution (default: true)

**Returns:**
- `url`: The crawled URL
- `title`: Page title
- `markdown`: Content in Markdown format
- `screenshot`: Base64-encoded screenshot (if requested)
- `success`: Whether the crawl succeeded

### 2. `embed_text`
Generate embeddings (vector representations) from text for semantic search and similarity matching.

**Parameters:**
- `text` (string or array of strings, required): Text to embed
- `model` (string, optional): Embedding model to use (default: "all-MiniLM-L6-v2")

**Returns:**
- `embeddings`: List of embedding vectors
- `model`: Model used
- `dimensions`: Embedding dimensionality

### 3. `generate_text`
Generate text from a prompt using a language model.

**Parameters:**
- `prompt` (string, required): Input prompt
- `model` (string, optional): Model ID (e.g., "qwen-0.5b", default: "qwen-0.5b")
- `max_new_tokens` (integer, optional): Maximum tokens to generate (default: 512)
- `temperature` (float, optional): Sampling temperature (default: 0.7)
- `top_p` (float, optional): Nucleus sampling threshold (default: 0.9)

**Returns:**
- `generated_text`: The generated text
- `model`: Model used
- `prompt_tokens`: Number of input tokens
- `completion_tokens`: Number of generated tokens

### 4. `caption_image`
Generate a natural language caption for an image.

**Parameters:**
- `image_base64` (string, required): Base64-encoded image data
- `model` (string, optional): Model ID (e.g., "blip-base", default: "blip-base")
- `prompt` (string, optional): Optional prompt to guide caption generation

**Returns:**
- `caption`: Generated caption text
- `model`: Model used

### 5. `ocr_image`
Extract text from an image using OCR (Optical Character Recognition).

**Parameters:**
- `image_base64` (string, required): Base64-encoded image data
- `model` (string, optional): Model ID (default: "paddleocr")

**Returns:**
- `text`: Extracted text
- `boxes`: List of bounding boxes with text and coordinates
- `model`: Model used

### 6. `transcribe_audio`
Transcribe speech from an audio file.

**Parameters:**
- `audio_base64` (string, required): Base64-encoded audio data (WAV, MP3, etc.)
- `model` (string, optional): Model ID (e.g., "whisper-tiny", default: "whisper-tiny")
- `language` (string, optional): Language code (e.g., "en", "zh")

**Returns:**
- `text`: Transcribed text
- `language`: Detected or specified language
- `model`: Model used

### 7. `get_hardware_info`
Get hardware information including GPU stats, memory, and system metrics.

**Parameters:** None

**Returns:**
- `gpu`: GPU information (if available)
- `cpu`: CPU usage
- `memory`: Memory usage
- `disk`: Disk usage

## Connecting from Claude Code

### Prerequisites

1. **Homelab AI Services running**
   ```bash
   # Docker (recommended)
   docker-compose up -d

   # Or local development
   uv run python main.py
   ```

2. **Service accessible** at `http://localhost:12310` (or your configured host/port)

3. **MCP client** (Claude Code, or any MCP-compatible client)

### Configuration for Claude Code

Claude Code supports remote MCP servers via Streamable HTTP. To connect:

1. **Ensure the service is running:**
   ```bash
   curl http://localhost:12310/api/health
   # Should return: {"status":"healthy"}
   ```

2. **Configure MCP server in Claude Code:**

   The exact configuration method depends on your Claude Code setup. Generally, you'll need to add an MCP server configuration pointing to:

   ```
   http://localhost:12310/mcp
   ```

3. **Test the connection:**

   Once configured, Claude Code should be able to discover and use all available MCP tools automatically.

### Configuration for Remote Access

To use the MCP server remotely (outside localhost):

1. **Ensure the service is exposed** via reverse proxy (nginx, Caddy, etc.) or port forwarding

2. **Use HTTPS** for production deployments (MCP supports both HTTP and HTTPS)

3. **Update the MCP endpoint URL** to your public URL:
   ```
   https://your-domain.com/mcp
   ```

### Authentication (Optional)

For production deployments, you may want to add authentication to the MCP endpoint. Future versions will support:
- Bearer token authentication
- OAuth 2.0 integration

Currently, the MCP endpoint is **open** (no authentication required). For security in production:
- Use a reverse proxy with authentication (nginx, Caddy, Cloudflare Access, etc.)
- Restrict access via firewall rules or VPN
- Deploy behind a private network

## Examples

### Using MCP Tools via Claude Code

Once connected, you can use the tools naturally in your conversation with Claude Code:

```
You: Can you crawl https://example.com and summarize it?
Claude Code: [Uses crawl_web tool internally]

You: Generate an embedding for "machine learning tutorial"
Claude Code: [Uses embed_text tool internally]

You: What's the GPU memory usage?
Claude Code: [Uses get_hardware_info tool internally]
```

### Manual Testing (HTTP)

You can test the MCP endpoint manually using curl or any HTTP client:

```bash
# Test MCP discovery endpoint
curl http://localhost:12310/mcp

# Example tool invocation (Streamable HTTP protocol)
curl -X POST http://localhost:12310/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "method": "tools/call",
    "params": {
      "name": "get_hardware_info",
      "arguments": {}
    }
  }'
```

## Troubleshooting

### MCP endpoint not available

**Symptom:** `/mcp` returns 404 or connection refused

**Solutions:**
1. Check if MCP package is installed: `uv pip list | grep mcp`
2. Check server logs for MCP mount errors: `docker logs homelab-ai` or check console output
3. Ensure the service is running: `curl http://localhost:12310/api/health`

### Tools not discovered by Claude Code

**Symptom:** Claude Code connects but doesn't see any tools

**Solutions:**
1. Verify the endpoint URL is correct (should end with `/mcp`)
2. Check server logs for errors during MCP initialization
3. Test the endpoint manually using curl (see examples above)

### Model not found errors

**Symptom:** Tool calls fail with "model not found" or "skill not found" errors

**Solutions:**
1. Download required models first using the skills API:
   ```bash
   curl -X POST http://localhost:12310/api/skills/download \
     -H "Content-Type: application/json" \
     -d '{"skill_id": "qwen-0.5b"}'
   ```
2. Check available models: `curl http://localhost:12310/api/skills`

### Performance issues

**Symptom:** Slow response times or timeouts

**Solutions:**
1. Check GPU availability: `curl http://localhost:12310/api/hardware`
2. Verify models are loaded (first call is slower due to model loading)
3. Adjust `model_idle_timeout_seconds` setting to keep models in memory longer
4. Consider using smaller models for faster inference

## Advanced Configuration

### Custom Model Configuration

Models are configured in the skills manifest (`src/api/skills/skills_manifest.json`). You can:
- Add custom models
- Configure quantization settings
- Set default parameters

See [tech-design.md](tech-design.md) for details on the skills system.

### Settings API

The MCP server respects all application settings configured via the Settings API:

```bash
# Get current settings
curl http://localhost:12310/api/settings

# Update model idle timeout (seconds before unloading from GPU)
curl -X POST http://localhost:12310/api/settings/model_idle_timeout_seconds \
  -H "Content-Type: application/json" \
  -d '{"value": 300}'
```

## Protocol Details

### Streamable HTTP Transport

The MCP server uses **Streamable HTTP** transport (MCP protocol v2024-11-05+), which:
- Works as standard HTTP/HTTPS (no persistent connections required)
- Supports both stateless and stateful implementations
- Compatible with all standard web infrastructure (load balancers, CDNs, etc.)
- Supports streaming responses for long-running operations

### Compatibility

- **Claude Code**: Full support (latest versions)
- **Other MCP clients**: Any client supporting Streamable HTTP transport
- **Legacy SSE clients**: Not supported (use Streamable HTTP instead)

## Security Considerations

### Production Deployment Checklist

For production deployments:

- [ ] Use HTTPS (not HTTP)
- [ ] Add authentication (reverse proxy or future built-in support)
- [ ] Restrict network access (firewall, VPN, private network)
- [ ] Monitor access logs
- [ ] Set resource limits (rate limiting, request timeouts)
- [ ] Use read-only API keys for HuggingFace (if downloading models)

### Data Privacy

- The MCP server runs **locally** on your infrastructure
- No data is sent to third-party services (except HuggingFace for model downloads)
- All inference happens on your hardware
- Request history is stored locally in SQLite database

## Resources

- [Model Context Protocol Documentation](https://modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [Homelab AI Technical Design](tech-design.md)
