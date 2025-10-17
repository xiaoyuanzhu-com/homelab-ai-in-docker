# API Reference

HTTP endpoint details and examples

## Endpoint

```
POST /api/crawl
```

## Request Body

```json
{
  "url": "https://example.com",
  "screenshot": false,
  "wait_for_js": true
}
```

## Parameters

- `url` (string, required) - The URL to crawl
- `screenshot` (boolean, optional) - Capture screenshot (default: false)
- `wait_for_js` (boolean, optional) - Wait for JavaScript execution (default: true)

## Response

```json
{
  "request_id": "uuid",
  "url": "https://example.com/",
  "title": "Example Domain",
  "markdown": "# Example Domain\n\nThis domain...",
  "html": "<!DOCTYPE html>...",
  "screenshot_base64": null,
  "fetch_time_ms": 1580,
  "success": true
}
```

## cURL Example

```bash
curl -X POST {{API_BASE_URL}}/api/crawl \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com",
    "wait_for_js": true
  }'
```
