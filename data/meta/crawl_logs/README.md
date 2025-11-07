# Crawl Log Format

Each crawl run should drop a JSON file in this directory with the shape:

```json
{
  "run_id": "2025-01-01_poc_a",
  "timestamp": "2025-01-01T12:34:56Z",
  "classes": [
    {"name": "기장", "requested": 100, "downloaded": 94, "errors": 3},
    {"name": "보리", "requested": 100, "downloaded": 102, "errors": 1}
  ],
  "source": ["google_images", "naver_images"],
  "user": "agent-name",
  "notes": "throttled to 1 req/sec"
}
```

- Use ISO 8601 timestamps (UTC).
- Include any throttling/delay parameters and failure reasons inside `notes`.
- Keep filenames consistent: `<run_id>.json`.
