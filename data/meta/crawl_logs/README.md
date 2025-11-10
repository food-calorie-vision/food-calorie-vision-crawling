# Crawl Log Format

Each crawl run should drop a JSON file in this directory with the shape:

```json
{
  "run_id": "2025-01-01_poc_a",
  "timestamp": "2025-01-01T12:34:56Z",
  "classes": [
    {"name": "기장", "requested": 100, "downloaded": 94, "errors": 3, "folder": "기장"},
    {"name": "보리", "requested": 100, "downloaded": 102, "errors": 1, "folder": "보리"}
  ],
  "source": ["google_images", "naver_images"],
  "user": "agent-name",
  "notes": "throttled to 1 req/sec",
  "history": [
    {
      "timestamp": "2025-01-01T12:34:56Z",
      "classes": [
        {"name": "기장", "requested": 100, "downloaded": 94, "errors": 3, "folder": "기장"}
      ],
      "source": ["google_images"],
      "notes": "dry-run first batch"
    },
    {
      "timestamp": "2025-01-01T14:20:10Z",
      "classes": [
        {"name": "보리", "requested": 100, "downloaded": 102, "errors": 1, "folder": "보리"}
      ],
      "source": ["naver_images"],
      "notes": "second batch"
    }
  ]
}
```

- Use ISO 8601 timestamps (UTC).
- Include any throttling/delay parameters and failure reasons inside `notes`.
- `folder` captures the actual directory/filename prefix used after sanitizing (for auditing collisions).
- Re-running the same `run_id` appends `history` entries and merges class summaries. Default DuckDuckGo 수집 시 `source`는 `duckduckgo_images_playwright`를 사용하세요.
- Keep filenames consistent: `<run_id>.json`.
