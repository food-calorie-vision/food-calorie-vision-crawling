# Data Source Register

| source_name | url_or_identifier | license | usage_scope | notes | last_checked |
|-------------|------------------|---------|-------------|-------|--------------|
| DuckDuckGo Images (Playwright) | https://duckduckgo.com/?iax=images&ia=images | TOS (search) | Research, non-commercial | `scripts/crawl_images_playwright.py` 사용, `--show-browser` 권장 (headless 제한), 동의 다이얼로그 자동 처리, JavaScript 동적 클래스 탐색(SZ76bwIlqO8BBoqOLqYV 같은 랜덤 클래스), 아이콘 필터링(.ico, 50x50 이하 자동 제거) | 2025-11-07 |
| example: Google Images | https://images.google.com | TOS (search) | Research, non-commercial | Limit rate; follow robots.txt | 2025-01-01 |
| example: Food-101 | https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz | CC BY-NC-SA 4.0 | Research only | Contains 101 classes; see README for splits | 2025-01-01 |

- Add one row per crawl/dataset.
- If a source requires purchase/credential, mention storage location and contact.
- Keep `last_checked` in `YYYY-MM-DD` format.
