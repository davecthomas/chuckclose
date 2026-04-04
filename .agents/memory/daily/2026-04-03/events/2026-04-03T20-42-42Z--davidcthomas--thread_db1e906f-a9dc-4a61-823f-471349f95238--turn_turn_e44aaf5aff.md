---
timestamp: "2026-04-03T20:42:42Z"
author: "davidcthomas"
branch: "main"
thread_id: "db1e906f-a9dc-4a61-823f-471349f95238"
turn_id: "turn_e44aaf5aff"
decision_candidate: false
ai_generated: true
ai_model: "claude-unknown"
ai_tool: "claude"
ai_surface: "claude-code"
ai_executor: "local-agent"
related_adrs:
files_touched:
  - "pyproject.toml"
verification:
  - "Tracked repo changes were detected in the working tree."
---

## Why

- The fix is already in v2.5.0 — it checks `model_extra` before falling back to `os.environ`. `GOOGLE_AUTH_METHOD=api_key` will now be read correctly from `.env`.

## Repo changes

- Updated pyproject.toml

## Evidence

- Tracked repo changes were detected in the working tree.

## Next

- Review the generated shard and summary, then explicitly commit and push them with the related code changes if ready.
