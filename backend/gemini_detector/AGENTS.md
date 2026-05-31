# AGENTS.md (Scope: `gemini_detector/`)

## Purpose
- Provide a SAHI `DetectionModel` adapter that calls OpenRouter-hosted Gemini models.
- Detect two categories only: `instrument_tag` (`category_id=1`) and `line_number` (`category_id=2`).

## Pre-Integration Checklist (Target Repo)
- Confirm Python version supports `X | Y` type hints (Python 3.10+).
- Install required packages: `sahi`, `openai`, `numpy`, `opencv-python`.
- Configure `OPENROUTER_API_KEY` in runtime environment (do not hardcode in source).
- Optionally set `OPENROUTER_MODEL`; default is `google/gemini-3-flash-preview`.
- Confirm outbound network access to `https://openrouter.ai/api/v1`.
- Decide error mode explicitly. Default is fail-fast (`raise_on_error=True`).
- Set `raise_on_error=False` only if caller handles empty predictions safely.
- Decide debug artifact policy explicitly. Default debug is off (`debug=False`).
- If debug is enabled, ensure write access for `debug_root` (default `results/debug`).

## Implementation Rules
- Preferred usage: `detector = GeminiSahiDetector(confidence_threshold=..., device="cpu")`.
- Preferred usage: `detector.set_config(GeminiSahiConfig(openrouter_api_key=..., model_name=...))`.
- Keep SAHI postprocessing in caller code (NMM/NMS thresholds are app-level decisions).
- Treat `model_path` API-key fallback as backward compatibility only; do not use for new code.
- Preserve category IDs `1` and `2` unless you also update downstream label assumptions.

## Verification Before Merge
- Run compile check: `python -m py_compile gemini_detector/*.py`.
- Run import check: `python -c "from gemini_detector import GeminiSahiDetector, GeminiSahiConfig"`.
- Run one real inference on a small sample image with valid API key.
- Assert all of the following in logs/artifacts.
- Response JSON parses successfully.
- Bounding boxes are valid (`x1 < x2`, `y1 < y2`).
- SAHI returns non-crashing `ObjectPrediction` objects.
- If inference fails, confirm failure is visible (exception or `last_error`).

## Security and Ops Notes
- Never commit `.env` with real API keys.
- Avoid enabling debug artifacts in production unless needed for incident triage.
- If debug is enabled, review artifacts for sensitive plant/document text before sharing.
