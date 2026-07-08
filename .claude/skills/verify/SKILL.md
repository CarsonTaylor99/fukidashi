---
name: verify
description: Build/launch/drive recipe for verifying fukidashi changes (FastAPI server + single-file frontend reader).
---

# Verifying fukidashi

## Launch

```bash
cd /home/claude/fukidashi
.venv/bin/uvicorn fukidashi.server:app --host 127.0.0.1 --port 8015 &
```

Port 8014 is the user's real instance (start.sh) — use another port.
Ollama is usually offline in the dev sandbox: `/api/status` reports it,
and anything past the bubble stage (bible/translate) can't run live.

## Surfaces

- **API**: curl `/api/volumes`, `/api/volumes/test/reader?language=english`,
  `/api/volumes/{slug}/pages/0` and `/cleaned/0`, `/fonts/*.ttf`.
  The `test` volume (3 pages, translated) and `demo-daifuku-the-cat`
  (5 pages) in `data/library/` are safe fixtures — back up any json you
  regenerate (translations index blocks by position; new OCR misaligns them).
- **Pipeline stages** run standalone without Ollama:
  `.venv/bin/python -m fukidashi.ocr test` (GPU, ~15s/page + model load),
  `.venv/bin/python -m fukidashi.bubbles test`.
- **Reader UI**: no browser in this environment. Drive it with jsdom
  against the live server — working harness at
  `scratchpad/drive_reader.mjs` pattern. Required stubs in `beforeParse`:
  `IntersectionObserver` (fake, keep targets to trigger manually),
  `fetch` rebased to the server URL, `EventSource`, nominal
  `clientWidth/clientHeight/scrollHeight`, `scrollIntoView`, `scrollBy`,
  `Range.prototype.getClientRects` (jsdom lacks it — fitText crashes
  otherwise). jsdom never fires `img.onload` (no canvas): call
  `img.onload()` by hand after setting src. `getComputedStyle` does not
  resolve `var()` — assert against the `--letter` declaration in the
  stylesheet instead.

## Gotchas

- Text-fit code (fitText/fitShaped) needs real layout; jsdom only proves
  it doesn't throw. Anything visual (shape-outside flow, spacing, font
  rendering) needs the user's browser (Brave) — say so in the report.
- `force=true` on `/translate` deletes ocr/bubbles/bible/translations —
  only probe it against a throwaway volume, or with Ollama down (the 503
  short-circuits before deletion; verify files survive).
