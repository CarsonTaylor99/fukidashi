# fukidashi 吹き出し

*fukidashi* — the speech bubble in manga.

A manga reader that translates the **entire work with full context**, not bubble by bubble. Line-by-line tools (Google Translate et al.) translate each bubble blind — losing who is speaking, what happened three pages ago, honorifics, running jokes, and names that look like common nouns. fukidashi reads the way a human translator does:

1. **OCR the whole volume** — [mokuro](https://github.com/kha-white/mokuro) (comic-text-detector + manga-ocr) finds every speech bubble and its pixel coordinates.
2. **Context pass** — the full text is read start to finish by a local LLM, which builds a *story bible*: characters, name renderings, relationships, speech styles, honorifics policy, recurring terms and jokes.
3. **Translation pass** — each page is translated with the story bible plus a rolling window of previous pages in context, into **any target language**.
4. **Read** — a web reader overlays translations on the actual bubbles; hover any bubble to see the original Japanese.

Everything runs locally: [Ollama](https://ollama.com) drives the translation (default model `qwen2.5:14b`), OCR runs on your GPU if you have one.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt   # torch is large; grab coffee
ollama pull qwen2.5:14b
./start.sh                                   # http://localhost:8014
```

## Usage

1. Open the web UI, drop in a `.cbz`/`.zip` of page images, give it a title.
2. Type a target language (anything: `English`, `Spanish`, `Brazilian Portuguese`…) and hit **Translate**. Progress streams live; first ever run also downloads the OCR models.
3. Hit **Read**. Arrow keys turn pages (manga order: ← is *next*), `t` toggles translations, `b` opens the story bible, hover a bubble for the original text.

Translation is resumable — if a run is interrupted, re-running skips already-translated pages. A story bible is built once per volume per language and reused.

## No manga handy?

Generate a 5-page synthetic test volume (a story specifically designed to break context-free translators — a cat named 大福…):

```bash
.venv/bin/python scripts/make_demo_cbz.py --font /path/to/NotoSansJP.ttf -o demo.cbz
```

## Configuration

| env var | default | |
|---|---|---|
| `FUKIDASHI_MODEL` | `qwen2.5:14b` | Ollama model for both passes |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |
| `FUKIDASHI_LIBRARY` | `data/library` | where volumes live |

## Layout

```
fukidashi/
  ocr.py        mokuro wrapper → ocr.json (blocks + boxes, reading order)
  bible.py      context pass → bible.json (story bible)
  translate.py  page-by-page pass → translations.<lang>.json
  pipeline.py   background job: OCR → bible → translate, SSE progress
  server.py     FastAPI app + reader API
frontend/       single-page library + reader UI
data/library/   your volumes (images + JSON, no database)
```
