# fukidashi 吹き出し

*fukidashi* — the speech bubble in manga.

A manga reader that translates the **entire work with full context**, not bubble by bubble. Line-by-line tools (Google Translate et al.) translate each bubble blind — losing who is speaking, what happened three pages ago, honorifics, running jokes, and names that look like common nouns. fukidashi reads the way a human translator does:

1. **OCR the whole volume** — [mokuro](https://github.com/kha-white/mokuro) (comic-text-detector + manga-ocr) finds every speech bubble and its pixel coordinates.
2. **Context pass** — the full text is read start to finish by a local LLM, which builds a *story bible*: characters, name renderings, relationships, speech styles, honorifics policy, recurring terms and jokes.
3. **Translation pass** — each page is drafted three times at spread temperatures, then an editor pass picks or splices the most natural line per bubble (with the story bible plus a rolling window of previous pages in context), into **any target language**.
4. **Read** — a web reader flows each translation into the bubble's actual shape (long lines through the belly, short at the crown, like a human letterer); hover any bubble to see the original Japanese.

Everything runs locally: [Ollama](https://ollama.com) drives the translation. The default model is an abliterated gemma3 27B — same quality as stock, but it won't refuse or sanitize adult works (doujinshi etc.).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt   # torch is large; grab coffee
ollama pull aqualaguna/gemma-3-27b-it-abliterated-GGUF:q4_k_m   # ~17GB VRAM; SFW-only alternative: gemma3:27b
./start.sh                                   # http://localhost:8014
```

## Usage

1. Open the web UI, drop in a `.cbz`/`.zip` of page images, give it a title.
2. Type a target language (anything: `English`, `Spanish`, `Brazilian Portuguese`…) and hit **Translate**. Progress streams live; first ever run also downloads the OCR models.
3. Hit **Read**. Arrow keys turn pages (manga order: ← is *next*), `t` toggles translations, `v` toggles long-strip view, `b` opens the story bible, hover a bubble for the original text.
4. **⤓ ZIP** in the reader bar downloads the volume with the translations typeset into the pages — a zip of images (rename to `.cbz` and it opens in any manga reader). Fully-translated pages are rendered at native resolution with the same lettering the reader shows; untranslated pages pass through untouched.

Translation is resumable — if a run is interrupted, re-running skips already-translated pages. A story bible is built once per volume per language and reused.

## No manga handy?

Generate a 5-page synthetic test volume (a story specifically designed to break context-free translators — a cat named 大福…):

```bash
.venv/bin/python scripts/make_demo_cbz.py --font /path/to/NotoSansJP.ttf -o demo.cbz
```

## Configuration

| env var | default | |
|---|---|---|
| `FUKIDASHI_MODEL` | `aqualaguna/gemma-3-27b-it-abliterated-GGUF:q4_k_m` | Ollama model for all passes |
| `FUKIDASHI_DRAFTS` | `3` | drafts per page before the editor pass; `1` = single-shot, ~4× faster |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |
| `FUKIDASHI_LIBRARY` | `data/library` | where volumes live |

## Layout

```
fukidashi/
  ocr.py        mokuro wrapper → ocr.json (blocks + boxes, reading order)
  bubbles.py    flood-fill bubble detection → bubbles.json + cleaned/ pages
  bible.py      context pass → bible.json (story bible)
  translate.py  page-by-page pass → translations.<lang>.json
  pipeline.py   background job: OCR → bubbles → bible → translate, SSE progress
  server.py     FastAPI app + reader API
frontend/       single-page library + reader UI
data/library/   your volumes (images + JSON, no database)
```
