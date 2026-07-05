"""Translation pass: page by page, with the story bible and a rolling
window of previous pages in context.

Each page's blocks are numbered and sent in reading order; the model
returns one translation per block via Ollama structured outputs, so the
mapping back onto bubble coordinates is exact.
"""

from .config import CONTEXT_PAGES
from . import bible as bible_mod
from . import library, llm

SYSTEM = """\
You are a professional manga translator translating Japanese into {target_lang}.

You have already read the whole work and prepared this story bible — follow it strictly
for names, honorifics, recurring terms, and tone:

STORY BIBLE:
{bible}

You will be given the previous pages (for flow and context) and the current page's
text blocks, numbered in reading order. Translate EACH numbered block of the current
page into {target_lang}.

Rules:
- Return exactly one translation per numbered block, in the same order.
- Translate naturally for {target_lang} manga readers; match each speaker's voice.
- Sound effects / onomatopoeia: give a short {target_lang} equivalent (e.g. "WHAM").
- OCR noise: if a block is garbled fragments, translate what is recoverable; if nothing
  is, return the block unchanged.
- Never merge blocks, never leave one out.
"""


def _schema(n_blocks: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "translations": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": n_blocks,
                "maxItems": n_blocks,
            }
        },
        "required": ["translations"],
    }


def _page_prompt(page: dict, history: list[tuple[dict, list[str]]]) -> str:
    parts = []
    if history:
        ctx = []
        for prev, prev_tr in history:
            lines = [f"[page {prev['page'] + 1}]"]
            for b, t in zip(prev["blocks"], prev_tr):
                lines.append(f"{b['text']}  →  {t}")
            ctx.append("\n".join(lines))
        parts.append("PREVIOUS PAGES (already translated):\n" + "\n\n".join(ctx))
    blocks = "\n".join(f"{i + 1}. {b['text']}" for i, b in enumerate(page["blocks"]))
    parts.append(f"CURRENT PAGE (page {page['page'] + 1}) — translate these blocks:\n{blocks}")
    return "\n\n".join(parts)


def _translate_page(system: str, page: dict, history) -> list[str]:
    n = len(page["blocks"])
    prompt = _page_prompt(page, history)
    for attempt in range(2):
        result = llm.chat_json(system, prompt, _schema(n),
                               temperature=0.3 if attempt == 0 else 0.6)
        translations = result.get("translations", [])
        if len(translations) == n:
            return [str(t) for t in translations]
    raise llm.OllamaError(
        f"page {page['page'] + 1}: expected {n} translations, got {len(translations)}"
    )


def run(slug: str, target_lang: str, log=print) -> None:
    pages = library.load_json(slug, "ocr.json")
    if not pages:
        raise RuntimeError("no OCR data — run OCR first")
    saved = library.load_json(slug, "bible.json")
    if not saved or saved.get("target_lang") != target_lang:
        bible = bible_mod.build(slug, target_lang, log=log)
    else:
        bible = saved["bible"]

    system = SYSTEM.format(target_lang=target_lang, bible=bible)
    out_name = f"translations.{target_lang.lower().replace(' ', '-')}.json"
    # Resume support: keep already-translated pages if the file exists.
    existing = library.load_json(slug, out_name) or {}
    result = {k: v for k, v in existing.items()}

    history: list[tuple[dict, list[str]]] = []
    todo = [p for p in pages if p["blocks"]]
    for i, page in enumerate(todo):
        key = str(page["page"])
        if key in result:
            history.append((page, result[key]))
            history[:] = history[-CONTEXT_PAGES:]
            continue
        log(f"translating page {page['page'] + 1} ({i + 1}/{len(todo)})")
        translations = _translate_page(system, page, history)
        result[key] = translations
        history.append((page, translations))
        history[:] = history[-CONTEXT_PAGES:]
        if (i + 1) % 5 == 0 or i == len(todo) - 1:
            library.save_json(slug, out_name, result)
    library.save_json(slug, out_name, result)
    log(f"translation complete: {len(todo)} pages → {target_lang}")
