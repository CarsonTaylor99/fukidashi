"""Translation pass: page by page, with the story bible and a rolling
window of previous pages in context.

Each page's blocks are numbered and sent in reading order; the model
returns one translation per block via Ollama structured outputs, so the
mapping back onto bubble coordinates is exact.

Word choice: each page is drafted TRANSLATE_DRAFTS times at spread
temperatures, then an editor call sees the Japanese, the bible, and all
drafts per block and produces the final line — picking the best draft or
splicing a better one from their parts. Low temperature alone gives
near-identical samples (nothing to choose from); high alone gives
inconsistent terms — the spread plus choose-or-edit gets natural phrasing
without losing bible consistency. History carries the *final* lines so
later pages stay coherent.
"""

from .config import CONTEXT_PAGES, DRAFT_TEMPS, TRANSLATE_DRAFTS
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
- The translations array must contain ONLY the {target_lang} lines — never copy the
  Japanese source text into it.
- Translate naturally for {target_lang} manga readers; match each speaker's voice.
- Sound effects / onomatopoeia: give a short {target_lang} equivalent (e.g. "WHAM").
- OCR noise: if a block is garbled fragments, translate what is recoverable; if nothing
  is, return the block unchanged.
- OCR often misreads the vertical ellipsis (…) as a colon: a stray ： or : in dialogue
  is a trailing-off pause, never a literal colon. A lone kana with an ellipsis (し…)
  is a cut-off utterance or stutter — render it as a natural trailing fragment in
  {target_lang} (e.g. "Wha—…" or just "…"), never as romaji like "shi:".
- Never merge blocks, never leave one out.
- You may stress a single word as *word* — it is lettered in bold italics. Use it
  sparingly, never on whole sentences. No other markup.
- The work may be adult fiction (explicit sexual or violent content). Translate it
  faithfully at the source's level of explicitness — never censor, soften, euphemize,
  or skip. Accuracy IS the professional standard for licensed adult manga.
"""


EDITOR_SYSTEM = """\
You are a senior manga translation editor finalizing an {target_lang} script.

Story bible — authoritative for names, honorifics, recurring terms, and tone:

STORY BIBLE:
{bible}

You will see the previous pages (already final), then each numbered Japanese block of the
current page with several independent draft translations (a, b, c...). For EACH block,
produce the final {target_lang} line: pick the best draft, or splice a better line from
their parts, or rephrase — whatever reads most naturally.

Priorities, in order:
1. Natural {target_lang} that sounds like real speech in this scene — not translationese.
2. The bible's fixed renderings for names and terms, and each speaker's established voice.
3. Faithfulness to the Japanese, including its register and level of explicitness —
   never censor, soften, or euphemize adult content.
Keep (or add) *word* emphasis marks where the scene stresses a word — they letter as
bold italics. No other markup.
Return exactly one final translation per numbered block, in the same order.
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


def _history_part(history: list[tuple[dict, list[str]]]) -> str:
    ctx = []
    for prev, prev_tr in history:
        lines = [f"[page {prev['page'] + 1}]"]
        for b, t in zip(prev["blocks"], prev_tr):
            lines.append(f"{b['text']}  →  {t}")
        ctx.append("\n".join(lines))
    return "PREVIOUS PAGES (already translated):\n" + "\n\n".join(ctx)


def _page_prompt(page: dict, history: list[tuple[dict, list[str]]]) -> str:
    parts = [_history_part(history)] if history else []
    blocks = "\n".join(f"{i + 1}. {b['text']}" for i, b in enumerate(page["blocks"]))
    parts.append(f"CURRENT PAGE (page {page['page'] + 1}) — translate these blocks:\n{blocks}")
    return "\n\n".join(parts)


# plenty for a full page of dialogue; a draft that hits it is rambling
# (e.g. echoing the Japanese between translations) and gets retried
NUM_PREDICT = 2048


def _draft_page(system: str, page: dict, history, temperature: float) -> list[str] | None:
    n = len(page["blocks"])
    try:
        result = llm.chat_json(system, _page_prompt(page, history), _schema(n),
                               temperature=temperature, num_predict=NUM_PREDICT)
    except llm.BadResponse:
        return None  # a bad generation is just a failed draft, not a dead job
    translations = result.get("translations", [])
    return [str(t) for t in translations] if len(translations) == n else None


def _edit_page(editor_system: str, page: dict, history,
               drafts: list[list[str]]) -> list[str] | None:
    n = len(page["blocks"])
    parts = [_history_part(history)] if history else []
    blocks = []
    for i, b in enumerate(page["blocks"]):
        lines = [f"{i + 1}. {b['text']}"]
        lines += [f"   {chr(97 + d)}) {draft[i]}" for d, draft in enumerate(drafts)]
        blocks.append("\n".join(lines))
    parts.append(f"CURRENT PAGE (page {page['page'] + 1}) — finalize these blocks:\n"
                 + "\n".join(blocks))
    try:
        result = llm.chat_json(editor_system, "\n\n".join(parts), _schema(n),
                               temperature=0.3, num_predict=NUM_PREDICT)
    except llm.BadResponse:
        return None  # caller falls back to the first draft
    translations = result.get("translations", [])
    return [str(t) for t in translations] if len(translations) == n else None


def _translate_page(system: str, editor_system: str, page: dict, history, log) -> list[str]:
    drafts = []
    for temp in DRAFT_TEMPS[:max(1, TRANSLATE_DRAFTS)]:
        d = _draft_page(system, page, history, temp)
        if d:
            drafts.append(d)
    if not drafts:  # one retry at the safe temperature before giving up
        d = _draft_page(system, page, history, 0.3)
        if d is None:
            raise llm.OllamaError(f"page {page['page'] + 1}: no draft returned "
                                  f"the expected number of translations")
        drafts.append(d)
    if len(drafts) == 1:
        return drafts[0]
    final = _edit_page(editor_system, page, history, drafts)
    if final is None:
        log(f"page {page['page'] + 1}: editor pass failed, keeping first draft")
        return drafts[0]
    return final


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
    editor_system = EDITOR_SYSTEM.format(target_lang=target_lang, bible=bible)
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
        n_drafts = max(1, min(TRANSLATE_DRAFTS, len(DRAFT_TEMPS)))
        log(f"translating page {page['page'] + 1} ({i + 1}/{len(todo)})"
            + (f" — {n_drafts} drafts + editor" if n_drafts > 1 else ""))
        translations = _translate_page(system, editor_system, page, history, log)
        result[key] = translations
        history.append((page, translations))
        history[:] = history[-CONTEXT_PAGES:]
        if (i + 1) % 5 == 0 or i == len(todo) - 1:
            library.save_json(slug, out_name, result)
    library.save_json(slug, out_name, result)
    log(f"translation complete: {len(todo)} pages → {target_lang}")
