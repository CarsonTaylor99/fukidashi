"""Context pass: read the whole volume's text and build a story bible.

This is what separates fukidashi from bubble-by-bubble translators. Before
any translation happens, the full OCR'd text is fed through the model in
page chunks. Each chunk updates a running "story bible": characters and
how their names should be rendered, relationships, speech styles,
honorifics policy, recurring terms/jokes, and overall tone. The bible is
kept as a plain text document the model rewrites each round — simple to
merge, easy to inspect, and it drops straight into translation prompts.
"""

from .config import BIBLE_CHUNK_PAGES
from . import library, llm

SYSTEM = """\
You are a professional manga translator preparing to translate a work into {target_lang}.
You are doing a first read-through of the raw Japanese text (OCR'd from pages, in reading order)
to build translation notes ("story bible") BEFORE translating.

You will receive your current story bible and the next batch of pages.
Rewrite the story bible, updated with anything new you learned. Keep it under 600 words.

The bible must be written in {target_lang} and cover, as far as the text so far reveals:
- CHARACTERS: names (with the romanization/rendering you will use), roles, relationships,
  distinctive speech style (formal/rough/archaic/cute...), first-person pronouns used.
- HONORIFICS POLICY: which honorifics to keep as-is vs adapt, per this work's tone.
- TERMS: recurring terms, place names, made-up words, running jokes — with the fixed
  translation you will use for each, so they stay consistent.
- TONE: genre, register, target reading level.
- OPEN QUESTIONS: ambiguities to resolve as you read further (e.g. unknown speaker genders).

Output ONLY the updated story bible text, no preamble.
"""


def pages_text(pages: list[dict]) -> list[str]:
    """Per-page dialogue dump used by both passes."""
    out = []
    for p in pages:
        lines = [f"[page {p['page'] + 1}]"]
        lines += [b["text"] for b in p["blocks"]]
        out.append("\n".join(lines))
    return out


def build(slug: str, target_lang: str, log=print) -> str:
    pages = library.load_json(slug, "ocr.json")
    if not pages:
        raise RuntimeError("no OCR data — run OCR first")
    texts = pages_text([p for p in pages if p["blocks"]])
    bible = "(empty — first read-through just beginning)"
    system = SYSTEM.format(target_lang=target_lang)
    chunks = [texts[i:i + BIBLE_CHUNK_PAGES] for i in range(0, len(texts), BIBLE_CHUNK_PAGES)]
    for n, chunk in enumerate(chunks, 1):
        log(f"context pass: reading chunk {n}/{len(chunks)}")
        user = (
            f"CURRENT STORY BIBLE:\n{bible}\n\n"
            f"NEXT PAGES:\n" + "\n\n".join(chunk)
        )
        bible = llm.chat(system, user, temperature=0.3).strip()
    library.save_json(slug, "bible.json", {"target_lang": target_lang, "bible": bible})
    log("story bible complete")
    return bible
