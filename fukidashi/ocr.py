"""OCR a volume with mokuro's engine (comic-text-detector + manga-ocr).

mokuro is used as a library rather than a subprocess: the CLI hardwires
detector thresholds that drop exactly the blocks we care about. Two
knobs matter for SFX:

  * the YOLO block detector's confidence threshold (0.4 by default) —
    stylized action noises score low and vanish before OCR ever runs;
  * the line segmenter's score filter (hardcoded 0.6) — big single-kanji
    lettering often yields a detected *block* whose line pass finds
    nothing, so the block comes back with empty text.

We lower the first and, for blocks that still come back textless, OCR
the raw block crop directly with manga-ocr (which reads vertical and
horizontal text natively). Output is the same ocr.json as before: one
entry per page, blocks in reading order with pixel boxes.
"""

import re
import sys

from PIL import Image

from . import library

CONF_THRESH = 0.3   # detector default is 0.4; SFX lettering scores low
CROP_PAD = 8        # px of context around a block for fallback OCR
# fallback OCR on a false-positive block (art, screentone) yields dots
# and dashes; demand at least one letter, digit, or kana/kanji to keep it
_REAL_TEXT = re.compile(r"[0-9A-Za-z぀-ヿ㐀-鿿０-ｚｦ-ﾟ]")

_engine = None


def _get_engine(log):
    global _engine
    if _engine is None:
        log("loading OCR models (first run downloads them)...")
        from mokuro.manga_page_ocr import MangaPageOcr
        _engine = MangaPageOcr()
        _engine.text_detector.conf_thresh = CONF_THRESH
    return _engine


def run(slug: str, log=print) -> list[dict]:
    files = library.page_files(slug)
    if not files:
        raise RuntimeError("volume has no page images")
    mpo = _get_engine(log)

    pages, rescued = [], 0
    for i, path in enumerate(files):
        blocks = []
        width = height = None
        try:
            result = mpo(str(path))
            width, height = result["img_width"], result["img_height"]
            for b in result["blocks"]:
                box = [int(v) for v in b["box"]]
                text = "".join(b["lines"])
                if not text.strip():
                    # detected block, failed line pass: single-kanji SFX
                    # territory — read the whole crop directly
                    text = _ocr_box(mpo, path, box, width, height)
                    if not _REAL_TEXT.search(text):
                        continue
                    rescued += 1
                blocks.append({
                    "box": box,
                    "vertical": bool(b.get("vertical", True)),
                    "font_size": _num(b.get("font_size")),
                    "text": text,
                })
        except Exception as e:
            log(f"page {i + 1}: OCR failed ({e}) — skipping")
        # detection order → manga reading order: right-to-left by
        # column, top-to-bottom within a column.
        blocks.sort(key=lambda b: (-b["box"][2], b["box"][1]))
        pages.append({
            "page": i,
            "img_path": path.name,
            "width": width,
            "height": height,
            "blocks": blocks,
        })
        log(f"OCR page {i + 1}/{len(files)}: {len(blocks)} blocks")

    library.save_json(slug, "ocr.json", pages)
    n_blocks = sum(len(p["blocks"]) for p in pages)
    note = f" ({rescued} rescued by whole-block pass)" if rescued else ""
    log(f"OCR done: {len(pages)} pages, {n_blocks} text blocks{note}")
    return pages


def _ocr_box(mpo, path, box, width: int, height: int) -> str:
    from mokuro.utils import imread
    img = imread(str(path))
    if img is None:
        return ""
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1 - CROP_PAD), max(0, y1 - CROP_PAD)
    x2, y2 = min(width, x2 + CROP_PAD), min(height, y2 + CROP_PAD)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return ""
    # mokuro feeds manga-ocr BGR crops unconverted; match it
    return mpo.mocr(Image.fromarray(img[y1:y2, x1:x2]))


def _num(v):
    return None if v is None else float(v)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m fukidashi.ocr <slug>")
    run(sys.argv[1])
