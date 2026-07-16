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

The flip side of a permissive detector is false positives on art
(screentone, polka dots, fabric), which manga-ocr then "reads" as
plausible hallucinated dialogue — so blocks must also survive: a size
gate on low-confidence and textless boxes, a dots-only text filter, a
pixel text-mask coverage check, and near-duplicate box removal.
"""

import re
import sys

from PIL import Image

from . import library

CONF_THRESH = 0.3   # detector default is 0.4; SFX lettering scores low
CROP_PAD = 8        # px of context around a block for fallback OCR
# manga-ocr is a captioner: fed a no-text crop it *hallucinates* plausible
# Japanese (そういうことですから…) rather than returning nothing. Several
# guards keep false-positive detector boxes from becoming ghost dialogue:
DETECTOR_CONF = 0.4      # the detector's own default; blocks under it
                         # exist only because of our lowered threshold
MAX_SUSPECT_FRAC = 0.03   # low-confidence or textless blocks must be
                          # SFX-sized (the point of conf 0.3 was SFX
                          # lettering; a huge weak box is art, and OCR'ing
                          # it yields hallucinated dialogue). Observed:
                          # real full-page scream 2.1%, art slivers ≥4.4%
MIN_MASK_COV = 0.04      # min share of the box covered by the detector's
                         # own pixel text-mask (real text ≥ ~7%, art ~2%)
# OCR of art yields dots and dashes; demand a letter, digit, or kana/kanji
_REAL_TEXT = re.compile(r"[0-9A-Za-z぀-ヿ㐀-鿿０-ｚｦ-ﾟ]")
# manga-ocr misreads the vertical ellipsis as a fullwidth colon (し… →
# し：). Colons in manga dialogue only really occur between digits
# (times, scores) — swap every other ： for the ellipsis it actually is
_FAKE_COLON = re.compile(r"(?<![0-9０-９])：|：(?![0-9０-９])")


class _CaptureMask:
    """Wraps comic-text-detector to keep its refined pixel text-mask,
    which mokuro computes and then discards — it is the only evidence of
    *where the model actually saw glyphs*, used to reject boxes over art."""

    def __init__(self, det):
        self._det = det
        self.last_mask = None

    def __call__(self, *args, **kwargs):
        out = self._det(*args, **kwargs)
        self.last_mask = out[1]
        return out

    def __getattr__(self, name):
        return getattr(self._det, name)

_engine = None


def _get_engine(log):
    global _engine
    if _engine is None:
        log("loading OCR models (first run downloads them)...")
        from mokuro.manga_page_ocr import MangaPageOcr
        _engine = MangaPageOcr()
        _engine.text_detector.conf_thresh = CONF_THRESH
        _engine.text_detector = _CaptureMask(_engine.text_detector)
        # the per-box YOLO confidence dies inside group_output; capture it
        # at the postprocess step so low-conf boxes can be size-gated
        import comic_text_detector.inference as ctd
        orig = ctd.postprocess_yolo
        def capture(det, conf_thresh, nms_thresh, resize_ratio, sort_func=None):
            out = orig(det, conf_thresh, nms_thresh, resize_ratio, sort_func)
            _last_yolo[:] = [(list(b), float(c)) for b, c in zip(out[0], out[2])]
            return out
        ctd.postprocess_yolo = capture
    return _engine


_last_yolo: list = []


def _yolo_conf(box) -> float:
    """Detection confidence of the YOLO box behind this block. Blocks
    born from scattered seg-head lines have no YOLO box — those already
    passed the line score filter, so treat them as confident."""
    return max((c for yb, c in _last_yolo if _iou(box, yb) > 0.3), default=1.0)


def run(slug: str, log=print) -> list[dict]:
    files = library.page_files(slug)
    if not files:
        raise RuntimeError("volume has no page images")
    mpo = _get_engine(log)

    pages, rescued, ghosts = [], 0, 0
    for i, path in enumerate(files):
        blocks = []
        width = height = None
        try:
            result = mpo(str(path))
            width, height = result["img_width"], result["img_height"]
            mask = mpo.text_detector.last_mask
            for b in result["blocks"]:
                box = [int(v) for v in b["box"]]
                text = "".join(b["lines"])
                x1, y1, x2, y2 = box
                big = (x2 - x1) * (y2 - y1) > MAX_SUSPECT_FRAC * width * height
                if big and _yolo_conf(box) < DETECTOR_CONF:
                    # exists only because of our lowered threshold, yet
                    # far bigger than any SFX lettering: art, and its
                    # line pass "reads" hallucinated dialogue off it
                    ghosts += 1
                    continue
                was_rescue = False
                if not text.strip():
                    # detected block, failed line pass: single-kanji SFX
                    # territory — read the whole crop directly, but only
                    # if it's SFX-sized; a big textless box is art
                    if big:
                        ghosts += 1
                        continue
                    text = _ocr_box(mpo, path, box, width, height)
                    was_rescue = True
                text = _FAKE_COLON.sub("…", text)
                if not _REAL_TEXT.search(text):
                    ghosts += not was_rescue
                    continue
                if _mask_cov(mask, box) < MIN_MASK_COV:
                    ghosts += 1
                    continue
                rescued += was_rescue
                blocks.append({
                    "box": box,
                    "vertical": bool(b.get("vertical", True)),
                    "font_size": _num(b.get("font_size")),
                    "text": text,
                })
        except Exception as e:
            log(f"page {i + 1}: OCR failed ({e}) — skipping")
        blocks = _dedupe(blocks)
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
    note += f" ({ghosts} art false-positives dropped)" if ghosts else ""
    log(f"OCR done: {len(pages)} pages, {n_blocks} text blocks{note}")
    return pages


DUP_IOU = 0.6


def _dedupe(blocks: list[dict]) -> list[dict]:
    """The detector sometimes emits two near-identical boxes for one text
    column (seen on real scans, IoU ≥ 0.85); both OCR to the same sentence
    and the reader would letter the line twice. Keep the one whose OCR
    read the most text."""
    kept = []
    for b in sorted(blocks, key=lambda b: -len(b["text"])):
        if all(_iou(b["box"], k["box"]) < DUP_IOU for k in kept):
            kept.append(b)
    return kept


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def _mask_cov(mask, box) -> float:
    """Share of the box the detector's pixel text-mask covers — near zero
    when the box outlines art rather than glyphs."""
    if mask is None:
        return 1.0
    x1, y1, x2, y2 = (max(0, v) for v in box)
    region = mask[y1:y2, x1:x2]
    return float((region > 127).mean()) if region.size else 0.0


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
