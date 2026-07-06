"""Bubble detection pass: find the real speech bubble behind each OCR block.

mokuro's boxes are tight around the *text*, which says nothing about the
bubble it sits in — so the reader used to guess, and guessed badly. Here
we recover the actual bubble with plain image ops (no LLM): a speech
bubble is a connected region of near-white pixels enclosing the text, so
flood out from the edges of the text box across white pixels and that
region is the bubble interior. From it we derive:

  * a cleaned copy of the page (cleaned/<page filename>) with the bubble
    interior painted white — the original text is gone, no masking hacks;
  * per block, the largest centered rectangle inscribed in the bubble,
    plus (for blocks that own their bubble) horizontal cross-sections of
    the interior, so the reader can flow text into the bubble's actual
    shape rather than confining it to the rectangle.

Blocks whose flood fails validation (open bubbles, SFX drawn over art,
screentone backgrounds) get null and the frontend falls back to the old
mask-and-guess overlay for just those blocks.
"""

import sys
from collections import Counter, defaultdict

import cv2
import numpy as np

from . import library

WHITE_THRESH = 200         # gray level counted as bubble paper
MAX_AREA_FRAC = 0.35       # white region bigger than this share of the page = leak
MIN_TEXTBOX_OVERLAP = 0.7  # bubble bbox must cover this much of the text box
SEED_OFFSET = 6            # px outside the text box to probe for bubble interior
RECT_MARGIN = 3            # breathing room (px) between text rect and bubble edge
SHAPE_MARGIN = 5           # erosion (px) between flowed text and bubble edge
CHORD_ROWS = 18            # horizontal cross-sections sampled per bubble


def run(slug: str, log=print) -> list[dict]:
    pages = library.load_json(slug, "ocr.json")
    if not pages:
        raise RuntimeError("no OCR data — run OCR first")
    files = library.page_files(slug)
    out_dir = library.volume_dir(slug) / "cleaned"
    out_dir.mkdir(exist_ok=True)

    result, found, total = [], 0, 0
    for p in pages:
        bubbles = [None] * len(p["blocks"])
        result.append({"page": p["page"], "bubbles": bubbles})
        total += len(p["blocks"])
        if not p["blocks"] or p["page"] >= len(files):
            continue
        img = _imread(files[p["page"]])
        if img is None:
            continue
        if _detect_page(img, p["blocks"], bubbles):
            _imwrite(out_dir / files[p["page"]].name, img)
        found += sum(1 for b in bubbles if b)

    library.save_json(slug, "bubbles.json", result)
    log(f"bubble detection: {found}/{total} blocks matched to a bubble")
    return result


def _detect_page(img, blocks: list[dict], bubbles: list) -> bool:
    """Fill translated-over bubbles white in img (in place), write the
    inscribed text rect for each matched block into bubbles. Returns
    whether img was modified."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    white = (gray >= WHITE_THRESH).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=4)

    groups: dict[int, list[int]] = defaultdict(list)
    for i, b in enumerate(blocks):
        lab = _pick_label(labels, stats, b["box"], w, h)
        if lab:
            groups[lab].append(i)

    modified = False
    for lab, idxs in groups.items():
        comp = (labels == lab).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        # fill the whole interior (the text strokes are holes in the
        # white component) — this both cleans the page and gives the
        # solid region to inscribe the text rect into
        solid = np.zeros_like(comp)
        cv2.drawContours(solid, [contour], -1, 1, cv2.FILLED)
        rect = _inscribed_rect(solid)
        if rect is None:
            continue
        cv2.drawContours(img, [contour], -1, (255, 255, 255), cv2.FILLED)
        modified = True
        chords = _chords(solid)
        if len(idxs) == 1:
            bubbles[idxs[0]] = {"rect": list(rect), "chords": chords}
        else:
            _split_bubble(rect, chords, idxs, blocks, bubbles)
    return modified


def _chords(solid) -> list | None:
    """Horizontal cross-sections [y, x_left, x_right] of the bubble
    interior, top to bottom, eroded for breathing room. The frontend
    flows text into this shape (CSS shape-outside floats) so line breaks
    follow the bubble's curve. Narrow lead-in/out rows — bubble tails,
    pointy tips — are trimmed so text stays in the body."""
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * SHAPE_MARGIN + 1, 2 * SHAPE_MARGIN + 1))
    er = cv2.erode(solid, k)
    ys = np.nonzero(er.any(axis=1))[0]
    if len(ys) < 8:
        return None
    rows = []
    for y in np.linspace(ys[0], ys[-1], CHORD_ROWS):
        xs = np.nonzero(er[int(round(y))])[0]
        if len(xs):
            rows.append([int(round(y)), int(xs[0]), int(xs[-1])])
    widest = max((r[2] - r[1] for r in rows), default=0)
    while rows and rows[0][2] - rows[0][1] < 0.3 * widest:
        rows.pop(0)
    while rows and rows[-1][2] - rows[-1][1] < 0.3 * widest:
        rows.pop()
    return rows if len(rows) >= 3 else None


def _pick_label(labels, stats, box, w: int, h: int) -> int:
    """Probe white pixels just outside the text box and vote on which
    connected component is the bubble interior."""
    x1, y1, x2, y2 = (int(v) for v in box)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    o = SEED_OFFSET
    seeds = [(cx, cy), (cx, y1 - o), (cx, y2 + o), (x1 - o, cy), (x2 + o, cy),
             (x1 - o, y1 - o), (x2 + o, y1 - o), (x1 - o, y2 + o), (x2 + o, y2 + o)]
    votes = Counter()
    for px, py in seeds:
        if 0 <= px < w and 0 <= py < h and labels[py, px]:
            votes[labels[py, px]] += 1
    for lab, _ in votes.most_common():
        if _valid(stats[lab], box, w, h):
            return lab
    return 0


def _valid(st, box, w: int, h: int) -> bool:
    x, y, cw, ch, area = st
    # the page background touches the border; bubbles don't
    if x == 0 or y == 0 or x + cw == w or y + ch == h:
        return False
    if area > MAX_AREA_FRAC * w * h:
        return False
    bx1, by1, bx2, by2 = box
    ix = max(0, min(x + cw, bx2) - max(x, bx1))
    iy = max(0, min(y + ch, by2) - max(y, by1))
    tb = max(1, (bx2 - bx1) * (by2 - by1))
    return ix * iy >= MIN_TEXTBOX_OVERLAP * tb


def _inscribed_rect(solid) -> tuple | None:
    """Best rectangle centered on the bubble's centroid that fits
    entirely inside it. Sweeps aspect ratios (binary search on scale per
    aspect, integral image for the all-inside test) and scores candidates
    by area with a penalty on tall-narrow shapes — translations are
    horizontal text, so a wide band through a tall bubble beats a sliver
    that matches the bubble's own aspect."""
    m = cv2.moments(solid, binaryImage=True)
    if not m["m00"]:
        return None
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    ys, xs = np.nonzero(solid)
    bw, bh = int(xs.max() - xs.min()), int(ys.max() - ys.min())
    ii = cv2.integral(solid)
    ih, iw = solid.shape

    def fits(x1, y1, x2, y2):
        if x1 < 0 or y1 < 0 or x2 > iw or y2 > ih or x2 - x1 < 8 or y2 - y1 < 8:
            return False
        s = ii[y2, x2] - ii[y1, x2] - ii[y2, x1] + ii[y1, x1]
        return s == (x2 - x1) * (y2 - y1)

    best, best_score = None, 0.0
    for fx in (0.7, 1.0, 1.4, 2.0, 2.8):
        lo, hi, cand = 0.0, 1.0, None
        for _ in range(11):
            mid = (lo + hi) / 2
            rw, rh = bw * mid * fx / 2, bh * mid / 2
            r = (round(cx - rw), round(cy - rh), round(cx + rw), round(cy + rh))
            if fits(*r):
                lo, cand = mid, r
            else:
                hi = mid
        if cand:
            w, h = cand[2] - cand[0], cand[3] - cand[1]
            score = w * h * min(1.0, (w / h) / 1.2)
            if score > best_score:
                best, best_score = cand, score
    if best is None:
        return None
    x1, y1, x2, y2 = best
    mx = min(RECT_MARGIN, (x2 - x1) // 4)
    my = min(RECT_MARGIN, (y2 - y1) // 4)
    return (x1 + mx, y1 + my, x2 - mx, y2 - my)


def _split_bubble(rect, chords, idxs: list[int], blocks: list[dict], bubbles: list) -> None:
    """Several OCR blocks share one bubble (joined bubbles): each block
    becomes its own placement region, anchored at the original cluster's
    position — boundaries fall at the midpoints between neighbouring
    cluster centres along the axis the clusters spread over. The bubble's
    chords are clipped to each region so every cluster still wraps to the
    bubble's curve. Falls back to text-length slicing if the anchored
    boundaries collapse (clusters bunched at one end)."""
    centers = {i: ((blocks[i]["box"][0] + blocks[i]["box"][2]) / 2,
                   (blocks[i]["box"][1] + blocks[i]["box"][3]) / 2) for i in idxs}
    spread_x = max(c[0] for c in centers.values()) - min(c[0] for c in centers.values())
    spread_y = max(c[1] for c in centers.values()) - min(c[1] for c in centers.values())
    axis = 0 if spread_x >= spread_y else 1
    ordered = sorted(idxs, key=lambda i: centers[i][axis])
    lo, hi = rect[axis], rect[axis + 2]
    cuts = [round((centers[a][axis] + centers[b][axis]) / 2)
            for a, b in zip(ordered, ordered[1:])]
    bounds = [lo] + [min(hi, max(lo, c)) for c in cuts] + [hi]
    if any(b2 - b1 < 16 for b1, b2 in zip(bounds, bounds[1:])):
        weights = [max(1, len(blocks[i]["text"])) for i in ordered]
        total, pos, bounds = sum(weights), float(lo), [lo]
        for w in weights:
            pos += (hi - lo) * w / total
            bounds.append(round(pos))
    for k, i in enumerate(ordered):
        s1, s2 = bounds[k], bounds[k + 1] - 2
        seg = list(rect)
        seg[axis], seg[axis + 2] = s1, s2
        sub = None
        if chords:
            if axis == 0:
                sub = [[y, max(xl, s1), min(xr, s2)] for y, xl, xr in chords
                       if min(xr, s2) - max(xl, s1) >= 12]
            else:
                sub = [[y, xl, xr] for y, xl, xr in chords if s1 <= y <= s2]
            if len(sub) < 3:
                sub = None
        bubbles[i] = {"rect": seg, "chords": sub}


def _imread(path):
    # np.fromfile + imdecode instead of cv2.imread: survives non-ASCII paths
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _imwrite(path, img) -> None:
    ok, buf = cv2.imencode(path.suffix, img)
    if ok:
        buf.tofile(str(path))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m fukidashi.bubbles <slug>")
    run(sys.argv[1])
