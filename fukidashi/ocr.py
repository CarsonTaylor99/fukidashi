"""OCR a volume with mokuro (comic-text-detector + manga-ocr).

mokuro is run as a subprocess against the volume's pages/ directory and
produces pages.mokuro JSON next to it. We parse that into our own
ocr.json: one entry per page, blocks in reading order with pixel boxes.
"""

import json
import subprocess
import sys
from pathlib import Path

from . import library

MOKURO_BIN = Path(sys.executable).parent / "mokuro"


def run(slug: str, log=print) -> list[dict]:
    vol = library.volume_dir(slug)
    pages_dir = vol / "pages"
    log("running mokuro (first run downloads OCR models)...")
    proc = subprocess.run(
        [str(MOKURO_BIN), str(pages_dir), "--disable_confirmation", "--ignore_errors"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"mokuro failed:\n{proc.stderr[-2000:]}")

    mokuro_json = vol / "pages.mokuro"
    if not mokuro_json.exists():
        raise RuntimeError(f"mokuro produced no output at {mokuro_json}")
    data = json.loads(mokuro_json.read_text())

    pages = []
    for i, page in enumerate(data.get("pages", [])):
        blocks = []
        for b in page.get("blocks", []):
            text = "".join(b.get("lines", []))
            if not text.strip():
                continue
            blocks.append({
                "box": b["box"],  # [x1, y1, x2, y2] pixels
                "vertical": b.get("vertical", True),
                "font_size": b.get("font_size"),
                "text": text,
            })
        # mokuro emits blocks in detection order; sort into manga reading
        # order: right-to-left by column, top-to-bottom within a column.
        blocks.sort(key=lambda b: (-b["box"][2], b["box"][1]))
        pages.append({
            "page": i,
            "img_path": page.get("img_path", ""),
            "width": page.get("img_width"),
            "height": page.get("img_height"),
            "blocks": blocks,
        })

    # mokuro exits 0 even when every page failed (e.g. a missing
    # dependency) — an empty result is an error, not a success.
    if not pages:
        raise RuntimeError(
            "mokuro processed no pages — check its log:\n" + proc.stderr[-1000:]
        )
    library.save_json(slug, "ocr.json", pages)
    n_blocks = sum(len(p["blocks"]) for p in pages)
    log(f"OCR done: {len(pages)} pages, {n_blocks} text blocks")
    return pages
