"""Volume library: import, layout, and metadata.

Each volume lives at LIBRARY_DIR/<slug>/ :
    pages/          page images, sorted by filename
    ocr.json        parsed OCR blocks (written by ocr.py)
    bible.json      story bible (written by bible.py)
    translations.<lang>.json   per-page block translations (translate.py)
    meta.json       {"title": ..., "page_count": ...}
"""

import json
import re
import shutil
import unicodedata
import zipfile
from pathlib import Path

from .config import LIBRARY_DIR

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def slugify(title: str) -> str:
    slug = unicodedata.normalize("NFKC", title)
    slug = re.sub(r"[^\w\-]+", "-", slug, flags=re.UNICODE).strip("-").lower()
    return slug or "untitled"


def volume_dir(slug: str) -> Path:
    return LIBRARY_DIR / slug


def _sorted_images(d: Path) -> list[Path]:
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def import_cbz(cbz_path: Path, title: str) -> str:
    """Extract a CBZ/ZIP of page images into the library. Returns slug."""
    slug = slugify(title)
    dest = volume_dir(slug)
    if dest.exists():
        raise FileExistsError(f"volume '{slug}' already exists")
    pages = dest / "pages"
    pages.mkdir(parents=True)
    try:
        with zipfile.ZipFile(cbz_path) as zf:
            names = sorted(
                n for n in zf.namelist()
                if Path(n).suffix.lower() in IMAGE_EXTS and not n.startswith("__MACOSX")
            )
            if not names:
                raise ValueError("archive contains no page images")
            width = len(str(len(names)))
            for i, name in enumerate(names, 1):
                ext = Path(name).suffix.lower()
                with zf.open(name) as src, open(pages / f"{i:0{width}d}{ext}", "wb") as dst:
                    shutil.copyfileobj(src, dst)
    except Exception:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    _write_meta(slug, title)
    return slug


def import_folder(src: Path, title: str) -> str:
    """Copy a folder of page images into the library. Returns slug."""
    images = _sorted_images(src)
    if not images:
        raise ValueError(f"no page images found in {src}")
    slug = slugify(title)
    dest = volume_dir(slug)
    if dest.exists():
        raise FileExistsError(f"volume '{slug}' already exists")
    pages = dest / "pages"
    pages.mkdir(parents=True)
    width = len(str(len(images)))
    for i, img in enumerate(images, 1):
        shutil.copy2(img, pages / f"{i:0{width}d}{img.suffix.lower()}")
    _write_meta(slug, title)
    return slug


def _write_meta(slug: str, title: str) -> None:
    d = volume_dir(slug)
    meta = {"title": title, "page_count": len(_sorted_images(d / "pages"))}
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))


def page_files(slug: str) -> list[Path]:
    return _sorted_images(volume_dir(slug) / "pages")


def load_json(slug: str, name: str):
    p = volume_dir(slug) / name
    return json.loads(p.read_text()) if p.exists() else None


def save_json(slug: str, name: str, data) -> None:
    p = volume_dir(slug) / name
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.replace(p)


def list_volumes() -> list[dict]:
    if not LIBRARY_DIR.exists():
        return []
    out = []
    for d in sorted(LIBRARY_DIR.iterdir()):
        meta_path = d / "meta.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text())
        langs = sorted(
            p.name.removeprefix("translations.").removesuffix(".json")
            for p in d.glob("translations.*.json")
        )
        out.append({
            "slug": d.name,
            "title": meta.get("title", d.name),
            "page_count": meta.get("page_count", 0),
            "ocr_done": (d / "ocr.json").exists(),
            "languages": langs,
        })
    return out


def delete_volume(slug: str) -> None:
    d = volume_dir(slug)
    if d.exists():
        shutil.rmtree(d)
