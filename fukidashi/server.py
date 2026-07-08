"""FastAPI app: library management, processing jobs with SSE progress,
and the reader API."""

import json
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import library, llm, pipeline
from .config import FRONTEND_DIR, LIBRARY_DIR, OLLAMA_MODEL

app = FastAPI(title="fukidashi")


@app.get("/api/status")
def status():
    return {
        "ollama": llm.available(),
        "model": OLLAMA_MODEL,
        "model_present": llm.model_present(),
    }


@app.get("/api/volumes")
def volumes():
    return library.list_volumes()


@app.post("/api/import")
async def import_volume(file: UploadFile, title: str = Form(...)):
    if not title.strip():
        raise HTTPException(400, "title is required")
    with tempfile.NamedTemporaryFile(suffix=".cbz", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        slug = library.import_cbz(tmp_path, title.strip())
    except (FileExistsError, ValueError) as e:
        raise HTTPException(400, str(e))
    finally:
        tmp_path.unlink(missing_ok=True)
    return {"slug": slug}


@app.delete("/api/volumes/{slug}")
def delete_volume(slug: str):
    library.delete_volume(slug)
    return {"ok": True}


@app.post("/api/volumes/{slug}/translate")
def start_translate(slug: str, language: str = Form("English"), force: bool = Form(False)):
    if not library.volume_dir(slug).exists():
        raise HTTPException(404, "no such volume")
    if not llm.available():
        raise HTTPException(503, "Ollama is not reachable")
    lang = language.strip() or "English"
    if force:
        # full redo: OCR/bubbles too, so detector improvements reach old
        # volumes. New OCR shifts block indices, which would misalign
        # every stored translation — so all languages go, not just this
        # one (translation otherwise resumes past finished pages).
        d = library.volume_dir(slug)
        for name in ("ocr.json", "bubbles.json", "bible.json"):
            (d / name).unlink(missing_ok=True)
        for p in d.glob("translations.*.json"):
            p.unlink()
        shutil.rmtree(d / "cleaned", ignore_errors=True)
    try:
        pipeline.start(slug, lang)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {"ok": True}


@app.get("/api/volumes/{slug}/progress")
def progress(slug: str):
    job = pipeline.get(slug)
    if job is None:
        raise HTTPException(404, "no job for this volume")

    def stream():
        sent = 0
        while True:
            events = job.wait_events(sent)
            for ev in events:
                yield f"data: {json.dumps(ev)}\n\n"
            sent += len(events)
            if job.done and sent >= len(job.events):
                yield f"data: {json.dumps({'done': True, 'error': job.error})}\n\n"
                return
            if not events:
                yield ": keep-alive\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/volumes/{slug}/pages/{n}")
def page_image(slug: str, n: int):
    files = library.page_files(slug)
    if not 0 <= n < len(files):
        raise HTTPException(404, "no such page")
    return FileResponse(files[n])


@app.get("/api/volumes/{slug}/cleaned/{n}")
def cleaned_page_image(slug: str, n: int):
    """Page with original bubble text painted out (falls back to the
    original image if the bubble pass didn't touch this page)."""
    files = library.page_files(slug)
    if not 0 <= n < len(files):
        raise HTTPException(404, "no such page")
    cleaned = library.volume_dir(slug) / "cleaned" / files[n].name
    return FileResponse(cleaned if cleaned.exists() else files[n])


@app.get("/api/volumes/{slug}/reader")
def reader_data(slug: str, language: str = "English"):
    """Everything the reader needs: per-page blocks with boxes, original
    text, and translations."""
    meta = library.load_json(slug, "meta.json")
    if meta is None:
        raise HTTPException(404, "no such volume")
    ocr = library.load_json(slug, "ocr.json") or []
    lang_file = f"translations.{language.lower().replace(' ', '-')}.json"
    translations = library.load_json(slug, lang_file) or {}
    bible = library.load_json(slug, "bible.json") or {}
    bubbles = {p["page"]: p["bubbles"] for p in library.load_json(slug, "bubbles.json") or []}
    pages = []
    by_page = {p["page"]: p for p in ocr}
    for n in range(meta["page_count"]):
        p = by_page.get(n)
        blocks = []
        if p:
            trs = translations.get(str(n), [])
            bubs = bubbles.get(n, [])
            tr = lambda i: trs[i] if i < len(trs) else None
            emitted = set()
            for i, b in enumerate(p["blocks"]):
                bub = bubs[i] if i < len(bubs) else None
                if bub and bub.get("group"):
                    # blocks sharing a bubble are one utterance split
                    # across JP columns: emit a single merged block over
                    # the whole bubble, translations joined in reading
                    # order, so the reader typesets one paragraph
                    if i in emitted:
                        continue
                    g = bub["group"]
                    emitted.update(g)
                    members = [p["blocks"][j] for j in g]
                    parts = [tr(j) for j in g]
                    blocks.append({
                        "box": [min(m["box"][0] for m in members),
                                min(m["box"][1] for m in members),
                                max(m["box"][2] for m in members),
                                max(m["box"][3] for m in members)],
                        "vertical": members[0]["vertical"],
                        "text": " ".join(m["text"] for m in members),
                        "translation": " ".join(t for t in parts if t) or None,
                        "bubble": {"rect": bub["rect"], "chords": bub["chords"]},
                    })
                    continue
                blocks.append({
                    "box": b["box"],
                    "vertical": b["vertical"],
                    "text": b["text"],
                    "translation": tr(i),
                    "bubble": bub,
                })
        pages.append({
            "width": p["width"] if p else None,
            "height": p["height"] if p else None,
            "blocks": blocks,
        })
    return {
        "title": meta["title"],
        "page_count": meta["page_count"],
        "language": language,
        "bible": bible.get("bible"),
        "pages": pages,
    }


LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
