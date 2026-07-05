"""Background job runner: OCR → story bible → translation, with progress
that the server streams to the browser over SSE."""

import threading
import traceback

from . import bible, library, ocr, translate


class Job:
    def __init__(self, slug: str, target_lang: str):
        self.slug = slug
        self.target_lang = target_lang
        self.events: list[dict] = []
        self.done = False
        self.error: str | None = None
        self._cond = threading.Condition()

    def log(self, message: str) -> None:
        with self._cond:
            self.events.append({"message": message})
            self._cond.notify_all()

    def wait_events(self, start: int, timeout: float = 15.0) -> list[dict]:
        """Return events[start:], blocking until there is something new,
        the job finishes, or timeout (SSE keep-alive tick)."""
        with self._cond:
            if len(self.events) <= start and not self.done:
                self._cond.wait(timeout)
            return self.events[start:]

    def finish(self, error: str | None = None) -> None:
        with self._cond:
            self.error = error
            self.done = True
            self._cond.notify_all()


_jobs: dict[str, Job] = {}
_lock = threading.Lock()


def get(slug: str) -> Job | None:
    return _jobs.get(slug)


def start(slug: str, target_lang: str) -> Job:
    with _lock:
        existing = _jobs.get(slug)
        if existing and not existing.done:
            raise RuntimeError(f"a job is already running for '{slug}'")
        job = Job(slug, target_lang)
        _jobs[slug] = job
    threading.Thread(target=_run, args=(job,), daemon=True).start()
    return job


def _run(job: Job) -> None:
    try:
        if not library.load_json(job.slug, "ocr.json"):
            job.log("stage: OCR")
            ocr.run(job.slug, log=job.log)
        saved = library.load_json(job.slug, "bible.json")
        if not saved or saved.get("target_lang") != job.target_lang:
            job.log("stage: context pass (reading the whole work)")
            bible.build(job.slug, job.target_lang, log=job.log)
        job.log("stage: translation")
        translate.run(job.slug, job.target_lang, log=job.log)
        job.log("all done")
        job.finish()
    except Exception:
        err = traceback.format_exc(limit=3)
        job.log(f"FAILED: {err.strip().splitlines()[-1]}")
        job.finish(error=err)
