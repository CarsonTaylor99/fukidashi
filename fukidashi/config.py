import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIBRARY_DIR = Path(os.environ.get("FUKIDASHI_LIBRARY", ROOT / "data" / "library"))
FRONTEND_DIR = ROOT / "frontend"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("FUKIDASHI_MODEL", "qwen2.5:14b")

# Rolling context for the translation pass: how many previous pages of
# dialogue (original + translation) to include in each page's prompt.
CONTEXT_PAGES = 3

# Context pass: pages of raw text per bible-building chunk.
BIBLE_CHUNK_PAGES = 25
