import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIBRARY_DIR = Path(os.environ.get("FUKIDASHI_LIBRARY", ROOT / "data" / "library"))
FRONTEND_DIR = ROOT / "frontend"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("FUKIDASHI_MODEL", "gemma3:27b")

# Context window per request. The user runs a 24GB card and wants it
# worked hard (~85%+ VRAM): big window = more KV cache + no truncated
# bible chunks. Lower via env if a smaller card needs headroom.
OLLAMA_NUM_CTX = int(os.environ.get("FUKIDASHI_NUM_CTX", 32768))

# Keep the model resident between pages/volumes instead of paying a
# ~1 min reload after every 5 idle minutes.
OLLAMA_KEEP_ALIVE = os.environ.get("FUKIDASHI_KEEP_ALIVE", "30m")

# Rolling context for the translation pass: how many previous pages of
# dialogue (original + translation) to include in each page's prompt.
CONTEXT_PAGES = 5

# Context pass: pages of raw text per bible-building chunk.
BIBLE_CHUNK_PAGES = 25
