import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIBRARY_DIR = Path(os.environ.get("FUKIDASHI_LIBRARY", ROOT / "data" / "library"))
FRONTEND_DIR = ROOT / "frontend"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
# Abliterated gemma3 (mlabonne's method, q4_k_m = same 17GB footprint as
# stock gemma3:27b): identical translation quality, no refusals/sanitizing
# on adult works. Stock gemma3:27b remains a fine fallback for SFW.
OLLAMA_MODEL = os.environ.get(
    "FUKIDASHI_MODEL", "aqualaguna/gemma-3-27b-it-abliterated-GGUF:q4_k_m")

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

# Drafts per page for the editor pass: N independent translations at
# spread temperatures, then an editor call picks/synthesizes the final
# line per block (choose-or-edit beats pure best-of-N selection).
# 1 = single-shot, no editor pass (old behaviour, ~4x faster).
TRANSLATE_DRAFTS = int(os.environ.get("FUKIDASHI_DRAFTS", 3))
DRAFT_TEMPS = (0.3, 0.7, 1.0)

# Context pass: pages of raw text per bible-building chunk.
BIBLE_CHUNK_PAGES = 25
