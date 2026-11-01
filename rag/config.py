import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # 没装 python-dotenv 也能跑
    load_dotenv = None

ROOT = Path(__file__).resolve().parent.parent
if load_dotenv is not None:
    load_dotenv(ROOT / ".env")

CORPUS_DIR = ROOT / "data" / "corpus"
INDEX_DIR = ROOT / "outputs" / "index"
INDEX_PATH = INDEX_DIR / "index.json"
EVAL_PATH = ROOT / "eval" / "questions.jsonl"

CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("RAG_TOP_K", "3"))
CANDIDATE_K = int(os.getenv("RAG_CANDIDATE_K", "20"))
MMR_LAMBDA = float(os.getenv("RAG_MMR_LAMBDA", "0.5"))
REFUSE_THRESHOLD = float(os.getenv("RAG_REFUSE_THRESHOLD", "0.04"))

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "default")
