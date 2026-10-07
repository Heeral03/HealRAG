import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CORPUS_DIR = DATA_DIR / "corpus"
DB_DIR = BASE_DIR / "db"

# Ensure directories exist
for directory in [DATA_DIR, CORPUS_DIR, DB_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Vector DB configuration (Qdrant)
QDRANT_DIR = DB_DIR / "qdrant"
QDRANT_COLLECTION_NAME = "healrag_chunks"
METADATA_PATH = DB_DIR / "metadata.json"

# Model configuration
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-small")
ENABLE_CROSS_ENCODER = os.environ.get("ENABLE_CROSS_ENCODER", "0" if os.environ.get("RENDER") or os.environ.get("DISABLE_BG_PRELOAD") else "1") == "1"
CHUNK_SIZE_WORDS = 300
CHUNK_OVERLAP_WORDS = 50

# LLM Generation configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

# CRAG Evaluator thresholds
EVALUATOR_UPPER_THRESHOLD = 0.60  # Confidence score >= 0.60 -> CORRECT
EVALUATOR_LOWER_THRESHOLD = 0.45  # Confidence score < 0.45 -> INCORRECT (0.45-0.60 -> AMBIGUOUS)

# Web Search backend (Tavily API key, optional — falls back to DuckDuckGo if absent)
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

