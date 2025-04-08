"""Application settings and configuration."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
SAMPLES_DIR = DATA_DIR / "samples"

# Ensure directories exist
for directory in [DATA_DIR, CACHE_DIR, SAMPLES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Document Processing Settings
CHUNK_SIZE = 1000  # Maximum tokens per chunk
CHUNK_OVERLAP = 100  # Token overlap between chunks
MAX_SECTIONS_PER_CHUNK = 3  # Maximum number of sections to combine

# Vector Store Settings
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_CHUNK_SIZE = 1000
EMBEDDING_MAX_RETRIES = 1
EMBEDDING_TIMEOUT = 10

# QA Chain Settings
QA_MODEL = "gpt-3.5-turbo-16k"
QA_TEMPERATURE = 0.0
QA_MAX_TOKENS = 10000
QA_TOP_K = 6
QA_FETCH_K = 10
QA_SCORE_THRESHOLD = 0.5

# File Processing Settings
SUPPORTED_EXTENSIONS = ['.txt', '.pdf']
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Cache Settings
CACHE_ENABLED = True
CACHE_TTL = 3600  # 1 hour
MAX_CACHE_SIZE = 100 * 1024 * 1024  # 100MB
