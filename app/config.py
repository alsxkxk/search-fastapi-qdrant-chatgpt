import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths
ROOT = Path(__file__).parent
DATA = ROOT / "data"
MAX_SENTENCE_LENGTH = 100

# Qdrant

QDRANT_URL = os.getenv("QDRANT_URL")
# QDRANT_PORT = os.getenv("QDRANT_PORT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "my_collection"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
