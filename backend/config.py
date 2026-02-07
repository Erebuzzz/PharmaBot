"""
PharmaBot configuration — loads from .env at project root.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


class Settings:
    # --- LLM ---
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "")

    # --- Embeddings / Vector DB ---
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    CHROMA_DB_PATH: str = os.getenv(
        "CHROMA_DB_PATH",
        str(Path(__file__).resolve().parent / "chroma_db"),
    )

    # --- Data ---
    DATA_DIR: str = str(_PROJECT_ROOT)
    MEDICINE_CSV: str = str(_PROJECT_ROOT / "medicine_dataset.csv")
    DISEASE_CSV: str = str(_PROJECT_ROOT / "Diseases_Symptoms.csv")

    # --- OpenFDA ---
    OPENFDA_BASE_URL: str = "https://api.fda.gov"
    OPENFDA_CACHE_TTL: int = int(os.getenv("OPENFDA_CACHE_TTL", "3600"))

    # --- RAG ---
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))

    @property
    def gemini_model(self) -> str:
        return self.LLM_MODEL or "gemini-2.0-flash"

    @property
    def openrouter_model(self) -> str:
        return self.LLM_MODEL or "deepseek/deepseek-r1-0528:free"


settings = Settings()
