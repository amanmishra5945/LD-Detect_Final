"""
config.py
Application configuration and environment settings.
Adheres to zero-leakage, ethical privacy standards, and configurable providers.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Database
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'ld_detection.db'}")
# Fix SQLAlchemy 1.4+ compatibility for Heroku/Render postgres:// prefix
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Server & Network Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
ALLOWED_ORIGINS = [
    orig.strip() for orig in os.getenv("ALLOWED_ORIGINS", "*").split(",") if orig.strip()
]

# Application Info
APP_TITLE = "Smart Learning Disability Screening System"
APP_VERSION = "2.0.0"
APP_DESCRIPTION = (
    "AI-powered multi-stage screening support for Dyslexia and Dysgraphia (Ages 5–12). "
    "Educational screening aid only — not a clinical diagnosis."
)

# Speech Configuration
# Options: "browser" (primary Web Speech API) | "whisper" (optional local/backend provider)
SPEECH_PROVIDER = os.getenv("SPEECH_PROVIDER", "browser")
STORE_RAW_AUDIO = os.getenv("STORE_RAW_AUDIO", "false").lower() in ("true", "1", "yes")

# Age Tiers & Difficulty Mapping
MIN_AGE = 5
MAX_AGE = 12

AGE_TIERS = {
    "beginner": {"min": 5, "max": 6, "label": "Beginner (Ages 5–6)"},
    "elementary": {"min": 7, "max": 8, "label": "Elementary (Ages 7–8)"},
    "intermediate": {"min": 9, "max": 10, "label": "Intermediate (Ages 9–10)"},
    "advanced": {"min": 11, "max": 12, "label": "Advanced (Ages 11–12)"},
}

def get_age_tier(age: int) -> str:
    age = int(age)
    if age <= 6:
        return "beginner"
    elif age <= 8:
        return "elementary"
    elif age <= 10:
        return "intermediate"
    else:
        return "advanced"

def get_tier_label(age: int) -> str:
    tier = get_age_tier(age)
    return AGE_TIERS[tier]["label"]
