# constants.py
import os
from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent.resolve()  # Go up two levels from src/constants.py
CONFIG_DIR = PROJECT_ROOT / "config"
SETTINGS_PATH = CONFIG_DIR / "settings.yaml"
CONFIG_PATH = CONFIG_DIR / "config.json"

# Default paths - now using absolute paths
DEFAULT_PDF_DIR = PROJECT_ROOT / "data/pdfs"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"

# Environment variables
HF_API_KEY = "HF_API_KEY"