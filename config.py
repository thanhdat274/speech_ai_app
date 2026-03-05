"""
config.py
=============================================================================
Central Configuration File
=============================================================================
Holds all shared constants, application states, and UI thresholds.
"""

import os
import sys
import logging
from pathlib import Path

# --- Environment Configuration ---
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# ---------------------------------
# Newer torchaudio removed 'set_audio_backend', but some older libraries still call it.
try:
    import torchaudio
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda x: None
except ImportError:
    pass
# ----------------------------------------------------------

# App Info
APP_NAME = "Speech AI"
APP_VERSION = "4.0.0-PROD"

# Paths
BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "speech_ai.log"
TEMP_DIR = Path(os.environ.get("TEMP", "/tmp"))

# Logging Config
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_LEVEL = logging.INFO

# Hardware Fallbacks Thresholds
RAM_CRITICAL_PERCENT = 85.0
VRAM_CRITICAL_PERCENT = 90.0

# Supported Extensions
SUPPORTED_AUDIO_EXT = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4", ".mkv")

# Language Mappings
SUPPORTED_LANGUAGES = {
    "Auto-Detect": None,
    "English": "en",
    "Vietnamese": "vi",
    "Spanish": "es",
    "French": "fr",
    "Japanese": "ja"
}

TRANSLATION_MAPPING = {
    "None (Off)": None,
    "English": "eng_Latn",
    "Vietnamese": "vie_Latn",
    "Spanish": "spa_Latn",
    "French": "fra_Latn",
    "Japanese": "jpn_Jpan",
    "Korean": "kor_Hang",
    "Chinese": "zho_Hans",
    "German": "deu_Latn"
}

# Setup Logger
def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    if not logger.handlers:
        fmt = logging.Formatter(LOG_FORMAT)
        
        # Console
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        
        # File
        try:
            fh = logging.FileHandler(str(LOG_FILE), encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except Exception:
            pass
            
    return logger
