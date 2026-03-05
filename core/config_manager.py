"""
config_manager.py
=============================================================================
Handles persistence of user settings (JSON based).
=============================================================================
"""

import json
import os
from pathlib import Path
from config import setup_logger

logger = setup_logger("ConfigManager")

class ConfigManager:
    def __init__(self, config_path: str = "settings.json"):
        self.config_path = Path(config_path)
        self.defaults = {
            "compute_mode": "Auto (Recommended)",
            "whisper_model": "base",
            "source_lang": "Auto-Detect",
            "target_lang": "None (Off)",
            "input_device_index": None,
            "theme": "Dark"
        }
        self.settings = self.defaults.copy()
        self.load()

    def load(self):
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    self.settings.update(saved)
                logger.info("Settings loaded from disc.")
            except Exception as e:
                logger.error(f"Error loading settings: {e}")
        else:
            self.save()

    def save(self):
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=4)
            logger.info("Settings saved to disc.")
        except Exception as e:
            logger.error(f"Error saving settings: {e}")

    def get(self, key, default=None):
        return self.settings.get(key, default if default is not None else self.defaults.get(key))

    def set(self, key, value):
        self.settings[key] = value
        self.save()
