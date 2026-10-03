"""
Translation Post-Processor - Fix common NLLB artifacts
=============================================================================
"""

import re
import unicodedata
from typing import Optional


class TranslationPostProcessor:
    """Post-process NLLB translation output to fix common artifacts."""

    EXTRA_SPACES = re.compile(r'\s+')
    PUNCT_SPACING = re.compile(r'\s([.,!?;:)]+)')
    LEADING_PUNCT = re.compile(r'^([.,!?;:]+)\s*')
    TRAILING_PUNCT = re.compile(r'\s*([.,!?;:]+)$')

    @classmethod
    def fix_nllb_artifacts(cls, text: str, tgt_lang: str = "en") -> str:
        """Apply all applicable post-processing fixes."""
        if not text:
            return text

        result = text

        # 1. Remove extra spaces
        result = cls.EXTRA_SPACES.sub(' ', result)

        # 2. Fix punctuation spacing
        result = cls.PUNCT_SPACING.sub(r'\1', result)
        result = cls.LEADING_PUNCT.sub(r'\1', result)
        result = cls.TRAILING_PUNCT.sub(r'\1', result)

        # 3. Language-specific fixes
        if cls._is_vietnamese(tgt_lang):
            result = cls._fix_vietnamese(result)
        elif cls._is_east_asian(tgt_lang):
            result = cls._fix_east_asian(result)

        return result.strip()

    @classmethod
    def _fix_vietnamese(cls, text: str) -> str:
        """Apply Vietnamese-specific corrections."""
        # Fix over-capitalization
        text = re.sub(r'\b([A-Z]{2,})\b',
                      lambda m: m.group(1).capitalize(), text)

        # Normalize Unicode
        text = unicodedata.normalize('NFC', text)

        return text

    @classmethod
    def _fix_east_asian(cls, text: str) -> str:
        """Apply East Asian language fixes."""
        # Add spacing between English and East Asian characters
        text = re.sub(r'([a-zA-Z0-9])([\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af])',
                      r'\1 \2', text)
        text = re.sub(r'([\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af])([a-zA-Z0-9])',
                      r'\1 \2', text)

        return text

    @classmethod
    def _is_vietnamese(cls, lang: str) -> bool:
        return lang.lower() in ('vie_latn', 'vi', 'vietnamese')

    @classmethod
    def _is_east_asian(cls, lang: str) -> bool:
        return lang.lower() in ('jpn_jpan', 'japanese', 'kor_hang', 'korean',
                                'zho_hans', 'chinese')
