"""
vietnamese_corrector.py
=============================================================================
Vietnamese Text Correction Layer

Applies heuristic, regex-based, and rule-based corrections to Whisper outputs.
Whisper often skips punctuation or miscapitalises Vietnamese names in 
streaming contexts. This module dynamically adds basic punctuation 
and fixes known casing anomalies.
=============================================================================
"""

import re
import logging

logger = logging.getLogger(__name__)

class VietnameseCorrector:
    """Thread-safe static class for formatting and correcting Vietnamese text."""

    # Common words that Whisper wrongly capitalizes mid-sentence
    _LOWERCASE_OVERRIDES = {
        "Vâng", "Dạ", "Ừ", "Thì", "Mà", "Là", "Có", "Không", "Được", "Rồi", "Nhé"
    }

    # Common sentences that should end with a question mark
    _QUESTION_WORDS_END = (
        "không", "chưa", "hả", "sao", "thế", "nào", "gì", "nhỉ", "nhé"
    )
    _QUESTION_WORDS_START = (
        "tại sao", "làm sao", "vì sao", "bao giờ", "khi nào", "kẻ nào", "ai"
    )

    @classmethod
    def apply_corrections(cls, text: str) -> str:
        """Applies grammatical and stylistic fixes to a string of Vietnamese text."""
        if not text:
            return text

        # 1. Clean up duplicated whitespaces
        text = re.sub(r'\s+', ' ', text).strip()

        # 2. Fix mid-sentence unnecessary capitalisation
        words = text.split()
        if len(words) > 1:
            for i in range(1, len(words)):
                # If it's not the first word and no punctuation precedes it,
                # forcefully lowercase common conjunctions/particles
                if words[i] in cls._LOWERCASE_OVERRIDES and not words[i-1][-1] in {'.', ',', '!', '?'}:
                    words[i] = words[i].lower()
            text = " ".join(words)

        # 3. Smart punctuation check
        # If it doesn't end with punctuation, check if it's a question
        if text[-1] not in {'.', '!', '?', ','}:
            low_text = text.lower()
            is_question = False

            # Check if it ends with a question particle
            for w in cls._QUESTION_WORDS_END:
                if low_text.endswith(" " + w):
                    is_question = True
                    break
            
            # Check if it starts with a question trigger
            if not is_question:
                for w in cls._QUESTION_WORDS_START:
                    if low_text.startswith(w):
                        is_question = True
                        break
            
            if is_question:
                text += "?"
            else:
                text += "."

        # 4. Capitalise first letter if it's a full sentence (ends with punctuation)
        if text:
            text = text[0].upper() + text[1:]

        # 5. Spacing fixes around punctuation
        text = re.sub(r'\s+([,.\?!])', r'\1', text)

        return text
