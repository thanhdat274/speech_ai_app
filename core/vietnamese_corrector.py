"""
vietnamese_corrector.py
=============================================================================
Vietnamese Text Correction Layer — v2

Applied after translation (or after Whisper for Vietnamese-source audio).

Two entry points:
    VietnameseCorrector.apply_corrections(text)
        → Full correction passes (punctuation, casing, whitespace)
        → Use for complete sentences from SentenceBuilder / NLLB output

    VietnameseCorrector.apply_quick(text)
        → Lightweight pass: whitespace + punctuation spacing only
        → Use for streaming partial transcriptions (low latency)

Correction order (apply_corrections):
    1. Whitespace normalisation
    2. Mid-sentence capitalisation fixes (NLLB often over-capitalises)
    3. Spacing around punctuation
    4. Smart punctuation inference (add . or ? at sentence end)
    5. Sentence-start capitalisation
    6. Common NLLB translation artifact removal
=============================================================================
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class VietnameseCorrector:
    """Stateless correction layer for Vietnamese text.

    All methods are class methods — no instance state needed.
    Thread-safe by design (pure function calls, no shared mutable state).

    Usage::

        # After NLLB translation to Vietnamese:
        output = VietnameseCorrector.apply_corrections(translated_text)

        # For partial streaming chunks (minimal processing):
        output = VietnameseCorrector.apply_quick(partial_text)
    """

    # ------------------------------------------------------------------
    # Vocabulary / rule tables
    # ------------------------------------------------------------------

    # Words Whisper / NLLB wrongly capitalise mid-sentence
    _LOWERCASE_OVERRIDES = {
        # Discourse particles / filler
        "Vâng", "Dạ", "Ừ", "Nhé", "Nha", "Nhỉ", "Hả",
        # Conjunctions  
        "Thì", "Mà", "Là", "Có", "Không", "Được", "Rồi",
        "Và", "Hay", "Hoặc", "Nhưng", "Vì", "Nên", "Để",
        "Với", "Từ", "Về", "Trong", "Ngoài", "Trên", "Dưới",
        # Pronouns (mid-sentence only)
        "Tôi", "Bạn", "Anh", "Chị", "Em", "Họ", "Chúng",
    }

    # Sentence-final words that indicate a question
    _QUESTION_WORDS_END = (
        "không", "chưa", "hả", "hả bạn", "sao", "thế",
        "nào", "gì", "nhỉ", "nhé", "nha", "hé",
        "phải không", "đúng không", "vậy không",
    )

    # Sentence-initial words that indicate a question
    _QUESTION_WORDS_START = (
        "tại sao", "làm sao", "vì sao", "bao giờ", "khi nào",
        "ai ", "cái gì", "điều gì", "việc gì", "cách nào",
        "như thế nào", "thế nào", "bao nhiêu", "mấy ",
    )

    # NLLB translation artifacts — substrings to strip from output
    # NLLB-600M occasionally outputs these for Vietnamese targets
    _NLLB_ARTIFACTS = [
        re.compile(r"^[Vv]ăn bản:\s*"),           # "Văn bản: ..."
        re.compile(r"^[Dd]ịch:\s*"),               # "Dịch: ..."
        re.compile(r"\[.*?\]$"),                   # "[some tag]" at end
    ]

    # Punctuation spacing: remove space BEFORE . , ! ? : ;
    _PUNCT_BEFORE = re.compile(r"\s+([,.!?:;])")
    # Multiple spaces → single space
    _MULTI_SPACE  = re.compile(r" {2,}")
    # Capitalise after . ! ? followed by space
    _SENTENCE_SEP = re.compile(r"([.!?])\s+([a-zA-Zàáâãèéêìíòóôõùúýăđơưạắặầẩấẫậ])")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def apply_corrections(cls, text: str) -> str:
        """Full correction pipeline for complete Vietnamese sentences.

        Suitable for:
            - NLLB translation output (Vietnamese target)
            - Whisper output for Vietnamese-source audio
            - Completed sentences from SentenceBuilder

        Args:
            text: Raw Vietnamese text string.

        Returns:
            Corrected, formatted Vietnamese string.
            Returns original text unchanged if empty.
        """
        if not text or not text.strip():
            return text

        # 0. Remove known NLLB artifacts
        text = cls._strip_nllb_artifacts(text)

        # 1. Normalise whitespace
        text = cls._MULTI_SPACE.sub(" ", text).strip()

        # 2. Fix mid-sentence capitalisation
        text = cls._fix_midsentence_caps(text)

        # 3. Spacing around punctuation
        text = cls._PUNCT_BEFORE.sub(r"\1", text)

        # 4. Smart end punctuation
        text = cls._add_end_punctuation(text)

        # 5. Capitalise sentence start
        if text:
            text = text[0].upper() + text[1:]

        # 6. Re-capitalise after in-sentence sentence boundaries
        text = cls._SENTENCE_SEP.sub(lambda m: m.group(1) + " " + m.group(2).upper(), text)

        return text

    @classmethod
    def apply_quick(cls, text: str) -> str:
        """Lightweight correction for streaming partial chunks.

        Only fixes: whitespace, spacing, sentence-start capitalisation.
        No punctuation inference, no capitalisation remap.
        Target latency: < 0.5 ms.

        Args:
            text: Partial transcript chunk.

        Returns:
            Lightly cleaned string.
        """
        if not text or not text.strip():
            return text

        text = cls._MULTI_SPACE.sub(" ", text).strip()
        text = cls._PUNCT_BEFORE.sub(r"\1", text)

        if text:
            text = text[0].upper() + text[1:]

        return text

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @classmethod
    def _strip_nllb_artifacts(cls, text: str) -> str:
        """Remove known NLLB output artifacts from the start/end of text."""
        for pattern in cls._NLLB_ARTIFACTS:
            text = pattern.sub("", text)
        return text.strip()

    @classmethod
    def _fix_midsentence_caps(cls, text: str) -> str:
        """Lowercase words that NLLB/Whisper incorrectly capitalise mid-sentence.

        Only applies when the preceding token does not end a sentence
        (i.e. is not followed by . ! ? ,).
        """
        words = text.split()
        if len(words) <= 1:
            return text

        for i in range(1, len(words)):
            word = words[i]
            # Strip any leading punctuation to get the bare word
            bare = word.lstrip("\"'(«")
            if bare in cls._LOWERCASE_OVERRIDES:
                prev_last = words[i - 1].rstrip()[-1] if words[i - 1] else ""
                if prev_last not in {".", "!", "?"}:
                    # Lowercase only the override part, preserve leading punct
                    prefix = word[: len(word) - len(bare)]
                    words[i] = prefix + bare[0].lower() + bare[1:]

        return " ".join(words)

    @classmethod
    def _add_end_punctuation(cls, text: str) -> str:
        """Add sentence-terminating punctuation if missing.

        Rules:
            - Already ends with . ! ? , → leave as-is
            - Ends with a question word  → append ?
            - Otherwise                  → append .
        """
        if not text:
            return text

        last_char = text.rstrip()[-1] if text.rstrip() else ""
        if last_char in {".", "!", "?", ",", ":", ";", "…"}:
            return text

        low = text.lower()

        # Check question endings (longest match wins)
        is_question = any(
            low.endswith(" " + q) or low == q
            for q in sorted(cls._QUESTION_WORDS_END, key=len, reverse=True)
        )

        # Check question beginnings
        if not is_question:
            is_question = any(low.startswith(q) for q in cls._QUESTION_WORDS_START)

        return text + ("?" if is_question else ".")
