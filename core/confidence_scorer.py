"""
Confidence Scorer - Score translation quality
=============================================================================
"""

import re
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class ConfidenceScore:
    """Confidence score for a translation."""
    overall: float
    translation_quality: float
    length_penalty: float
    punctuation_score: float

    def to_dict(self) -> Dict:
        return {
            "overall": round(self.overall, 2),
            "translation_quality": round(self.translation_quality, 2),
            "length_penalty": round(self.length_penalty, 2),
            "punctuation_score": round(self.punctuation_score, 2),
            "label": self.get_label()
        }

    def get_label(self) -> str:
        if self.overall >= 0.8:
            return "High"
        elif self.overall >= 0.6:
            return "Medium"
        else:
            return "Low"


class ConfidenceScorer:
    """Calculate confidence scores for translations."""

    MIN_WORDS = 2
    MAX_WORDS = 100

    def score_translation(self, src_text: str, tgt_text: str,
                         model_logits: Optional[Dict] = None) -> ConfidenceScore:
        """Calculate confidence score for a translation."""

        translation_quality = self._score_translation_quality(model_logits)
        length_penalty = self._score_length(tgt_text)
        punct_score = self._score_punctuation(tgt_text)
        lang_consistency = self._score_language_consistency(src_text, tgt_text)

        overall = (
            translation_quality * 0.5 +
            length_penalty * 0.2 +
            punct_score * 0.2 +
            lang_consistency * 0.1
        )

        return ConfidenceScore(
            overall=overall,
            translation_quality=translation_quality,
            length_penalty=length_penalty,
            punctuation_score=punct_score,
        )

    def _score_translation_quality(self, model_logits: Optional[Dict]) -> float:
        """Score from model's internal confidence."""
        if model_logits is None:
            return 0.7

        if 'avg_logprob' in model_logits:
            logprob = model_logits['avg_logprob']
            return min(1.0, max(0.0, (logprob + 1) / 2 + 0.5))

        return 0.7

    def _score_length(self, text: str) -> float:
        """Penalize very short or unusually long translations."""
        words = len(text.split())

        if words < self.MIN_WORDS:
            return 0.3
        elif words > self.MAX_WORDS:
            return 0.5
        else:
            return 1.0

    def _score_punctuation(self, text: str) -> float:
        """Check if translation has proper punctuation."""
        if not text:
            return 0.0

        has_end_punct = any(text.endswith(p)
                           for p in ['.', '!', '?', '.', '…'])
        parens_balanced = text.count('(') == text.count(')')
        quotes_balanced = text.count('"') % 2 == 0

        if has_end_punct and parens_balanced and quotes_balanced:
            return 1.0
        elif has_end_punct:
            return 0.7
        else:
            return 0.5

    def _score_language_consistency(self, src: str, tgt: str) -> float:
        """Check if translation preserved important source elements."""
        src_proper_nouns = re.findall(r'\b[A-Z]{2,}\b', src)

        if not src_proper_nouns:
            return 1.0

        tgt_lower = tgt.lower()
        preserved = sum(1 for noun in src_proper_nouns
                      if noun.lower() in tgt_lower)

        return preserved / len(src_proper_nouns)
