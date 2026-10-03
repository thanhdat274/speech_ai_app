"""
Transcription Confidence Tracker
=============================================================================
Tracks and scores transcription confidence across chunks and segments.

Provides:
    - Per-chunk confidence scoring based on Whisper metrics
    - Confidence trend tracking
    - Low-confidence chunk flagging
    - Confidence-based re-transcription suggestions

Usage:
    tracker = TranscriptionConfidence()
    score = tracker.score_chunk(segment)
    trend = tracker.track_confidence_trend(chunks)
    flagged = tracker.flag_low_confidence(chunks)
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ConfidenceScore:
    """Confidence score for a transcription chunk."""
    overall: float
    language_confidence: float
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float
    repeat_penalty: float

    def to_dict(self) -> Dict:
        return {
            "overall": round(self.overall, 3),
            "language_confidence": round(self.language_confidence, 3),
            "avg_logprob": round(self.avg_logprob, 3),
            "no_speech_prob": round(self.no_speech_prob, 3),
            "compression_ratio": round(self.compression_ratio, 3),
            "repeat_penalty": round(self.repeat_penalty, 3),
            "label": self.get_label(),
        }

    def get_label(self) -> str:
        if self.overall >= 0.8:
            return "high"
        elif self.overall >= 0.6:
            return "medium"
        else:
            return "low"


class TranscriptionConfidence:
    """Track and score transcription confidence across chunks."""

    def __init__(
        self,
        low_confidence_threshold: float = 0.5,
        track_window: int = 20,
    ):
        """
        Args:
            low_confidence_threshold: Threshold below which chunks are flagged.
            track_window: Number of recent chunks to track for trend analysis.
        """
        self.low_confidence_threshold = low_confidence_threshold
        self.track_window = track_window

        # Recent scores for trend tracking
        self._recent_scores: deque = deque(maxlen=track_window)

    def score_chunk(
        self,
        segments: List,
        detected_language: Optional[str] = None,
        language_confidence: Optional[float] = None,
    ) -> ConfidenceScore:
        """Calculate confidence score for a transcription chunk.

        Args:
            segments: List of Whisper segments from transcription.
            detected_language: Detected language code (e.g., "vi", "en").
            language_confidence: Whisper's language detection confidence (0-1).

        Returns:
            ConfidenceScore dataclass with detailed breakdown.
        """
        if not segments:
            return ConfidenceScore(
                overall=0.0,
                language_confidence=language_confidence or 0.0,
                avg_logprob=0.0,
                no_speech_prob=0.0,
                compression_ratio=0.0,
                repeat_penalty=0.0,
            )

        # Aggregate segment metrics
        logprobs = []
        no_speech_probs = []
        compression_ratios = []
        repeat_penalties = []

        for seg in segments:
            # Log probability
            if hasattr(seg, "avg_logprob"):
                logprobs.append(seg.avg_logprob)

            # No speech probability
            if hasattr(seg, "no_speech_prob"):
                no_speech_probs.append(seg.no_speech_prob)

            # Compression ratio (indicates repetition/hallucination)
            if hasattr(seg, "compression_ratio"):
                compression_ratios.append(seg.compression_ratio)

            # Check for repeated words
            if hasattr(seg, "text"):
                repeat_penalty = self._calculate_repeat_penalty(seg.text)
                repeat_penalties.append(repeat_penalty)

        # Calculate aggregate scores
        avg_logprob = sum(logprobs) / len(logprobs) if logprobs else -1.0
        avg_no_speech = max(no_speech_probs) if no_speech_probs else 0.0
        avg_compression = max(compression_ratios) if compression_ratios else 1.0
        avg_repeat = min(repeat_penalties) if repeat_penalties else 1.0

        # Language confidence (default to 0.7 if not provided)
        lang_conf = language_confidence if language_confidence is not None else 0.7

        # Calculate overall score
        overall = self._calculate_overall_score(
            lang_conf=lang_conf,
            avg_logprob=avg_logprob,
            no_speech_prob=avg_no_speech,
            compression_ratio=avg_compression,
            repeat_penalty=avg_repeat,
        )

        score = ConfidenceScore(
            overall=overall,
            language_confidence=lang_conf,
            avg_logprob=avg_logprob,
            no_speech_prob=avg_no_speech,
            compression_ratio=avg_compression,
            repeat_penalty=avg_repeat,
        )

        # Track for trend analysis
        self._recent_scores.append(score)

        return score

    def track_confidence_trend(self, chunks: List[Dict]) -> Dict:
        """Analyze confidence trend across multiple chunks.

        Args:
            chunks: List of chunk dicts with 'confidence_score' or segment data.

        Returns:
            Dict with trend analysis:
                - direction: "improving", "declining", or "stable"
                - average_confidence: Mean confidence across chunks
                - variance: Confidence variance
                - low_confidence_count: Number of low-confidence chunks
                - recommendation: Suggested action (e.g., "retranscribe")
        """
        if not chunks:
            return {
                "direction": "unknown",
                "average_confidence": 0.0,
                "variance": 0.0,
                "low_confidence_count": 0,
                "recommendation": "none",
            }

        # Extract confidence scores
        scores = []
        low_count = 0

        for chunk in chunks:
            if "confidence_score" in chunk:
                conf = chunk["confidence_score"].overall
            elif "segments" in chunk:
                score = self.score_chunk(chunk["segments"])
                conf = score.overall
            else:
                continue

            scores.append(conf)
            if conf < self.low_confidence_threshold:
                low_count += 1

        if not scores:
            return {
                "direction": "unknown",
                "average_confidence": 0.0,
                "variance": 0.0,
                "low_confidence_count": 0,
                "recommendation": "none",
            }

        # Calculate statistics
        avg_confidence = sum(scores) / len(scores)
        variance = sum((s - avg_confidence) ** 2 for s in scores) / len(scores)

        # Determine trend direction
        direction = self._calculate_trend_direction(scores)

        # Generate recommendation
        recommendation = self._generate_recommendation(
            avg_confidence=avg_confidence,
            direction=direction,
            low_count=low_count,
            total=len(chunks),
        )

        return {
            "direction": direction,
            "average_confidence": round(avg_confidence, 3),
            "variance": round(variance, 3),
            "low_confidence_count": low_count,
            "recommendation": recommendation,
        }

    def flag_low_confidence(self, chunks: List[Dict]) -> List[Dict]:
        """Flag chunks with low confidence.

        Args:
            chunks: List of chunk dicts with segment data.

        Returns:
            List of chunks with confidence < threshold, with reasons.
        """
        flagged = []

        for i, chunk in enumerate(chunks):
            if "confidence_score" in chunk:
                score = chunk["confidence_score"]
            elif "segments" in chunk:
                score = self.score_chunk(chunk["segments"])
            else:
                continue

            if score.overall < self.low_confidence_threshold:
                reasons = self._identify_low_confidence_reasons(score)
                flagged.append({
                    "index": i,
                    "text": chunk.get("text", ""),
                    "confidence_score": score.overall,
                    "reasons": reasons,
                    "should_rerun": "logprob" in reasons or "compression" in reasons,
                })

        return flagged

    def should_retranscribe(self, chunk: Dict) -> bool:
        """Determine if a chunk should be re-transcribed with more context.

        Args:
            chunk: Chunk dict with segment data.

        Returns:
            True if re-transcription is recommended.
        """
        if "confidence_score" in chunk:
            score = chunk["confidence_score"]
        elif "segments" in chunk:
            score = self.score_chunk(chunk["segments"])
        else:
            return False

        # Retranscribe if:
        # 1. Overall confidence is low
        # 2. Log probability is very low (model uncertain)
        # 3. Compression ratio is high (likely repetition/hallucination)
        # 4. No-speech probability is high (may be noise)
        should_rerun = (
            score.overall < self.low_confidence_threshold or
            score.avg_logprob < -1.5 or
            score.compression_ratio > 2.4 or
            score.no_speech_prob > 0.5
        )

        return should_rerun

    def get_recent_trend(self) -> str:
        """Get the recent confidence trend from tracked scores.

        Returns:
            "improving", "declining", or "stable".
        """
        if len(self._recent_scores) < 3:
            return "insufficient_data"

        recent = list(self._recent_scores)[-5:]
        first_half = sum(s.overall for s in recent[:len(recent)//2]) / (len(recent)//2)
        second_half = sum(s.overall for s in recent[len(recent)//2:]) / (len(recent) - len(recent)//2)

        diff = second_half - first_half

        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "declining"
        else:
            return "stable"

    def _calculate_overall_score(
        self,
        lang_conf: float,
        avg_logprob: float,
        no_speech_prob: float,
        compression_ratio: float,
        repeat_penalty: float,
    ) -> float:
        """Calculate overall confidence score from individual metrics.

        Weighted combination:
            - Language confidence: 30%
            - Log probability: 25%
            - No speech probability: 20%
            - Compression ratio: 15%
            - Repeat penalty: 10%
        """
        # Normalize each metric to 0-1 scale

        # Language confidence (already 0-1)
        lang_score = lang_conf * 0.30

        # Log probability (typically -1 to -0.5 for good transcriptions)
        # Map: -2 -> 0, -0.5 -> 1
        logprob_score = max(0, min(1, (avg_logprob + 2) / 1.5)) * 0.25

        # No speech probability (lower is better)
        # Map: 1 -> 0, 0 -> 1
        no_speech_score = (1 - no_speech_prob) * 0.20

        # Compression ratio (lower is better, typically 1-2)
        # Map: >3 -> 0, 1 -> 1
        compression_score = max(0, 1 - (compression_ratio - 1) / 2) * 0.15

        # Repeat penalty (higher is better, typically 0.8-1.0)
        repeat_score = repeat_penalty * 0.10

        return lang_score + logprob_score + no_speech_score + compression_score + repeat_score

    def _calculate_repeat_penalty(self, text: str) -> float:
        """Calculate repeat penalty based on word repetition."""
        if not text:
            return 1.0

        words = text.lower().split()
        if len(words) < 3:
            return 1.0

        # Count unique words
        unique_words = set(words)
        unique_ratio = len(unique_words) / len(words)

        # Check for repeated phrases
        import re
        # Check for repeated 2-3 word sequences
        bigrams = [text[i:i+10] for i in range(0, len(text)-10, 5)]
        repeated_ngrams = len(bigrams) - len(set(bigrams))

        # Combine metrics
        repeat_penalty = unique_ratio * 0.7 + max(0, 1 - repeated_ngrams / 3) * 0.3

        return min(repeat_penalty, 1.0)

    def _calculate_trend_direction(self, scores: List[float]) -> str:
        """Calculate trend direction from list of scores."""
        if len(scores) < 3:
            return "stable"

        # Compare first half vs second half
        mid = len(scores) // 2
        first_half = sum(scores[:mid]) / mid
        second_half = sum(scores[mid:]) / (len(scores) - mid)

        diff = second_half - first_half

        if diff > 0.03:
            return "improving"
        elif diff < -0.03:
            return "declining"
        return "stable"

    def _generate_recommendation(
        self,
        avg_confidence: float,
        direction: str,
        low_count: int,
        total: int,
    ) -> str:
        """Generate action recommendation based on confidence analysis."""
        low_ratio = low_count / total if total > 0 else 0

        if avg_confidence < 0.4 and low_ratio > 0.5:
            return "retranscribe_all"
        elif avg_confidence < 0.5 or low_ratio > 0.3:
            return "review_low_confidence"
        elif direction == "declining":
            return "check_context"
        elif avg_confidence > 0.7:
            return "none"
        else:
            return "none"

    def _identify_low_confidence_reasons(self, score: ConfidenceScore) -> List[str]:
        """Identify specific reasons for low confidence."""
        reasons = []

        if score.language_confidence < 0.5:
            reasons.append("low_language_confidence")

        if score.avg_logprob < -1.5:
            reasons.append("low_logprob")

        if score.no_speech_prob > 0.5:
            reasons.append("high_no_speech_prob")

        if score.compression_ratio > 2.4:
            reasons.append("high_compression_ratio")

        if score.repeat_penalty < 0.7:
            reasons.append("high_repetition")

        return reasons
