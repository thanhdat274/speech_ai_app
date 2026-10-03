"""
Hotword Manager - Dynamic Topic Detection & Management
=============================================================================
Manages hotwords (topic keywords) for Whisper transcription.

Features:
    - Auto-extraction of hotwords from recent transcript
    - Language-specific hotword dictionaries
    - Hotword decay over time
    - Weight adjustment based on frequency

Usage:
    manager = HotwordManager()
    manager.add_text("Today we'll discuss machine learning and AI")
    hotwords = manager.get_active_hotwords()
    manager.prune_inactive()
"""

import re
import time
from collections import Counter, defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple


# Pre-defined hotword dictionaries by topic and language
BASE_HOTWORDS = {
    "vi": {
        "technology": [
            "AI", "machine learning", "deep learning", "neural network",
            "Python", "programming", "software", "algorithm",
            "blockchain", "cryptocurrency", "cloud", "database",
            "mô hình", "huấn luyện", "dữ liệu", "hệ thống", "mạng",
            "trí tuệ nhân tạo", "tự động hóa", "xử lý ngôn ngữ",
        ],
        "education": [
            "học tập", "giảng dạy", "trường học", "bài học", "giáo dục",
            "sinh viên", "giáo viên", "university", "college",
            "học viện", "kiến thức", "đào tạo", "khóa học",
        ],
        "business": [
            "kinh doanh", "doanh nghiệp", "đầu tư", "thị trường",
            "lợi nhuận", "bán hàng", "marketing", "startup",
            "revenue", "sales", "investment", "cổ phiếu",
        ],
        "entertainment": [
            "phim", "âm nhạc", "ca sĩ", "ca khúc", "review",
            "game", "trò chơi", "streaming", "movie", "music",
        ],
    },
    "en": {
        "technology": [
            "AI", "artificial intelligence", "machine learning", "deep learning",
            "neural network", "Python", "programming", "software", "algorithm",
            "blockchain", "cryptocurrency", "cloud", "database", "API",
            "API", "REST", "GraphQL", "microservices", "container", "Docker",
        ],
        "education": [
            "learning", "education", "school", "university", "course",
            "student", "teacher", "knowledge", "training", "tutorial",
        ],
        "business": [
            "business", "company", "investment", "market", "profit",
            "sales", "marketing", "startup", "revenue", "growth",
        ],
        "entertainment": [
            "movie", "music", "singer", "song", "review", "game",
            "gaming", "streaming", "Netflix", "YouTube", "podcast",
        ],
    },
    "default": [
        "hello", "world", "thank", "thanks", "please", "you're",
    ],
}


class HotwordManager:
    """Manage dynamic hotwords for Whisper transcription."""

    def __init__(
        self,
        decay_rate: float = 0.1,
        min_frequency: int = 2,
        max_hotwords: int = 10,
        max_text_length: int = 250,
    ):
        """
        Args:
            decay_rate: Rate at which hotwords lose weight over time.
            min_frequency: Minimum occurrences to become a hotword.
            max_hotwords: Maximum number of active hotwords.
            max_text_length: Maximum text length for hotword extraction.
        """
        self.decay_rate = decay_rate
        self.min_frequency = min_frequency
        self.max_hotwords = max_hotwords
        self.max_text_length = max_text_length

        # Current active hotwords with weights
        self._hotwords: Dict[str, float] = {}

        # Recent texts for extraction (last 5 chunks)
        self._recent_texts: deque = deque(maxlen=5)

        # Last update timestamp
        self._last_update = time.time()

        # Language tracking
        self._detected_language: Optional[str] = "vi"  # Default to Vietnamese
        self._language_history: deque = deque(maxlen=10)

    def add_text(self, text: str, language: Optional[str] = None) -> None:
        """Add text for hotword extraction and update hotwords.

        Args:
            text: Text to analyze for hotword extraction.
            language: Detected language code.
        """
        # Update language detection
        if language:
            self._language_history.append(language)
            self._detected_language = self._get_primary_language()

        # Add to recent texts
        if text and len(text) <= self.max_text_length:
            self._recent_texts.append((text, time.time(), self._detected_language))

        # Extract new hotwords
        self._extract_hotwords()

        # Decay existing hotwords
        self._decay_hotwords()

    def _get_primary_language(self) -> str:
        """Get the primary language from history."""
        if not self._language_history:
            return "vi"

        lang_counts = Counter(self._language_history)
        primary, _ = lang_counts.most_common(1)[0]
        return primary

    def _extract_hotwords(self) -> None:
        """Extract hotwords from recent texts."""
        if not self._recent_texts:
            return

        # Collect all text
        all_text = " ".join(text for text, _, _ in self._recent_texts)
        text_lower = all_text.lower()

        # Get language-specific hotwords
        lang = self._detected_language or "vi"
        base_hotwords = BASE_HOTWORDS.get(lang, BASE_HOTWORDS.get("vi", {}))

        # Extract noun phrases (simplified: 2-3 word sequences with content words)
        extracted = self._extract_noun_phrases(text_lower)

        # Score extracted phrases
        phrase_scores = Counter(extracted)

        # Add to hotwords
        for phrase, count in phrase_scores.items():
            if count >= self.min_frequency:
                # Weight based on frequency and recency
                weight = count * 1.0  # Base weight from frequency
                if phrase not in self._hotwords:
                    self._hotwords[phrase] = weight
                else:
                    # Decay existing and add new weight
                    self._hotwords[phrase] = self._hotwords[phrase] * 0.7 + weight

        # Also add from base dictionary
        for topic, keywords in base_hotwords.items():
            for keyword in keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in text_lower:
                    if keyword_lower not in self._hotwords:
                        self._hotwords[keyword_lower] = 1.5
                    else:
                        self._hotwords[keyword_lower] += 0.5

        # Prune if too many
        if len(self._hotwords) > self.max_hotwords * 2:
            self._prune_hotwords()

    def _extract_noun_phrases(self, text: str) -> List[str]:
        """Extract potential noun phrases from text.

        Simplified approach: find sequences of content words.
        """
        # Tokenize
        tokens = re.findall(r'\b\w+\b', text)

        # Filter to content words (rough heuristic)
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once",
            "và", "nhưng", "với", "từ", "về", "trong", "ngoài", "trên",
            "dưới", "có", "không", "là", "thì", "mà", "đang", "đã", "sẽ",
        }

        content_words = [
            w for w in tokens
            if w not in stop_words and len(w) > 2
        ]

        # Extract 2-3 word sequences
        phrases = []
        for i in range(len(content_words) - 1):
            # Bigrams
            bigram = f"{content_words[i]} {content_words[i+1]}"
            if len(bigram) < 25:  # Limit phrase length
                phrases.append(bigram)

            # Trigrams
            if i < len(content_words) - 2:
                trigram = f"{content_words[i]} {content_words[i+1]} {content_words[i+2]}"
                if len(trigram) < 40:
                    phrases.append(trigram)

        return phrases

    def _decay_hotwords(self) -> None:
        """Decay hotword weights over time."""
        current_time = time.time()
        elapsed = current_time - self._last_update
        self._last_update = current_time

        # Decay all hotwords
        for phrase in list(self._hotwords.keys()):
            # Decay based on elapsed time
            self._hotwords[phrase] *= (1 - self.decay_rate * elapsed)

            # Remove if below threshold
            if self._hotwords[phrase] < 0.5:
                del self._hotwords[phrase]

    def _prune_hotwords(self) -> None:
        """Remove lowest-weighted hotwords."""
        # Sort by weight
        sorted_hotwords = sorted(
            self._hotwords.items(),
            key=lambda x: x[1],
        )

        # Keep top max_hotwords
        self._hotwords = dict(sorted_hotwords[-self.max_hotwords:])

    def get_active_hotwords(self, as_string: bool = False) -> List[str]:
        """Get current active hotwords.

        Args:
            as_string: If True, return as comma-separated string.

        Returns:
            List of hotword phrases, sorted by weight.
        """
        # Sort by weight descending
        sorted_hotwords = sorted(
            self._hotwords.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        result = [phrase for phrase, _ in sorted_hotwords]

        if as_string:
            return ", ".join(result)
        return result

    def prune_inactive(self, min_weight: float = 0.5) -> None:
        """Prune inactive hotwords.

        Args:
            min_weight: Minimum weight to keep a hotword.
        """
        for phrase in list(self._hotwords.keys()):
            if self._hotwords[phrase] < min_weight:
                del self._hotwords[phrase]

    def get_hotwords_string(self, max_length: int = 100) -> str:
        """Get hotwords as a string suitable for Whisper prompt.

        Args:
            max_length: Maximum string length.

        Returns:
            Comma-separated string of hotwords.
        """
        hotwords = self.get_active_hotwords(as_string=True)

        # Truncate if needed
        if len(hotwords) > max_length:
            hotwords = hotwords[:max_length-10] + "..."

        return hotwords

    def reset(self) -> None:
        """Reset all state."""
        self._hotwords.clear()
        self._recent_texts.clear()
        self._language_history.clear()
        self._last_update = time.time()
        self._detected_language = "vi"

    def clear(self) -> None:
        """Alias for reset()."""
        self.reset()


class PromptBuilder:
    """Build Whisper prompts with hotwords."""

    def __init__(self, hotword_manager: HotwordManager):
        """
        Args:
            hotword_manager: HotwordManager instance to use.
        """
        self.hotword_manager = hotword_manager

    def build_prompt(
        self,
        base_prompt: str = "",
        language: Optional[str] = None,
    ) -> str:
        """Build a prompt with hotwords.

        Args:
            base_prompt: Base prompt string.
            language: Language code.

        Returns:
            Complete prompt string.
        """
        parts = []

        # Base prompt
        if base_prompt:
            parts.append(base_prompt)

        # Add hotwords
        hotwords = self.hotword_manager.get_active_hotwords(as_string=True)
        if hotwords:
            parts.append(f"Từ khóa: {hotwords}")

        return " ".join(parts)
