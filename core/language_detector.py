"""
Language Detector - Multi-language Detection & Tracking
=============================================================================
Provides language detection, mixed-language detection, and cross-chunk
language consistency tracking for Whisper transcription.

Usage:
    detector = LanguageDetector()
    lang, confidence = detector.detect_language("xin chao ban")
    mixed = detector.detect_mixed_language("hello xin chao world")
    consistency = detector.track_language_consistency(chunks)
"""

import re
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

# Language-specific word patterns and indicators
LANGUAGE_INDICATORS = {
    "vi": {
        "words": {
            "và", "nhưng", "với", "từ", "về", "trong", "ngoài", "trên", "dưới",
            "có", "không", "là", "thì", "mà", "đang", "đã", "sẽ", "đến", "hơn",
            "một", "cái", "này", "kia", "này", "đó", "nơi", "bao giờ", "khi nào",
            "làm sao", "tại sao", "tại sao", "vậy", "rất", "quá", "hãy", "đừng",
            "vâng", "dạ", "ơi", "nhé", "nha", "nhỉ", "hả", "á", "à", "ư",
            "tôi", "tớ", "mình", "bạn", "anh", "chị", "em", "con", "ông", "bà",
            "cha", "mẹ", "thầy", "cô", "chú", "cô", "dì", "cậu", "mình", "chúng",
            "chúng ta", "chúng tôi", "chúng mình", "của", "cho", "theo", "sau",
            "trước", "bên", "ngoài", "giữa", "trong", "trên", "dưới", "vào", "ra",
        },
        "particles": ["hả", "nhé", "nha", "nhỉ", "ừ", "vâng", "dạ", "á", "ơ", "ư"],
        "question_words": [
            "tại sao", "làm sao", "vì sao", "bao giờ", "khi nào", "ai", "cái gì",
            "điều gì", "việc gì", "cách nào", "như thế nào", "thế nào", "bao nhiêu",
            "mấy", "ở đâu", "tên gì", "là gì",
        ],
        "stopwords": [
            "và", "nhưng", "với", "từ", "về", "trong", "ngoài", "trên", "dưới",
            "có", "không", "là", "thì", "mà", "đang", "đã", "sẽ", "đến", "hơn",
            "một", "cái", "này", "kia", "đó", "nơi", "rất", "quá", "hãy", "đừng",
            "của", "cho", "theo", "sau", "trước", "bên", "giữa", "vào", "ra",
        ],
    },
    "en": {
        "words": {
            "the", "is", "in", "to", "of", "and", "a", "that", "it", "for",
            "on", "with", "as", "this", "was", "are", "be", "have", "from",
            "an", "or", "i", "you", "he", "she", "we", "they", "but", "not",
            "at", "by", "my", "his", "her", "our", "their", "do", "so", "can",
            "all", "which", "about", "what", "when", "where", "who", "how",
            "because", "then", "there", "their", "into", "than", "over", "been",
            "would", "could", "should", "will", "must", "may", "might", "if",
        },
        "particles": ["oh", "ah", "uh", "um", "well", "yes", "no", "okay", "sure"],
        "question_words": ["what", "why", "when", "where", "who", "how", "which",
                          "whose", "whom", "why", "whether", "if"],
        "stopwords": [
            "the", "is", "in", "to", "of", "and", "a", "that", "it", "for",
            "on", "with", "as", "this", "was", "are", "be", "have", "from",
            "an", "or", "i", "you", "he", "she", "we", "they", "but", "not",
            "at", "by", "my", "his", "her", "our", "their", "do", "so", "can",
            "all", "which", "about", "what", "when", "where", "who", "how",
            "because", "then", "there", "their", "into", "than", "over", "been",
            "would", "could", "should", "will", "must", "may", "might", "if",
        ],
    },
    "ja": {
        "words": {
            "は", "が", "を", "に", "で", "へ", "と", "も", "の", "か",
            "な", "ら", "ね", "よ", "さ", "な", "ぞ", "ぜ", "わ", "だ",
            "です", "ます", "だ", "である", "たり", "とか", "やら", "なり",
            "っぽい", "みたい", "そう", "みたい", "かな", "ね", "よ", "わ",
            "ましょう", "ましょうか", "て", "ての", "たら", "たら", "ば",
        },
        "particles": ["は", "が", "を", "に", "で", "へ", "と", "も", "の", "か"],
        "question_words": ["いつ", "どこ", "だれ", "なに", "どう", "なぜ", "どちら", "いくら"],
        "stopwords": ["は", "が", "を", "に", "で", "へ", "と", "も", "の", "か", "な"],
    },
    "ko": {
        "words": {
            "은", "는", "이", "가", "을", "를", "에", "에서", "로", "부터",
            "까지", "하고", "와", "과", "또는", "또한", "그리고", "하지만", "하지만",
            "그래서", "그래서", "또", "다른", "같은", "그", "이", "그녀", "우리",
            "당신", "나", "내", "당신의", "나의", "제", "저", "저의", "저는", "제가",
        },
        "particles": ["은", "는", "이", "가", "을", "를", "에", "에서", "로"],
        "question_words": ["누구", "무엇", "어디", "언제", "어떻게", "왜", "몇"],
        "stopwords": ["은", "는", "이", "가", "을", "를", "에", "에서", "로", "과", "와"],
    },
    "zh": {
        "words": {
            "的", "了", "是", "在", "有", "我", "不", "人", "他", "我们", "你们",
            "这个", "那个", "什么", "怎么", "哪里", "什么时候", "为什么", "多少",
            "和", "与", "或", "但", "而", "所以", "因为", "如果", "如果", "就",
        },
        "particles": ["的", "了", "是", "吗", "呢", "吧", "啊", "哦", "呀", "哎"],
        "question_words": ["什么", "为什么", "哪里", "什么时候", "谁", "怎么", "多少"],
        "stopwords": ["的", "了", "是", "在", "有", "我", "他", "我们", "和", "与"],
    },
    "fr": {
        "words": {
            "le", "la", "les", "un", "une", "des", "du", "de", "dans", "et",
            "à", "pour", "que", "qui", "ce", "cette", "ces", "se", "me", "te",
            "nous", "vous", "ils", "elles", "ils", "mais", "ou", "si", "est",
            "sont", "être", "avoir", "faire", "aller", "pouvoir", "vouloir", "dire",
        },
        "particles": ["oui", "non", "eh", "ah", "hein", "ben", "enfin", "alors"],
        "question_words": ["comment", "où", "quand", "pourquoi", "qui", "quoi",
                          "lequel", "laquelle", "combien", "quel", "quelle"],
        "stopwords": ["le", "la", "les", "un", "une", "des", "du", "de", "dans", "et",
                      "à", "pour", "que", "qui", "ce", "mais", "ou", "si", "est"],
    },
    "de": {
        "words": {
            "der", "die", "das", "ein", "eine", "den", "dem", "des", "und", "oder",
            "in", "auf", "an", "bei", "mit", "aus", "von", "zu", "für", "ist",
            "sind", "war", "haben", "habe", "werden", "kann", "könnte", "muss",
            "wir", "sie", "es", "ich", "du", "du", "mein", "dein", "sein",
        },
        "particles": ["ja", "nein", "ach", "oh", "äh", "hm", "also", "dann"],
        "question_words": ["wie", "wo", "wann", "warum", "wer", "was", "welche",
                          "wie viel", "wohin", "woher"],
        "stopwords": ["der", "die", "das", "ein", "eine", "den", "dem", "des",
                      "und", "oder", "in", "auf", "an", "bei", "mit", "aus", "von", "zu"],
    },
    "es": {
        "words": {
            "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del",
            "en", "y", "a", "que", "por", "con", "para", "por", "pro", "como",
            "ser", "estar", "tener", "hacer", "ir", "poder", "decir", "ver",
            "yo", "tú", "él", "ella", "nosotros", "vosotros", "ellos", "ellas",
        },
        "particles": ["sí", "no", "ay", "oh", "eh", "bueno", "pues", "vale"],
        "question_words": ["cómo", "dónde", "cuándo", "por qué", "quién", "qué",
                          "cuánto", "cuál", "cuáles"],
        "stopwords": ["el", "la", "los", "las", "un", "una", "de", "del", "en",
                      "y", "a", "que", "por", "con", "para", "como", "ser", "estar"],
    },
}

# Fallback language codes for Whisper/NLLB
LANGUAGE_CODE_MAP = {
    "vi": "vie_Latn",
    "en": "eng_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "zh": "zho_Hans",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
}


class LanguageDetector:
    """Multi-language detection and tracking for speech transcription."""

    def __init__(self, min_confidence: float = 0.3):
        """
        Args:
            min_confidence: Minimum confidence threshold for language detection.
        """
        self.min_confidence = min_confidence

        # Track recent language detections for consistency
        self._recent_languages: deque = deque(maxlen=10)
        self._language_history: List[Dict] = []

    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect the primary language of a text.

        Args:
            text: Input text string.

        Returns:
            Tuple of (language_code, confidence_score).
            language_code is one of: vi, en, ja, ko, zh, fr, de, es
            confidence_score is between 0.0 and 1.0
        """
        if not text or not text.strip():
            return ("unknown", 0.0)

        text_lower = text.lower().strip()

        # Remove common artifacts and special characters
        text_clean = self._clean_text(text_lower)

        if not text_clean:
            return ("unknown", 0.0)

        scores = {}

        for lang, indicators in LANGUAGE_INDICATORS.items():
            score = self._calculate_language_score(text_clean, indicators)
            scores[lang] = score

        # Get best language
        best_lang = max(scores, key=scores.get)
        best_score = scores[best_lang]

        # Normalize score
        total_score = sum(scores.values())
        if total_score > 0:
            normalized_score = best_score / total_score
        else:
            normalized_score = 0.0

        # Update tracking
        self._recent_languages.append((best_lang, normalized_score))

        # Only track if above minimum confidence
        if normalized_score >= self.min_confidence:
            self._language_history.append({
                "text": text[:50],
                "language": best_lang,
                "confidence": normalized_score,
                "timestamp": 0,  # Will be set by caller
            })

        return (best_lang, normalized_score)

    def detect_mixed_language(self, text: str, threshold: float = 0.2) -> List[Tuple[str, float]]:
        """Detect if text contains mixed languages.

        Args:
            text: Input text string.
            threshold: Minimum proportion to be considered a separate language.

        Returns:
            List of (language_code, proportion) tuples, sorted by proportion.
            Only includes languages above threshold.
        """
        if not text or not text.strip():
            return []

        # Split into words
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return []

        # Count language indicators per word
        lang_counts: Dict[str, int] = defaultdict(int)
        total_words = len(words)

        for word in words:
            for lang, indicators in LANGUAGE_INDICATORS.items():
                if word in indicators["words"] or word in indicators["particles"]:
                    lang_counts[lang] += 1
                    break

        # Calculate proportions
        proportions = []
        for lang, count in lang_counts.items():
            proportion = count / total_words
            if proportion >= threshold:
                proportions.append((lang, proportion))

        return sorted(proportions, key=lambda x: x[1], reverse=True)

    def track_language_consistency(
        self,
        chunks: List[Dict],
        tolerance: float = 0.3,
    ) -> Dict:
        """Track language consistency across multiple transcription chunks.

        Args:
            chunks: List of chunk dicts with 'text' and optionally 'detected_lang'.
            tolerance: Tolerance for considering languages as consistent.

        Returns:
            Dict with consistency analysis:
                - primary_language: Most common detected language
                - consistency_score: How consistent the languages are (0-1)
                - mixed_language: True if multiple languages detected
                - switch_points: List of chunk indices where language switched
                - confidence_trend: List of confidence scores over time
        """
        if not chunks:
            return {
                "primary_language": "unknown",
                "consistency_score": 0.0,
                "mixed_language": False,
                "switch_points": [],
                "confidence_trend": [],
            }

        # Analyze each chunk
        language_counts: Dict[str, int] = defaultdict(int)
        language_confidences: Dict[str, List[float]] = defaultdict(list)
        detected_languages: List[Tuple[int, str, float]] = []

        for idx, chunk in enumerate(chunks):
            text = chunk.get("text", "")
            detected_lang = chunk.get("detected_lang")
            confidence = chunk.get("confidence", 0.0)

            if detected_lang:
                # Use provided detection
                language_counts[detected_lang] += 1
                language_confidences[detected_lang].append(confidence)
                detected_languages.append((idx, detected_lang, confidence))
            elif text:
                # Detect language
                lang, conf = self.detect_language(text)
                if conf >= self.min_confidence:
                    language_counts[lang] += 1
                    language_confidences[lang].append(conf)
                    detected_languages.append((idx, lang, conf))

        if not detected_languages:
            return {
                "primary_language": "unknown",
                "consistency_score": 0.0,
                "mixed_language": False,
                "switch_points": [],
                "confidence_trend": [],
            }

        # Determine primary language
        primary_language = max(language_counts, key=language_counts.get)
        primary_count = language_counts[primary_language]

        # Calculate consistency score
        total_chunks = len(detected_languages)
        consistency_score = primary_count / total_chunks

        # Detect mixed language
        mixed_language = len(language_counts) > 1

        # Find switch points
        switch_points = []
        for i in range(1, len(detected_languages)):
            if detected_languages[i][1] != detected_languages[i - 1][1]:
                switch_points.append(detected_languages[i][0])

        # Confidence trend
        confidence_trend = [conf for _, _, conf in detected_languages]

        return {
            "primary_language": primary_language,
            "consistency_score": consistency_score,
            "mixed_language": mixed_language,
            "switch_points": switch_points,
            "confidence_trend": confidence_trend,
        }

    def get_session_language(self) -> Optional[str]:
        """Get the primary language for the current session based on history.

        Returns:
            Primary language code or None if insufficient data.
        """
        if not self._recent_languages:
            return None

        # Count recent languages
        lang_counts: Dict[str, int] = defaultdict(int)
        for lang, _ in self._recent_languages:
            lang_counts[lang] += 1

        # Get most common
        primary = max(lang_counts, key=lang_counts.get)
        count = lang_counts[primary]

        # Require at least 50% of recent detections
        if count / len(self._recent_languages) >= 0.5:
            return primary

        return None

    def _clean_text(self, text: str) -> str:
        """Clean text for language detection."""
        # Remove URLs, emails, code
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        text = re.sub(r'\S+@\S+', ' ', text)
        text = re.sub(r'`[^`]+`', ' ', text)
        text = re.sub(r'\b[A-Z]{2,}\b', ' ', text)  # Remove all-caps words
        return text.strip()

    def _calculate_language_score(self, text: str, indicators: Dict) -> float:
        """Calculate language score based on word indicators."""
        words = re.findall(r'\b\w+\b', text)
        if not words:
            return 0.0

        # Count matches
        word_matches = 0
        particle_matches = 0

        for word in words:
            # Check words
            if word in indicators["words"]:
                word_matches += 1
            # Check particles
            elif word in indicators["particles"]:
                particle_matches += 0.5
            # Check question words (partial match)
            else:
                for qword in indicators["question_words"]:
                    if qword in word:
                        word_matches += 0.3
                        break

        # Calculate score (weighted)
        total_matches = word_matches + particle_matches
        score = total_matches / len(words)

        # Apply bonus for question words presence
        question_count = sum(
            1 for qword in indicators["question_words"]
            if any(qword in word for word in words)
        )
        score += min(question_count * 0.02, 0.2)

        return min(score, 1.0)
