# Design Document: Speech AI App Upgrade v2

## Overview

This document details the technical implementation approach for each upgrade task.

---

## Task 1: Batch Translation for NLLB

### Current Architecture
```
Whisper Output → SentenceBuilder → StreamingTranslator → NLLB (1 sentence) → UI
                                                        ↓
                                                   One-at-a-time
```

### Target Architecture
```
Whisper Output → SentenceBuilder → BatchAccumulator (3-5 sentences)
                                           ↓
                                    StreamingTranslator
                                           ↓
                                      NLLB (batch) → UI
```

### Implementation Details

#### Modified File: `core/streaming_translator.py`

**Add BatchAccumulator Class:**
```python
class BatchAccumulator:
    """Accumulates sentences until batch threshold is reached."""

    def __init__(self, max_batch_size=4, flush_timeout_s=0.3):
        self._buffer: List[str] = []
        self._max_batch_size = max_batch_size
        self._flush_timeout_s = flush_timeout_s
        self._last_sentence_time: float = 0
        self._lock = threading.Lock()
        self._flush_callback: Optional[Callable[[List[str]], None]] = None
        self._timer: Optional[threading.Timer] = None

    def add_sentence(self, sentence: str) -> Optional[List[str]]:
        """Add sentence, return batch if ready (or None to accumulate)."""
        with self._lock:
            self._buffer.append(sentence)
            self._last_sentence_time = time.monotonic()

            # Check if batch is ready
            if len(self._buffer) >= self._max_batch_size:
                batch = self._buffer.copy()
                self._buffer = []
                self._cancel_timer()
                return batch

            # Start/restart flush timer
            self._restart_timer()
            return None

    def flush_immediate(self) -> Optional[List[str]]:
        """Force flush accumulated sentences."""
        with self._lock:
            self._cancel_timer()
            if self._buffer:
                batch = self._buffer.copy()
                self._buffer = []
                return batch
            return None
```

**Modified StreamingTranslator._process_sentence:**
```python
def _process_sentence(self, sentence: str, src_lang: str, tgt_lang: str) -> None:
    # Check if batch translation is available
    if hasattr(self, '_batch_accumulator'):
        batch = self._batch_accumulator.add_sentence(sentence)
        if batch is not None:
            # Translate batch
            self._translate_batch(batch, src_lang, tgt_lang)
            return

        # Also check if timeout reached
        if self._batch_accumulator.should_flush():
            batch = self._batch_accumulator.flush_immediate()
            if batch:
                self._translate_batch(batch, src_lang, tgt_lang)
                return
```

**New Method: _translate_batch**
```python
def _translate_batch(self, sentences: List[str], src_lang: str, tgt_lang: str) -> None:
    """Translate multiple sentences in one NLLB call."""
    t_start = time.perf_counter()

    # Build context hint from recent translations
    context_hint = self._build_context_hint()

    # Prepend context to first sentence only
    if context_hint and sentences:
        sentences[0] = f"{context_hint} {sentences[0]}"

    # Batch translate
    raw_results = self._translator.translate(
        sentences,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
    )

    # Process results
    results = raw_results if isinstance(raw_results, list) else [raw_results]

    # Apply Vietnamese correction to each result
    if _is_vietnamese(tgt_lang):
        from core.vietnamese_corrector import VietnameseCorrector
        results = [
            VietnameseCorrector.apply_corrections(r) for r in results
        ]

    # Emit each result to smoother
    for result in results:
        self._smoother.push(result)

    latency_ms = (time.perf_counter() - t_start) * 1000
    logger.info(f"[StreamingTranslator:{self._label}] BATCH ✓ "
                f"{len(sentences)} sent | {latency_ms:.0f}ms")
```

### Configuration
```python
# New constants in streaming_translator.py
_BATCH_MAX_SIZE = 4           # Accumulate 4 sentences before batch
_BATCH_FLUSH_TIMEOUT_S = 0.3  # Flush after 300ms of inactivity
```

### Edge Cases
1. **Single short sentence**: Wait for more sentences (timeout-based)
2. **Long sentence (>100 words)**: Translate immediately, don't wait
3. **Error in batch**: Fallback to individual translation

---

## Task 2: Int4 Quantization Support

### Current Architecture
```python
# ai_engine.py
if self.device == "cuda":
    if self.vram_gb >= 3.0:
        self.compute_type = "float16"
    else:
        self.compute_type = "int8_float16"
```

### Target Architecture
```python
# ai_engine.py
if self.device == "cuda":
    if self.vram_gb >= 6.0:
        self.compute_type = "float16"
    elif self.vram_gb >= 3.0:
        self.compute_type = "int8_float16"
    else:
        self.compute_type = "int4"  # NEW: bitsandbytes int4
```

### Implementation Details

#### Modified File: `core/ai_engine.py`

**Add Int4 Loading Logic:**
```python
def load_whisper(self, model_size: str = "base", force_reload: bool = False) -> None:
    # ... existing code ...

    # Determine compute type based on VRAM
    if self.device == "cuda":
        if self.vram_gb >= 6.0:
            effective_compute = "float16"
        elif self.vram_gb >= 3.0:
            effective_compute = "int8_float16"
        else:
            # Try int4 for low-VRAM GPUs
            effective_compute = self._try_int4_quantization()

    # Load model
    try:
        if effective_compute == "int4":
            self._load_int4_model(model_size)
        else:
            self._load_standard_model(model_size, effective_compute)
    except Exception as e:
        logger.warning(f"{effective_compute} failed, falling back to int8_float16")
        self._load_standard_model(model_size, "int8_float16")
```

**New Method: _try_int4_quantization**
```python
def _try_int4_quantization(self) -> str:
    """Test if bitsandbytes is available for int4 quantization."""
    try:
        import bitsandbytes
        logger.info("bitsandbytes detected - int4 quantization available")
        return "int4"
    except ImportError:
        logger.warning("bitsandbytes not installed - skipping int4")
        return "int8_float16"
```

**New Method: _load_int4_model**
```python
def _load_int4_model(self, model_size: str) -> None:
    """Load Whisper model with int4 quantization."""
    from faster_whisper import WhisperModel

    logger.info(f"Loading Whisper '{model_size}' with int4 quantization...")

    self._whisper_model = WhisperModel(
        model_size,
        device=self.device,
        compute_type="int4_float16",  # bitsandbytes int4
        cpu_threads=4,
        num_workers=2,
        download_root=None,
        local_files_only=False,
    )

    self._whisper_model_name = model_size
    self._batched_pipeline = None
    self._batch_size = 1

    logger.info(
        f"Whisper '{model_size}' loaded with int4 | "
        f"CUDA | int4_float16 | VRAM={self.vram_gb:.1f} GB"
    )
```

**Add Fallback Logic:**
```python
def _load_standard_model(self, model_size: str, compute_type: str) -> None:
    """Load Whisper model with standard quantization."""
    from faster_whisper import WhisperModel

    logger.info(f"Loading Whisper '{model_size}' | {compute_type}...")

    self._whisper_model = WhisperModel(
        model_size,
        device=self.device,
        compute_type=compute_type,
        cpu_threads=6,
        num_workers=4 if self.device == "cuda" else 2,
        download_root=None,
        local_files_only=False,
    )

    self._whisper_model_name = model_size
    self._batched_pipeline = None
    self._batch_size = 1

    logger.info(
        f"Whisper '{model_size}' loaded | "
        f"{self.device.upper()} | {compute_type}"
    )
```

#### Modified File: `requirements.txt`
```
# Add optional int4 support
bitsandbytes>=0.41.0; sys_platform == "win32" and cuda_available
```

**Note**: Use conditional dependency - only install on Windows with CUDA

### Hardware Detection Updates

```python
def _detect_hardware(self) -> None:
    # ... existing code ...

    # Check for int4 capability
    self._int4_available = False
    try:
        import bitsandbytes
        self._int4_available = True
    except ImportError:
        pass

    logger.info(f"Int4 quantization: {'Available' if self._int4_available else 'Not available'}")
```

### Configuration UI Update

Add to settings UI:
```
[ ] Enable Int4 Quantization (for GPUs with <3GB VRAM)
    → Requires bitsandbytes (auto-installed)
```

---

## Task 3: Translation Post-Processor

### New File: `core/translation_post_processor.py`

```python
"""
Translation Post-Processor - Fix common NLLB artifacts
=============================================================================
"""

import re
from typing import Optional

class TranslationPostProcessor:
    """Post-process NLLB translation output to fix common artifacts."""

    # Common NLLB error patterns
    EXTRA_SPACES = re.compile(r'\s+')
    PUNCT_SPACING = re.compile(r'\s([.,!?;:)]+)')
    LEADING_PUNCT = re.compile(r'^([.,!?;:]+)')
    TRAILING_PUNCT = re.compile(r'([.,!?;:]+)\s*$')

    # Vietnamese-specific fixes
    VI_CAP_FIX = re.compile(r'\b(vi\w*)\b', re.IGNORECASE)

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
        text = re.sub(r'\b([A-Z]{2,})\b', lambda m: m.group(1).capitalize(), text)

        # Fix tone mark positioning (common NLLB error)
        # This is a simplified version - full implementation needs Unicode normalization
        text = cls._normalize_unicode(text)

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
    def _normalize_unicode(cls, text: str) -> str:
        """Normalize Unicode combining marks for Vietnamese."""
        import unicodedata
        return unicodedata.normalize('NFC', text)

    @classmethod
    def _is_vietnamese(cls, lang: str) -> bool:
        return lang.lower() in ('vie_latn', 'vi', 'vietnamese')

    @classmethod
    def _is_east_asian(cls, lang: str) -> bool:
        return lang.lower() in ('jpn_jpan', 'japanese', 'kor_hang', 'korean',
                                'zho_hans', 'chinese')
```

#### Integration in StreamingTranslator

```python
def _process_sentence(self, sentence: str, src_lang: str, tgt_lang: str) -> None:
    # ... existing translation code ...

    output = raw if isinstance(raw, str) else str(raw)

    # Apply post-processing
    from core.translation_post_processor import TranslationPostProcessor
    output = TranslationPostProcessor.fix_nllb_artifacts(output, tgt_lang)

    # ... continue with VietnameseCorrector ...
```

---

## Task 4: Enhanced Whisper Prompts

### Modified File: `core/ai_engine.py`

**Add Topic Detection (Lightweight):**
```python
class AIEngine:
    # ... existing code ...

    def __init__(self):
        # ... existing init ...

        # Topic hotwords dictionary
        self._topic_keywords = {
            "technology": [
                "AI", "machine learning", "deep learning", "neural network",
                "Python", "programming", "software", "algorithm",
                "blockchain", "cryptocurrency", "cloud", "database"
            ],
            "education": [
                "học tập", "giảng dạy", "trường học", "bài học", "giáo dục",
                "sinh viên", "giáo viên", "universit", "học viện"
            ],
            "business": [
                "kinh doanh", "doanh nghiệp", "đầu tư", "thị trường",
                "lợi nhuận", "bán hàng", "marketing", "startup"
            ],
            "entertainment": [
                "phim", "âm nhạc", "ca sĩ", "ca khúc", "review",
                "game", "trò chơi", "streaming"
            ]
        }

        # Detected topics for current session
        self._detected_topics: set = set()
```

**Add Topic Detection Method:**
```python
def _detect_topics(self, text: str) -> set:
    """Detect topics from text using keyword matching."""
    detected = set()
    text_lower = text.lower()

    for topic, keywords in self._topic_keywords.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                detected.add(topic)
                break

    return detected

def _build_dynamic_prompt(self, language: Optional[str], user_prompt: str) -> str:
    """Build prompt with detected topic hotwords."""
    if language == "vi":
        base = self.vi_base_prompt.strip()

        # Add detected topics
        if self._detected_topics:
            topic_keywords = []
            for topic in self._detected_topics:
                topic_keywords.extend(self._topic_keywords.get(topic, []))

            if topic_keywords:
                base += f"\nTừ khóa chuyên ngành: {', '.join(topic_keywords[:10])}"

        if user_prompt:
            return f"{base} {user_prompt.strip()}"
        return base

    return user_prompt
```

**Update Transcribe Method:**
```python
def transcribe(self, audio: np.ndarray, language: Optional[str] = None,
               prompt: str = "", **kwargs) -> Dict[str, Any]:
    # ... existing code ...

    # Detect topics from first chunk
    if not self._detected_topics:
        # Extract initial text for topic detection
        sample_text = self._extract_sample_text(segments)
        self._detected_topics = self._detect_topics(sample_text)

        if self._detected_topics:
            logger.info(f"Detected topics: {self._detected_topics}")

    effective_prompt = self._build_dynamic_prompt(language, prompt)

    # ... continue with transcription ...
```

---

## Task 5: Export Functions

### New File: `core/transcript_exporter.py`

```python
"""
Transcript Exporter - Export transcripts to various formats
=============================================================================
"""

import json
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path


class TranscriptExporter:
    """Export transcripts to SRT, VTT, JSON formats."""

    def __init__(self, transcript_buffer):
        self._buffer = transcript_buffer

    def export_srt(self, output_path: str) -> str:
        """Export to SRT subtitle format."""
        segments = self._buffer.get_all_segments()

        srt_content = ""
        for idx, seg in enumerate(segments, 1):
            start = self._format_timestamp(seg['start'])
            end = self._format_timestamp(seg['end'])
            text = seg['text']

            srt_content += f"{idx}\n"
            srt_content += f"{start} --> {end}\n"
            srt_content += f"{text}\n\n"

        Path(output_path).write_text(srt_content, encoding='utf-8')
        return output_path

    def export_vtt(self, output_path: str) -> str:
        """Export to VTT subtitle format."""
        segments = self._buffer.get_all_segments()

        vtt_content = "WEBVTT\n\n"
        for idx, seg in enumerate(segments, 1):
            start = self._format_vtt_timestamp(seg['start'])
            end = self._format_vtt_timestamp(seg['end'])
            text = seg['text']

            vtt_content += f"{idx}\n"
            vtt_content += f"{start} --> {end}\n"
            vtt_content += f"{text}\n\n"

        Path(output_path).write_text(vtt_content, encoding='utf-8')
        return output_path

    def export_json(self, output_path: str) -> str:
        """Export to JSON with metadata."""
        segments = self._buffer.get_all_segments()

        export_data = {
            "exported_at": datetime.now().isoformat(),
            "total_segments": len(segments),
            "total_duration_s": segments[-1]['end'] - segments[0]['start'] if segments else 0,
            "segments": segments
        }

        Path(output_path).write_text(
            json.dumps(export_data, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
        return output_path

    def export_txt(self, output_path: str) -> str:
        """Export plain text (no timestamps)."""
        segments = self._buffer.get_all_segments()
        text = "\n\n".join(seg['text'] for seg in segments)

        Path(output_path).write_text(text, encoding='utf-8')
        return output_path

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Format seconds to SRT timestamp: HH:MM:SS,mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

    @staticmethod
    def _format_vtt_timestamp(seconds: float) -> str:
        """Format seconds to VTT timestamp: HH:MM:SS.mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"
```

#### Integration in Main UI

Add export buttons to UI:
```python
# In main.py
class TranscriptPanel(QWidget):
    def __init__(self):
        # ... existing code ...

        # Export section
        export_label = QLabel("Export Transcript:")
        self.export_formats = QComboBox()
        self.export_formats.addItems(["SRT (Subtitles)", "VTT (Subtitles)", "JSON", "TXT"])

        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self._handle_export)

        # ... layout setup ...

    def _handle_export(self):
        format_map = {
            "SRT (Subtitles)": "srt",
            "VTT (Subtitles)": "vtt",
            "JSON": "json",
            "TXT": "txt"
        }

        selected = format_map[self.export_formats.currentText()]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"transcript_{timestamp}"

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Transcript", default_name,
            f"{selected.upper()} Files (*.{selected});;All Files (*)"
        )

        if file_path:
            exporter = TranscriptExporter(self.app_controller.transcript_buffer)

            if selected == "srt":
                exporter.export_srt(file_path)
            elif selected == "vtt":
                exporter.export_vtt(file_path)
            elif selected == "json":
                exporter.export_json(file_path)
            elif selected == "txt":
                exporter.export_txt(file_path)

            QMessageBox.information(self, "Export Complete",
                                   f"Transcript exported to:\n{file_path}")
```

---

## Task 6: Confidence Scoring

### New File: `core/confidence_scorer.py`

```python
"""
Confidence Scorer - Score translation quality
=============================================================================
"""

import math
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ConfidenceScore:
    """Confidence score for a translation."""
    overall: float          # 0.0 - 1.0
    translation_quality: float  # NLLB internal confidence
    length_penalty: float   # Penalize very short/long translations
    punctuation_score: float  # Check punctuation completeness

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

    def __init__(self):
        self._min_words = 2
        self._max_words = 100

    def score_translation(self, src_text: str, tgt_text: str,
                         model_logits: Optional[Dict] = None) -> ConfidenceScore:
        """Calculate confidence score for a translation."""

        # 1. Translation quality (from model if available)
        translation_quality = self._score_translation_quality(model_logits)

        # 2. Length penalty
        length_penalty = self._score_length(tgt_text)

        # 3. Punctuation check
        punct_score = self._score_punctuation(tgt_text)

        # 4. Language consistency
        lang_consistency = self._score_language_consistency(src_text, tgt_text)

        # 5. Overall score (weighted average)
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
            punctuation_score=punctuation_score,
        )

    def _score_translation_quality(self, model_logits: Optional[Dict]) -> float:
        """Score from model's internal confidence."""
        if model_logits is None:
            # Default to medium confidence when no logits available
            return 0.7

        # Extract average logprob if available
        if 'avg_logprob' in model_logits:
            logprob = model_logits['avg_logprob']
            # Convert logprob to 0-1 scale (typical range: -1 to 0)
            return min(1.0, max(0.0, (logprob + 1) / 2 + 0.5))

        return 0.7  # Default

    def _score_length(self, text: str) -> float:
        """Penalize very short or unusually long translations."""
        words = len(text.split())

        if words < self._min_words:
            return 0.3  # Too short
        elif words > self._max_words:
            return 0.5  # Too long
        else:
            return 1.0  # Normal

    def _score_punctuation(self, text: str) -> float:
        """Check if translation has proper punctuation."""
        if not text:
            return 0.0

        # Check for ending punctuation
        has_end_punct = any(text.endswith(p) for p in ['.', '!', '?', '.', '…'])

        # Check for balanced parentheses/quotes
        open_parens = text.count('(')
        close_parens = text.count(')')
        parens_balanced = open_parens == close_parens

        if has_end_punct and parens_balanced:
            return 1.0
        elif has_end_punct:
            return 0.7
        else:
            return 0.5

    def _score_language_consistency(self, src: str, tgt: str) -> float:
        """Check if translation preserved important source elements."""
        # Check for preserved proper nouns (simplified)
        import re

        # Find uppercase words in source (potential proper nouns)
        src_proper_nouns = re.findall(r'\b[A-Z]{2,}\b', src)

        if not src_proper_nouns:
            return 1.0  # Nothing to preserve

        # Check if they appear in target (case-insensitive)
        tgt_lower = tgt.lower()
        preserved = sum(1 for noun in src_proper_nouns if noun.lower() in tgt_lower)

        return preserved / len(src_proper_nouns)
```

#### Integration in UI

```python
# In main.py - TranscriptPanel
class TranscriptPanel(QWidget):
    # ... existing code ...

    def update_translation(self, source: str, translated: str,
                          confidence: Optional[Dict] = None):
        """Update translation display with confidence indicator."""

        # Set translated text
        self._translated_text = translated

        # Add confidence indicator if available
        if confidence:
            label = confidence.get('label', 'Unknown')
            score = confidence.get('overall', 0)

            # Color-code confidence
            if label == "High":
                color = "#4CAF50"  # Green
            elif label == "Medium":
                color = "#FF9800"  # Orange
            else:
                color = "#F44336"  # Red

            self._confidence_label.setText(f"Confidence: {score:.0%} ({label})")
            self._confidence_label.setStyleSheet(f"color: {color};")
        else:
            self._confidence_label.setText("")

        # ... update UI ...
```

---

## Summary of File Changes

| File | Task | Changes |
|------|------|---------|
| `core/streaming_translator.py` | 1 | Add BatchAccumulator, _translate_batch |
| `core/ai_engine.py` | 2, 4 | Int4 support, dynamic prompts |
| `core/translation_post_processor.py` | 3 | New file |
| `core/transcript_exporter.py` | 5 | New file |
| `core/confidence_scorer.py` | 6 | New file |
| `main.py` | 5, 6 | Add export buttons, confidence display |
| `requirements.txt` | 2 | Add bitsandbytes (optional) |
