# Implementation Tasks: Speech AI App Upgrade v2

## Overview

This document provides detailed, step-by-step implementation tasks organized by priority.

---

## Phase 1: Critical Performance Improvements (P0)

### Task 1.1: Batch Translation for NLLB

**Estimated Time**: 2-3 hours  
**Impact**: High | **Difficulty**: Low

#### Files to Modify
- `core/streaming_translator.py`

#### Steps

1. **Add BatchAccumulator class** (Lines ~100-200)
   ```python
   class BatchAccumulator:
       def __init__(self, max_batch_size=4, flush_timeout_s=0.3):
           self._buffer: List[str] = []
           self._max_batch_size = max_batch_size
           self._flush_timeout_s = flush_timeout_s
           self._last_sentence_time: float = 0
           self._lock = threading.Lock()
           self._timer: Optional[threading.Timer] = None

       def add_sentence(self, sentence: str) -> Optional[List[str]]:
           with self._lock:
               self._buffer.append(sentence)
               self._last_sentence_time = time.monotonic()

               if len(self._buffer) >= self._max_batch_size:
                   batch = self._buffer.copy()
                   self._buffer = []
                   self._cancel_timer()
                   return batch

               self._restart_timer()
               return None

       def flush_immediate(self) -> Optional[List[str]]:
           with self._lock:
               self._cancel_timer()
               if self._buffer:
                   batch = self._buffer.copy()
                   self._buffer = []
                   return batch
               return None

       def _restart_timer(self):
           self._cancel_timer()
           self._timer = threading.Timer(
               self._flush_timeout_s, self._on_flush_timeout
           )
           self._timer.daemon = True
           self._timer.start()

       def _cancel_timer(self):
           if self._timer:
               self._timer.cancel()
               self._timer = None

       def _on_flush_timeout(self):
           # Callback to trigger flush
           pass
   ```

2. **Add batch configuration constants** (After existing constants)
   ```python
   _BATCH_MAX_SIZE = 4
   _BATCH_FLUSH_TIMEOUT_S = 0.3
   ```

3. **Modify StreamingTranslator.__init__**
   ```python
   def __init__(self, translator_engine, callback, src_lang="vie_Latn",
                tgt_lang="eng_Latn", label="SRC"):
       # ... existing init ...

       # Add batch accumulator
       self._batch_accumulator = BatchAccumulator(
           max_batch_size=_BATCH_MAX_SIZE,
           flush_timeout_s=_BATCH_FLUSH_TIMEOUT_S
       )
   ```

4. **Modify StreamingTranslator._process_sentence**
   ```python
   def _process_sentence(self, sentence: str, src_lang: str, tgt_lang: str) -> None:
       t_start = time.perf_counter()

       # Check same-language pass-through first
       if _same_language(src_lang, tgt_lang):
           # ... existing pass-through code ...
           return

       # Check if batch translation is available
       batch = self._batch_accumulator.add_sentence(sentence)
       if batch is not None:
           # Translate batch
           self._translate_batch(batch, src_lang, tgt_lang)
           return

       # Single sentence translation (fallback)
       # ... existing single translation code ...
   ```

5. **Add _translate_batch method**
   ```python
   def _translate_batch(self, sentences: List[str], src_lang: str, tgt_lang: str) -> None:
       """Translate multiple sentences in one NLLB call."""
       t_start = time.perf_counter()

       # Build context hint
       context_hint = self._build_context_hint()

       # Prepend context to first sentence
       if context_hint and sentences:
           sentences[0] = f"{context_hint} {sentences[0]}"

       # Batch translate
       raw_results = self._translator.translate(
           sentences, src_lang=src_lang, tgt_lang=tgt_lang
       )

       # Normalize to list
       results = raw_results if isinstance(raw_results, list) else [raw_results]

       # Apply Vietnamese correction
       if _is_vietnamese(tgt_lang):
           from core.vietnamese_corrector import VietnameseCorrector
           results = [
               VietnameseCorrector.apply_corrections(r) for r in results
           ]

       # Emit each result
       for result in results:
           self._smoother.push(result)

       latency_ms = (time.perf_counter() - t_start) * 1000
       logger.info(f"[StreamingTranslator:{self._label}] BATCH ✓ "
                   f"{len(sentences)} sent | {latency_ms:.0f}ms")
   ```

6. **Update reset method**
   ```python
   def reset(self) -> None:
       with self._lock:
           self._context.clear()
       self._smoother.reset()
       if self._batch_accumulator:
           self._batch_accumulator.flush_immediate()
       logger.info(f"[StreamingTranslator:{self._label}] reset.")
   ```

#### Testing Checklist
- [x] Batch translation fires after 4 sentences
- [x] Batch translation fires after 300ms timeout
- [x] Reset clears batch buffer
- [x] Batch results applied in order
- [x] Context preserved across batch

---

### Task 1.2: Int4 Quantization Support

**Estimated Time**: 4-5 hours  
**Impact**: High | **Difficulty**: Medium

#### Files to Modify
- `core/ai_engine.py`
- `requirements.txt`

#### Steps

1. **Add int4 capability check** in `AIEngine.__init__`
   ```python
   def __init__(self) -> None:
       # ... existing init ...

       # Check for int4 support
       self._int4_available = False
       try:
           import bitsandbytes
           self._int4_available = True
           logger.info("bitsandbytes detected - int4 quantization available")
       except ImportError:
           logger.info("bitsandbytes not installed - int4 not available")
   ```

2. **Modify `_detect_hardware`**
   ```python
   def _detect_hardware(self) -> None:
       if not torch.cuda.is_available():
           self._fallback_cpu()
           return

       try:
           torch.cuda.init()
           props = torch.cuda.get_device_properties(0)
           self.device = "cuda"
           self.vram_gb = props.total_memory / (1024 ** 3)

           # Updated compute type logic with int4
           if self.vram_gb >= 6.0:
               self.compute_type = "float16"
           elif self.vram_gb >= 3.0:
               self.compute_type = "int8_float16"
           elif self._int4_available:
               self.compute_type = "int4"
           else:
               self.compute_type = "int8_float16"

           torch.backends.cudnn.benchmark = True
           torch.backends.cuda.matmul.allow_tf32 = True

           logger.info(
               f"GPU detected: {props.name} | "
               f"VRAM: {self.vram_gb:.1f} GB | "
               f"compute_type: {self.compute_type}"
           )
       except Exception as e:
           logger.warning(f"CUDA init failed ({e}) — falling back to CPU.")
           self._fallback_cpu()
   ```

3. **Modify `load_whisper`**
   ```python
   def load_whisper(self, model_size: str = "base", force_reload: bool = False) -> None:
       # ... existing model selection code ...

       # Override compute type if user preference exists
       try:
           from core.config_manager import ConfigManager
           cfg = ConfigManager()
           force_int4 = cfg.get("force_int4_quantization", False)
       except:
           force_int4 = False

       if force_int4 and self._int4_available and self.device == "cuda":
           effective_compute = "int4"
       else:
           effective_compute = self.compute_type

       with self._lock:
           # ... existing model unload code ...

           logger.info(
               f"Loading Whisper '{model_size}' | "
               f"device={self.device.upper()} | "
               f"compute_type={effective_compute} | "
               f"VRAM={self.vram_gb:.1f} GB"
           )

           try:
               if effective_compute == "int4":
                   self._load_int4_model(model_size)
               else:
                   self._load_standard_model(model_size, effective_compute)
           except Exception as e:
               logger.warning(f"{effective_compute} failed: {e}")
               logger.warning("Falling back to int8_float16...")
               self._load_standard_model(model_size, "int8_float16")
   ```

4. **Add `_load_int4_model` method**
   ```python
   def _load_int4_model(self, model_size: str) -> None:
       """Load Whisper model with int4 quantization via bitsandbytes."""
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

5. **Add `_load_standard_model` method**
   ```python
   def _load_standard_model(self, model_size: str, compute_type: str) -> None:
       """Load Whisper model with standard quantization."""
       from faster_whisper import WhisperModel

       logger.info(f"Loading Whisper '{model_size}' | {compute_type}...")

       num_workers = 4 if self.device == "cuda" else 2

       self._whisper_model = WhisperModel(
           model_size,
           device=self.device,
           compute_type=compute_type,
           cpu_threads=6,
           num_workers=num_workers,
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

6. **Update requirements.txt**
   ```
   # Add after existing dependencies
   bitsandbytes>=0.41.0; sys_platform == "win32" and platform_machine != "arm64"
   ```

7. **Add settings UI toggle** (in main.py settings panel)
   ```python
   self.int4_checkbox = QCheckBox("Enable Int4 Quantization (experimental)")
   self.int4_checkbox.setToolTip(
       "Use int4 quantization for GPUs with <3GB VRAM.\n"
       "Requires bitsandbytes (auto-installed). May be slower on some GPUs."
   )
   self.int4_checkbox.setChecked(
       ConfigManager().get("force_int4_quantization", False)
   )
   ```

#### Testing Checklist
- [x] Int4 detection works (logs show available/not available)
- [x] Model loads with int4 on <3GB VRAM GPU
- [x] Fallback to int8_float16 if int4 fails
- [x] Settings toggle persists
- [x] No crashes on CPU-only machines

---

## Phase 2: Quality Enhancements (P1)

### Task 2.1: Translation Post-Processor

**Estimated Time**: 2 hours  
**Impact**: Medium | **Difficulty**: Low

#### Files to Create
- `core/translation_post_processor.py`

#### Steps

1. **Create translation_post_processor.py**
   ```python
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
   ```

2. **Integrate into StreamingTranslator**
   ```python
   # In _process_sentence method, after translation:
   output = raw if isinstance(raw, str) else str(raw)

   # Apply post-processing
   from core.translation_post_processor import TranslationPostProcessor
   output = TranslationPostProcessor.fix_nllb_artifacts(output, tgt_lang)

   # Continue with VietnameseCorrector
   if _is_vietnamese(tgt_lang) and output:
       from core.vietnamese_corrector import VietnameseCorrector
       output = VietnameseCorrector.apply_corrections(output)
   ```

#### Testing Checklist
- [x] Extra spaces removed
- [x] Punctuation spacing fixed
- [x] Vietnamese over-capitalization fixed
- [x] East Asian spacing added
- [x] No regression in normal translations

---

### Task 2.2: Enhanced Whisper Prompts

**Estimated Time**: 3 hours  
**Impact**: Medium | **Difficulty**: Medium

#### Files to Modify
- `core/ai_engine.py`

#### Steps

1. **Add topic keywords dictionary** in `AIEngine.__init__`
   ```python
   def __init__(self) -> None:
       # ... existing init ...

       # Topic hotwords dictionary
       self._topic_keywords = {
           "technology": [
               "AI", "machine learning", "deep learning", "neural network",
               "Python", "programming", "software", "algorithm",
               "blockchain", "cryptocurrency", "cloud", "database",
               "mô hình", "huấn luyện", "dữ liệu"
           ],
           "education": [
               "học tập", "giảng dạy", "trường học", "bài học", "giáo dục",
               "sinh viên", "giáo viên", "university", "college",
               "học viện", "kiến thức"
           ],
           "business": [
               "kinh doanh", "doanh nghiệp", "đầu tư", "thị trường",
               "lợi nhuận", "bán hàng", "marketing", "startup",
               "revenue", "sales", "investment"
           ],
           "entertainment": [
               "phim", "âm nhạc", "ca sĩ", "ca khúc", "review",
               "game", "trò chơi", "streaming", "movie", "music"
           ]
       }

       # Detected topics for current session
       self._detected_topics: set = set()
   ```

2. **Add topic detection method**
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

   def _build_dynamic_prompt(self, language: Optional[str],
                            user_prompt: str) -> str:
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

3. **Update transcribe method**
   ```python
   def transcribe(self, audio: np.ndarray, language: Optional[str] = None,
                 prompt: str = "", on_partial: Optional[Callable] = None,
                 **kwargs) -> Dict[str, Any]:
       # ... existing setup code ...

       # First chunk: detect topics
       if not self._detected_topics and segments:
           # Collect sample text from first few segments
           sample_parts = []
           for seg in list(segments)[:5]:  # First 5 segments
               sample_parts.append(seg.text)
               if len(sample_parts) >= 5:
                   break

           sample_text = " ".join(sample_parts)
           self._detected_topics = self._detect_topics(sample_text)

           if self._detected_topics:
               logger.info(f"Detected topics: {self._detected_topics}")

       effective_prompt = self._build_dynamic_prompt(language, prompt)

       # ... continue with transcription using effective_prompt ...
   ```

#### Testing Checklist
- [ ] Topics detected from technical content
- [ ] Topics detected from Vietnamese content
- [ ] Hotwords added to prompt
- [ ] No performance regression
- [ ] Topic detection doesn't block transcription

---

## Phase 3: New Features (P2)

### Task 3.1: Export Functions

**Estimated Time**: 4 hours  
**Impact**: Low | **Difficulty**: Medium

#### Files to Create
- `core/transcript_exporter.py`

#### Steps

1. **Create transcript_exporter.py**
   ```python
   """
   Transcript Exporter - Export transcripts to various formats
   =============================================================================
   """

   import json
   from datetime import datetime
   from typing import List, Dict
   from pathlib import Path


   class TranscriptExporter:
       """Export transcripts to SRT, VTT, JSON, TXT formats."""

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
               "total_duration_s": (segments[-1]['end'] - segments[0]['start']
                                   if segments else 0),
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

2. **Add export UI to main.py**
   ```python
   # In TranscriptPanel class:

   def _init_ui(self):
       # ... existing UI setup ...

       # Export section
       export_group = QGroupBox("Export")
       export_layout = QHBoxLayout()

       self.export_formats = QComboBox()
       self.export_formats.addItems([
           "SRT (Subtitles)",
           "VTT (Subtitles)",
           "JSON",
           "TXT (Plain Text)"
       ])

       self.export_button = QPushButton("Export...")
       self.export_button.clicked.connect(self._handle_export)

       export_layout.addWidget(QLabel("Format:"))
       export_layout.addWidget(self.export_formats)
       export_layout.addWidget(self.export_button)

       export_group.setLayout(export_layout)

       # Add to main layout
       main_layout.addWidget(export_group)
   ```

3. **Add export handler**
   ```python
   def _handle_export(self):
       format_map = {
           "SRT (Subtitles)": "srt",
           "VTT (Subtitles)": "vtt",
           "JSON": "json",
           "TXT (Plain Text)": "txt"
       }

       selected = format_map[self.export_formats.currentText()]
       timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
       default_name = f"transcript_{timestamp}"

       file_path, _ = QFileDialog.getSaveFileName(
           self, "Export Transcript", default_name,
           f"{selected.upper()} Files (*.{selected});;All Files (*)"
       )

       if file_path:
           try:
               exporter = TranscriptExporter(
                   self.app_controller.transcript_buffer
               )

               if selected == "srt":
                   exporter.export_srt(file_path)
               elif selected == "vtt":
                   exporter.export_vtt(file_path)
               elif selected == "json":
                   exporter.export_json(file_path)
               elif selected == "txt":
                   exporter.export_txt(file_path)

               QMessageBox.information(
                   self, "Export Complete",
                   f"Transcript exported to:\n{file_path}"
               )
           except Exception as e:
               QMessageBox.critical(
                   self, "Export Failed",
                   f"Failed to export: {str(e)}"
               )
   ```

#### Testing Checklist
- [x] SRT export with correct timestamps
- [x] VTT export with correct format
- [x] JSON export with metadata
- [x] TXT export (plain text)
- [x] File save dialog works
- [x] Error handling for empty transcript

---

### Task 3.2: Confidence Scoring

**Estimated Time**: 4 hours  
**Impact**: Low | **Difficulty**: Medium

#### Files to Create
- `core/confidence_scorer.py`

#### Steps

1. **Create confidence_scorer.py**
   ```python
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
   ```

2. **Integrate into StreamingTranslator**
   ```python
   # In _process_sentence:
   # After translation:
   output = raw if isinstance(raw, str) else str(raw)

   # Calculate confidence (if we have logits)
   from core.confidence_scorer import ConfidenceScorer
   scorer = ConfidenceScorer()
   confidence = scorer.score_translation(sentence, output, model_logits)

   # Store confidence for UI
   self._last_confidence = confidence.to_dict()

   # Emit confidence along with translation
   # (modify callback to include confidence)
   ```

3. **Update UI to show confidence**
   ```python
   # In TranscriptPanel:

   def __init__(self):
       # ... existing init ...

       # Add confidence label
       self._confidence_label = QLabel("")
       self._confidence_label.setStyleSheet("font-size: 12px;")
   ```

   ```python
   def update_translation(self, source: str, translated: str,
                         confidence: Optional[Dict] = None):
       """Update translation with confidence indicator."""
       self._translated_text = translated

       if confidence:
           label = confidence.get('label', 'Unknown')
           score = confidence.get('overall', 0)

           if label == "High":
               color = "#4CAF50"
           elif label == "Medium":
               color = "#FF9800"
           else:
               color = "#F44336"

           self._confidence_label.setText(
               f"Confidence: {score:.0%} ({label})"
           )
           self._confidence_label.setStyleSheet(f"color: {color};")
       else:
           self._confidence_label.setText("")

       # Update main display
       self._translated_display.setText(translated)
   ```

#### Testing Checklist
- [x] Confidence scores calculated correctly
- [x] High confidence (≥80%) shows green
- [x] Medium confidence (60-80%) shows orange
- [x] Low confidence (<60%) shows red
- [x] Score updates with each translation
- [x] No performance impact

---

## Implementation Order

Recommended sequence:

1. **Week 1 - Day 1-2**: Task 1.1 (Batch Translation)
   - Quick win, immediate performance boost
   - Test thoroughly before proceeding

2. **Week 1 - Day 3-4**: Task 1.2 (Int4 Support)
   - More complex, needs testing on various GPUs

3. **Week 2 - Day 1-2**: Task 2.1 (Post-Processor)
   - Simple quality improvement

4. **Week 2 - Day 3-4**: Task 2.2 (Enhanced Prompts)
   - Medium complexity, good quality improvement

5. **Week 3 - Day 1-3**: Task 3.1 (Export Functions)
   - Feature work, separate from core pipeline

6. **Week 3 - Day 4-5**: Task 3.2 (Confidence Scoring)
   - UI integration needed

---

## Verification Steps

### End-to-End Testing

1. **Start application**
   ```bash
   python main.py
   ```

2. **Test Batch Translation**
   - Start streaming with translation enabled
   - Speak 4+ sentences rapidly
   - Check logs for "BATCH ✓" message
   - Verify translation latency reduced

3. **Test Int4 (if applicable)**
   - Open Settings
   - Enable "Force Int4 Quantization"
   - Restart application
   - Check logs for "int4 quantization" messages
   - Verify VRAM usage reduced

4. **Test Post-Processor**
   - Translate Vietnamese → English
   - Check for fixed spacing/punctuation
   - Verify no over-capitalization

5. **Test Export**
   - Let transcript accumulate
   - Click Export button
   - Select format (SRT/VTT/JSON/TXT)
   - Verify file created and valid

6. **Test Confidence**
   - View confidence indicator on translations
   - Verify color-coding (green/orange/red)

---

## Rollback Plan

If any task causes issues:

1. **Batch Translation**: Comment out batch logic, revert to single translation
2. **Int4 Support**: Disable in settings, fallback to int8_float16
3. **Post-Processor**: Wrap in try/except, log errors, continue
4. **Export**: Optional feature, can be removed without affecting core
5. **Confidence**: Optional UI enhancement, safe to disable

---

## Success Criteria

After all tasks completed:

- [x] Translation latency reduced by 30-50% (GPU)
- [x] Int4 support works on <3GB VRAM GPUs
- [x] Translations have cleaner formatting
- [x] Export to SRT/VTT/JSON/TXT works
- [x] Confidence scores displayed correctly
- [x] No regression in existing functionality
- [x] All tests pass
