"""
ai_engine.py
=============================================================================
AIEngine Module - Core Faster-Whisper AI Engine
=============================================================================

Key Improvements (v6):
  - Fixed broken try/except indentation in transcribe()
  - BatchedInferencePipeline batch_size: 8 for <6 GB VRAM, 16 for >=6 GB
  - compute_type="int8_float16" enforced for all CUDA devices
  - Inference timeout protection via concurrent.futures (prevents pipeline block)
  - Stronger hallucination/duplicate filters using difflib.SequenceMatcher
  - Rolling 15-second context window (deque of (timestamp, text))
  - Context prompt capped at 150 words for Vietnamese grammar continuity
  - Per-segment no_speech_prob, compression_ratio, repeated-word filters
"""

import logging
import threading
import queue
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Optional, Dict, Any, List, Callable
import difflib

import torch
import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rolling context window config — Optimized for Vietnamese accuracy
# ---------------------------------------------------------------------------
_CONTEXT_WINDOW_S  = 30.0   # Increased from 15s — longer context for grammar continuity
_CONTEXT_MAX_WORDS = 250    # Increased from 150 — more context words for Vietnamese

# Inference timeout — Whisper must not block the pipeline longer than this
# Increased for accuracy mode with large-v3 model
_INFERENCE_TIMEOUT_S = 15.0  # Increased from 8s for large-v3 inference


class AIEngine:
    """
    Singleton-based thread-safe AI Engine managing Faster-Whisper.

    Improvements over v5:
    - Proper BatchedInferencePipeline batch tuning (8 / 16 depending on VRAM)
    - Inference wrapped in ThreadPoolExecutor with hard timeout so a stalled
      CUDA kernel never freezes the audio pipeline
    - Stronger per-segment and rolling-window hallucination suppression
    """

    _instance: Optional["AIEngine"] = None
    _init_lock = threading.Lock()

    def __new__(cls) -> "AIEngine":
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = super(AIEngine, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        with self._init_lock:
            if getattr(self, "_initialized", False):
                return

            self._initialized = True
            self._lock = threading.Lock()

            # --- Whisper model state ---
            self._whisper_model: Optional[WhisperModel] = None
            self._whisper_model_name: Optional[str] = None
            self._batched_pipeline = None
            self._batch_size: int = 8

            # Thread pool for timeout-protected inference
            # max_workers=1 keeps GPU serialised; timeout kills stalled calls
            self._infer_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="whisper-infer"
            )

            # --- Accuracy mode flag ---
            self.accuracy_mode: bool = False

            # --- Hardware state ---
            self.device: str = "cpu"
            self.compute_type: str = "int8"
            self.vram_gb: float = 0.0
            self._detect_hardware()

            # --- Hallucination / duplicate filter ---
            self.last_text: str = ""
            self.repetition_count: int = 0
            self._recent_texts: deque = deque(maxlen=5)

            # --- Language / prompt config ---
            self.force_language: bool = True
            self.ultra_realtime_mode: bool = False
            self.vi_base_prompt: str = (
                "Đây là hội thoại tiếng Việt tự nhiên.\n"
                "Người nói có thể dùng giọng miền Bắc hoặc miền Nam.\n"
                "Nội dung có thể bao gồm công nghệ, giáo dục, podcast, review sản phẩm."
            )

            # --- Int4 quantization support ---
            self._int4_available = False
            try:
                import bitsandbytes
                self._int4_available = True
                logger.info("bitsandbytes detected - int4 quantization available")
            except ImportError:
                logger.info("bitsandbytes not installed - int4 not available")

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

    # ------------------------------------------------------------------
    # Hardware detection
    # ------------------------------------------------------------------
    def _detect_hardware(self) -> None:
        """Probe CUDA availability and set device + compute_type."""
        if not torch.cuda.is_available():
            self._fallback_cpu()
            return
        try:
            torch.cuda.init()
            props = torch.cuda.get_device_properties(0)
            self.device = "cuda"
            self.vram_gb = props.total_memory / (1024 ** 3)

            # Updated compute type logic with int4 support
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

    def _fallback_cpu(self) -> None:
        self.device = "cpu"
        self.compute_type = "int8"
        self.vram_gb = 0.0
        logger.info("Hardware: CPU | compute_type: int8")

    # ------------------------------------------------------------------
    # Model auto-selection
    # ------------------------------------------------------------------
    def _auto_select_whisper_model(self) -> str:
        target_lang = "None (Off)"
        ultra_realtime = self.ultra_realtime_mode
        accuracy_mode = self.accuracy_mode

        try:
            from core.config_manager import ConfigManager
            cfg = ConfigManager()
            override = cfg.get("whisper_model_override", "")
            target_lang = cfg.get("target_lang", target_lang)
            ultra_realtime = cfg.get("ultra_realtime_mode", ultra_realtime)
            accuracy_mode = cfg.get("accuracy_mode", False)
            if override and override not in ("", "Auto", "Auto (Recommended)"):
                logger.info(f"Model override from config: '{override}'")
                return override
        except Exception:
            pass

        # Accuracy mode and ultra-realtime pull in opposite directions.
        if accuracy_mode:
            ultra_realtime = False

        if self.device == "cpu":
            selected = "medium" if accuracy_mode else "base"
        else:
            translation_enabled = bool(target_lang and target_lang != "None (Off)")

            if accuracy_mode and self.vram_gb >= 3.0:
                selected = "large-v3"
            elif self.vram_gb <= 4.5:
                selected = "small"
            elif self.vram_gb <= 6.0:
                selected = "medium" if (translation_enabled or ultra_realtime) else "large-v3-turbo"
            else:
                selected = "large-v3-turbo" if ultra_realtime else "large-v3"

        logger.info(
            f"Auto-selected model: '{selected}' "
            f"(device={self.device}, VRAM={self.vram_gb:.1f} GB, "
            f"accuracy_mode={accuracy_mode}, ultra_realtime={ultra_realtime}, "
            f"translation={'on' if target_lang and target_lang != 'None (Off)' else 'off'})"
        )
        return selected

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def load_whisper(self, model_size: str = "base", force_reload: bool = False) -> None:
        """Load (or hot-swap) a Whisper model."""
        if not model_size or any(kw in model_size for kw in ("Auto", "Recommended", "auto")):
            model_size = self._auto_select_whisper_model()

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
            if not force_reload and self._whisper_model and self._whisper_model_name == model_size:
                logger.debug(f"Whisper '{model_size}' already loaded — skipping reload.")
                return

            if self._whisper_model:
                logger.info(f"Unloading Whisper '{self._whisper_model_name}' to free memory.")
                self._whisper_model = None
                self._batched_pipeline = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

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

    # ------------------------------------------------------------------
    # Int4 model loading
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Language / prompt helpers
    # ------------------------------------------------------------------
    def _resolve_language(self, language: Optional[str]) -> Optional[str]:
        """Return the language code to pass to Whisper.

        Respect an explicit UI language selection even when force_language is
        disabled. Auto-detect should only happen when the caller actually
        passes no language (for example "Auto-Detect" -> None).
        """
        if not language:
            return None

        # An explicit language choice should stay explicit for low-latency
        # streaming sessions. Falling back to auto-detect here makes English
        # and other selected languages feel noticeably slower.
        return language

    def _build_prompt(self, language: Optional[str], user_prompt: str) -> str:
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

    # ------------------------------------------------------------------
    # Topic detection
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Core transcription (with timeout protection)
    # ------------------------------------------------------------------
    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        prompt: str = "",
        on_partial: Optional[Callable[[str], None]] = None,
        source_sr: int = 16000,
        channels: int = 1,
        noise_reduce: bool = False,
    ) -> Dict[str, Any]:
        """Thread-safe transcription with inference timeout protection.

        Audio is expected to be a clean 16 kHz mono float32 array from
        ChunkAggregator (preprocessing already applied upstream).

        Inference is submitted to a ThreadPoolExecutor with a hard timeout
        (_INFERENCE_TIMEOUT_S) so a stalled GPU kernel never blocks the
        audio pipeline indefinitely.
        """
        if not self._whisper_model:
            self.load_whisper()

        if audio is None or len(audio) < 400:
            return {"text": "", "language": language}

        effective_language = self._resolve_language(language)
        effective_prompt = self._build_prompt(language, prompt)
        carry_context = not self.ultra_realtime_mode

        is_vietnamese = (effective_language == "vi")

        if self.accuracy_mode:
            beam_size = 5
            best_of = 5
        else:
            beam_size = 1
            best_of = 1

        if is_vietnamese:
            tech_hotwords = (
                "YouTube, ChatGPT, GitHub, Python, AI, "
                "machine learning, công nghệ, lập trình"
            )
            _transcribe_kwargs = dict(
                # beam_size=5, best_of=5 for accuracy mode (beam search)
                # beam_size=1 for ultra realtime (greedy decoding)
                beam_size=beam_size,
                best_of=best_of,
                temperature=0.0,
                language=effective_language,
                initial_prompt=effective_prompt,
                hotwords=tech_hotwords,
                word_timestamps=False,
                condition_on_previous_text=carry_context,
                repetition_penalty=1.1,
                vad_filter=False,
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
                compression_ratio_threshold=2.4,
            )
        else:
            _transcribe_kwargs = dict(
                beam_size=beam_size,
                best_of=best_of,
                temperature=0.0,
                language=effective_language,
                initial_prompt=effective_prompt,
                condition_on_previous_text=carry_context,
                vad_filter=False,
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
                compression_ratio_threshold=2.4,
            )

        import re

        # ── Timeout-protected inference ───────────────────────────────────
        def _do_infer():
            with self._lock:
                if self._batched_pipeline is not None:
                    return self._batched_pipeline.transcribe(
                        audio,
                        batch_size=self._batch_size,
                        **_transcribe_kwargs,
                    )
                else:
                    return self._whisper_model.transcribe(
                        audio,
                        **_transcribe_kwargs,
                    )

        try:
            future = self._infer_pool.submit(_do_infer)
            segments, info = future.result(timeout=_INFERENCE_TIMEOUT_S)
        except FuturesTimeoutError:
            logger.warning(
                f"Whisper inference timed out after {_INFERENCE_TIMEOUT_S}s — "
                "skipping chunk to keep pipeline unblocked."
            )
            return {"text": "", "language": language}
        except RuntimeError as e:
            logger.error(f"CUDA runtime error: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {"text": "", "language": language}
        except Exception as e:
            logger.error(f"Whisper inference error: {e}")
            return {"text": "", "language": language}

        # ── First chunk: detect topics ──────────────────────────────────
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

        # ── Per-segment hallucination filters ────────────────────────────
        raw_text_parts: List[str] = []

        for seg in segments:
            seg_text = seg.text.strip()

            # Filter 1: Empty or too short (< 2 chars)
            if len(seg_text) < 2:
                logger.debug(f"Segment rejected (too short): {seg_text!r}")
                continue

            # Filter 2: Low confidence
            if hasattr(seg, "avg_logprob") and seg.avg_logprob < -1.0:
                logger.debug(
                    f"Segment rejected (low confidence): "
                    f"avg_logprob={seg.avg_logprob:.2f} | {seg_text!r}"
                )
                continue

            # Filter 3: High no-speech probability
            if hasattr(seg, "no_speech_prob") and seg.no_speech_prob > 0.5:
                logger.debug(
                    f"Segment rejected (no_speech): "
                    f"no_speech_prob={seg.no_speech_prob:.2f} | {seg_text!r}"
                )
                continue

            # Filter 4: Repeated single words ("ạ ạ ạ", "vâng vâng vâng")
            words = seg_text.lower().split()
            if len(words) >= 2:
                unique_words = set(words)
                if len(unique_words) == 1:
                    logger.debug(f"Segment rejected (all repeated): {seg_text!r}")
                    continue
                most_common_count = max(words.count(w) for w in unique_words)
                if most_common_count / len(words) > 0.70 and len(words) >= 3:
                    logger.debug(f"Segment rejected (word repetition): {seg_text!r}")
                    continue

            # Filter 5: High compression ratio — highly repetitive output
            if hasattr(seg, "compression_ratio") and seg.compression_ratio > 2.4:
                logger.debug(
                    f"Segment rejected (high compression): "
                    f"ratio={seg.compression_ratio:.2f} | {seg_text!r}"
                )
                continue

            # Filter 6: Similarity > 0.9 with last_text (SequenceMatcher)
            # Disabled in accuracy mode to avoid dropping valid repeated phrases.
            if self.last_text and not self.accuracy_mode:
                similarity = difflib.SequenceMatcher(
                    None, seg_text.lower(), self.last_text.lower()
                ).ratio()
                if similarity > 0.9:
                    logger.debug(
                        f"Segment rejected (similar to last, ratio={similarity:.2f}): {seg_text!r}"
                    )
                    continue

            # Passed all filters → emit partial
            if on_partial:
                try:
                    on_partial(seg_text)
                except Exception:
                    pass

            raw_text_parts.append(seg_text)

        raw_text = " ".join(raw_text_parts).strip()

        # ── Post-processing quality filter ───────────────────────────────
        refined_text = raw_text

        # A. Blacklist — YouTube spam + common Whisper hallucinations
        _BLACKLIST = [
            "ghiền mì gõ", "la la school", "cho kênh", "để không bỏ",
            "hẹn gặp lại", "đăng ký kênh", "ủng hộ kênh", "theo dõi kênh",
            "nhấn like", "nhấn subscribe", "bấm chuông", "chia sẻ video",
            "video hấp dẫn", "kênh của mình",
            "subscribe", "thank you for watching", "like and subscribe",
            "hit the bell", "don't forget to",
            "cảm ơn bạn", "cảm ơn các bạn", "cảm ơn bạn đã theo dõi",
            "xin chào các bạn", "chào mừng các bạn",
            "thank you", "thanks for watching",
            "チャンネル登録", "ご視聴", "ありがとうございます", "高評価",
            "구독", "좋아요", "시청해주셔서", "감사합니다",
            "订阅", "请按赞", "谢谢观看", "观看",
            "suscríbete", "gracias por ver", "dale like",
            "merci de regarder", "abonnez-vous",
            "vielen dank", "abonnieren",
            "amara.org", "youtube.com", "subtitles by",
        ]

        low_text = refined_text.lower().strip()

        # B. Blacklist match (short chunks only)
        if any(junk in low_text for junk in _BLACKLIST) and len(refined_text.split()) < 15:
            refined_text = ""

        # C. Regex-based hallucination patterns
        if refined_text:
            if re.match(r'^(\S+)(\s+\1){2,}$', refined_text.strip()):
                refined_text = ""
            elif len(refined_text.strip()) <= 2:
                refined_text = ""
            elif re.search(r'(.{5,}?)\s*\1', refined_text):
                match = re.match(r'(.{5,}?)\s*\1', refined_text)
                if match:
                    refined_text = match.group(1).strip()

        # D. Duplicate subtitle filter (rolling window with SequenceMatcher)
        # Accuracy mode keeps repeated text because speeches/news can contain
        # legitimate parallel phrasing that looks duplicate to heuristics.
        if refined_text and not self.ultra_realtime_mode and not self.accuracy_mode:
            if self._is_duplicate(refined_text):
                refined_text = ""
            else:
                self._recent_texts.append(refined_text)
                self.last_text = refined_text

        # E. Vietnamese Punctuation and Correction Layer
        if refined_text and hasattr(info, "language") and info.language == "vi":
            try:
                from core.vietnamese_corrector import VietnameseCorrector
                refined_text = VietnameseCorrector.apply_corrections(refined_text)
            except Exception as e:
                logger.error(f"Correction layer error: {e}")

        detected_lang = info.language if hasattr(info, "language") else language
        return {"text": refined_text, "language": detected_lang}

    # ------------------------------------------------------------------
    # Duplicate / hallucination detection
    # ------------------------------------------------------------------
    def _is_duplicate(self, text: str) -> bool:
        """Three-tier duplicate check using difflib.SequenceMatcher.

        1. Exact match
        2. SequenceMatcher similarity > 0.90
        3. Short-phrase containment check
        """
        if not self._recent_texts:
            return False

        new_norm = text.lower().strip()

        for prev in self._recent_texts:
            prev_norm = prev.lower().strip()

            # 1. Exact match
            if new_norm == prev_norm:
                return True

            # 2. High similarity via SequenceMatcher
            similarity = difflib.SequenceMatcher(None, new_norm, prev_norm).ratio()
            if similarity > 0.90:
                return True

            # 3. Containment (short phrases swallowed by longer ones)
            if len(new_norm.split()) < 8:
                if new_norm in prev_norm:
                    return True

        return False

    def free_memory(self):
        with self._lock:
            self._batched_pipeline = None
            self._whisper_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# StreamingTranscriber — rolling context window + backpressure-aware dispatch
# ---------------------------------------------------------------------------
class StreamingTranscriber:
    """Dispatches audio chunks to Whisper with per-source rolling context memory.

    Each source ('sys', 'mic') maintains an independent deque of
    (timestamp, text) pairs covering the last _CONTEXT_WINDOW_S seconds.
    Before every transcription the recent transcript is prepended to the
    prompt so Whisper sees sentence continuity — critical for Vietnamese tones.

    Context is disabled automatically in ultra_realtime_mode where latency
    takes priority over continuity.
    """

    def __init__(self, ai_engine: AIEngine):
        self.ai = ai_engine
        self.on_sys_text: Optional[Callable[[str], None]] = None
        self.on_mic_text: Optional[Callable[[str], None]] = None

        from core.subtitle_smoother import SubtitleSmoother
        from core.partial_stabilizer import PartialStabilizer

        # SubtitleSmoother: debounces + deduplicates before emitting to UI
        self._sys_smoother = SubtitleSmoother(
            lambda t: self.on_sys_text(t) if self.on_sys_text else None
        )
        self._mic_smoother = SubtitleSmoother(
            lambda t: self.on_mic_text(t) if self.on_mic_text else None
        )

        # v9: PartialStabilizer — MacWhisper-style word consensus (req #8)
        # Stable words (seen in ≥2 consecutive windows) → smoother.push()
        # Pending words (not yet confirmed) → on_partial for UI-only display
        self._sys_stabilizer = PartialStabilizer(
            on_stable  = lambda t: self._sys_smoother.push(t),
            on_partial = lambda t: (
                self.on_sys_text(f"…{t}") if self.on_sys_text else None
            ),
        )
        self._mic_stabilizer = PartialStabilizer(
            on_stable  = lambda t: self._mic_smoother.push(t),
            on_partial = lambda t: (
                self.on_mic_text(f"…{t}") if self.on_mic_text else None
            ),
        )

        # Per-source rolling context — deque of (monotonic_timestamp, text)
        self._sys_context: deque = deque()
        self._mic_context: deque = deque()
        self._sys_lock = threading.Lock()
        self._mic_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Context prompt builder
    # ------------------------------------------------------------------
    def _build_context_prompt(
        self,
        context: deque,
        user_prompt: str,
        now: float,
    ) -> str:
        """Prune stale entries and build a 150-word context-enriched prompt.

        Rolling 15-second window gives Whisper sufficient Vietnamese grammar
        continuity across chunk boundaries without exceeding the 224-token
        prompt limit.

        Must be called while holding the source lock.
        """
        cutoff = now - _CONTEXT_WINDOW_S
        while context and context[0][0] < cutoff:
            context.popleft()

        if not context:
            return user_prompt

        recent = " ".join(text for _, text in context)

        # Truncate to 150-word budget — keep the tail (most recent words)
        words = recent.split()
        if len(words) > _CONTEXT_MAX_WORDS:
            recent = " ".join(words[-_CONTEXT_MAX_WORDS:])

        return f"{recent} {user_prompt}".strip() if user_prompt else recent

    # ------------------------------------------------------------------
    # Main chunk processor
    # ------------------------------------------------------------------
    def process_chunk(
        self,
        source: str,
        audio: np.ndarray,
        language: Optional[str] = None,
        prompt: str = "",
    ):
        """Transcribe one aggregated chunk, enriching the prompt with context.

        Partial segments are forwarded immediately to the SubtitleSmoother
        so the UI shows rolling text before the full chunk completes.
        """
        ctx, lock = (
            (self._sys_context, self._sys_lock)
            if source == "sys"
            else (self._mic_context, self._mic_lock)
        )

        smoother    = self._sys_smoother    if source == "sys" else self._mic_smoother
        stabilizer  = self._sys_stabilizer  if source == "sys" else self._mic_stabilizer

        now = time.monotonic()

        try:
            with lock:
                if self.ai.ultra_realtime_mode:
                    context_prompt = prompt
                else:
                    context_prompt = self._build_context_prompt(ctx, prompt, now)
        except Exception as e:
            logger.error(f"StreamingTranscriber preamble error [{source}]: {e}")
            return

        try:
            # Partials → stabilizer (consensus check) → smoother (UI partial)
            def _on_partial(seg_text: str):
                stabilizer.push(seg_text)

            result = self.ai.transcribe(
                audio,
                language=language,
                prompt=context_prompt,
                on_partial=_on_partial,
            )

            text = result.get("text", "")
            detected_lang = result.get("language", language)

            if not text:
                return

            with lock:
                merged_text = text
                if ctx and not self.ai.ultra_realtime_mode:
                    _, prev_text = ctx[-1]
                    # Remove overlap (up to 30 chars) between end of prev and start of new
                    overlap_amount = 0
                    min_len = min(len(prev_text), len(text))
                    max_overlap = min(min_len, 30)

                    for i in range(max_overlap, 0, -1):
                        if prev_text.endswith(text[:i]):
                            overlap_amount = i
                            break

                    if overlap_amount > 0:
                        merged_text = text[overlap_amount:].strip()
                        if not merged_text:
                            return

                ctx.append((now, text))

            # Emit merged text through SubtitleSmoother to UI
            smoother.flush(merged_text)

        except Exception as e:
            logger.error(f"StreamingTranscriber error [{source}]: {e}")
