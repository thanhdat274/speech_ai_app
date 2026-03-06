"""
ai_engine.py
=============================================================================
AIEngine Module - Core Faster-Whisper AI Engine
=============================================================================
"""

import logging
import threading
import queue
import time
from collections import deque
from typing import Optional, Dict, Any, List, Callable

import torch
import numpy as np
from faster_whisper import WhisperModel

# Configure module-level logger
logger = logging.getLogger(__name__)


class AIEngine:
    """
    Singleton-based thread-safe AI Engine managing Faster-Whisper.
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
        # Singleton guard — only the first call does real work
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
            # BatchedInferencePipeline — set by load_whisper on CUDA, None on CPU
            self._batched_pipeline = None
            self._batch_size: int = 4

            # --- Hardware state (populated by _detect_hardware) ---
            self.device: str = "cpu"
            self.compute_type: str = "int8"
            self.vram_gb: float = 0.0
            self._detect_hardware()

            # --- Hallucination / duplicate filter state ---
            self.last_text: str = ""
            self.repetition_count: int = 0
            # Rolling window of recent transcriptions for subtitle dedup.
            # Keeps the last 5 results; token-overlap check prevents re-emitting
            # phrases that were already shown in the previous ~10 seconds.
            self._recent_texts: deque = deque(maxlen=5)

            # --- Language config (synced from ConfigManager by AppController) ---
            # force_language=True  → forward UI language code to Whisper
            # force_language=False → always pass None (full auto-detect)
            self.force_language: bool = True
            self.ultra_realtime_mode: bool = False
            self.vi_base_prompt: str = (
                "Đây là hội thoại tiếng Việt tự nhiên.\n"
                "Người nói có thể dùng giọng miền Bắc hoặc miền Nam.\n"
                "Nội dung có thể bao gồm công nghệ, giáo dục, podcast, review sản phẩm."
            )

    # ------------------------------------------------------------------
    # Hardware detection
    # ------------------------------------------------------------------
    def _detect_hardware(self) -> None:
        """Probe CUDA availability and set device + compute_type accordingly."""
        if not torch.cuda.is_available():
            self._fallback_cpu()
            return
        try:
            torch.cuda.init()
            props = torch.cuda.get_device_properties(0)
            self.device = "cuda"
            self.vram_gb = props.total_memory / (1024 ** 3)

            # int8_float16 for ALL CUDA GPUs — weights are int8 (halves VRAM
            # vs float16) while activations stay fp16 for accuracy.
            # This allows large-v3 to fit in ~3 GB VRAM on GTX 1650 class cards.
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
        """Choose the best model for the detected hardware.

        Target model is large-v3 (superior Vietnamese accuracy) when VRAM
        is sufficient for float16.  Smaller cards fall back gracefully:

        GPU tiers:
            < 4 GB  → tiny    (memory-critical)
            4–6 GB  → medium  (float16 large-v3 would OOM; medium is stable)
            ≥ 6 GB  → large-v3 with float16 (main target)

        CPU always gets base (best accuracy/speed without a GPU).
        A ConfigManager override takes highest priority over auto-selection.
        """
        # ConfigManager override — import lazily to avoid circular deps at module load
        try:
            from core.config_manager import ConfigManager
            override = ConfigManager().get("whisper_model_override", "")
            if override and override not in ("", "Auto", "Auto (Recommended)"):
                logger.info(f"Model override from config: '{override}'")
                return override
        except Exception:
            pass  # ConfigManager unavailable — proceed with auto-selection

        if self.device == "cpu":
            selected = "base"
        else:
            # Always use large-v3 on CUDA for maximum Vietnamese accuracy.
            # int8_float16 compute type keeps VRAM usage at ~3 GB,
            # fitting comfortably even on GTX 1650 (4 GB) class cards.
            selected = "large-v3"

        logger.info(
            f"Auto-selected model: '{selected}' "
            f"(device={self.device}, VRAM={self.vram_gb:.1f} GB)"
        )
        return selected

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def load_whisper(self, model_size: str = "base", force_reload: bool = False) -> None:
        """Load (or hot-swap) a Whisper model.

        Passing "Auto (Recommended)" or any empty/auto string triggers
        _auto_select_whisper_model, which respects ConfigManager overrides.
        """
        # Normalise UI strings → auto-select
        if not model_size or any(kw in model_size for kw in ("Auto", "Recommended", "auto")):
            model_size = self._auto_select_whisper_model()

        # large-v3 with int8_float16 fits in ~3 GB VRAM — no downgrade needed.
        # All CUDA models use int8_float16 for optimal speed/accuracy tradeoff.
        effective_compute = self.compute_type

        with self._lock:
            if not force_reload and self._whisper_model and self._whisper_model_name == model_size:
                logger.debug(f"Whisper '{model_size}' already loaded — skipping reload.")
                return

            # Free previous model to reclaim GPU memory before allocating next
            if self._whisper_model:
                logger.info(f"Unloading Whisper '{self._whisper_model_name}' to free memory.")
                self._whisper_model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            logger.info(
                f"Loading Whisper '{model_size}' | "
                f"device={self.device.upper()} | "
                f"compute_type={effective_compute} | "
                f"VRAM={self.vram_gb:.1f} GB"
            )
            try:
                # num_workers controls CTranslate2 inter-op CPU threads used for
                # audio feature extraction and data preparation — more workers
                # overlap preprocessing with GPU kernels for lower latency.
                # 4 workers on GPU, 2 on CPU (CPU already saturated by inference).
                num_workers = 4 if self.device == "cuda" else 2
                self._whisper_model = WhisperModel(
                    model_size,
                    device=self.device,
                    compute_type=effective_compute,

                    # tăng tốc inference
                    cpu_threads=6,
                    num_workers=4,

                    # giảm latency realtime
                    download_root=None,
                    local_files_only=False
                )

                # batching for realtime streaming
                self._batch_size = 4
                self._whisper_model_name = model_size

                # BatchedInferencePipeline: processes the audio spectrogram in
                # parallel mini-batches, hiding GPU kernel launch overhead.
                # For GTX 1650 (4 GB): batch_size=8 is safe; 16 for ≥ 6 GB.
                # Falls back gracefully if the version of faster-whisper is old.
                self._batched_pipeline = None
                if self.device == "cuda":
                    try:
                        from faster_whisper import BatchedInferencePipeline
                        self._batch_size = 4 if self.vram_gb <= 6.0 else 8
                        self._batched_pipeline = BatchedInferencePipeline(
                            model=self._whisper_model
                        )
                        logger.info(
                            f"BatchedInferencePipeline enabled | "
                            f"batch_size={self._batch_size} | VRAM={self.vram_gb:.1f} GB"
                        )
                    except (ImportError, Exception) as e:
                        logger.warning(
                            f"BatchedInferencePipeline unavailable ({e}) — "
                            "using standard inference."
                        )
                        self._batched_pipeline = None

                logger.info(
                    f"Whisper '{model_size}' ready | "
                    f"{self.device.upper()} | {effective_compute}"
                )
            except Exception as e:
                logger.error(f"Failed to load Whisper '{model_size}': {e}")
                if model_size != "tiny":
                    logger.warning("Falling back to 'tiny' model.")
                    self.load_whisper("tiny", force_reload=True)
                else:
                    raise

    def _resolve_language(self, language: Optional[str]) -> Optional[str]:
        """Return the language code to pass to Whisper.

        If language is "vi", we always force it to prevent Whisper from
        incorrectly detecting English or noise as random languages.
        For others, we respect the force_language setting from UI.
        """
        if language == "vi":
            return "vi"
            
        if not self.force_language:
            return None
        return language  # None means auto-detect; any ISO code forces the model

    def _build_prompt(self, language: Optional[str], user_prompt: str) -> str:
        """Build the full initial_prompt for Whisper.

        For Vietnamese (language=="vi") the vi_base_prompt is prepended so
        Whisper is primed with Vietnamese vocabulary and context before seeing
        any user-supplied keywords.  For all other languages the user prompt
        is returned unchanged.
        """
        if language == "vi":
            base = self.vi_base_prompt.strip()
            if user_prompt:
                return f"{base} {user_prompt.strip()}"
            return base
        return user_prompt

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
        """Thread-safe transcription with full audio preprocessing pipeline.

        Args:
            audio:         Raw float32 PCM array (any sample rate / channel count).
            language:      ISO-639-1 language code, or None for auto-detect.
            prompt:        Optional context hint passed to Whisper.
            on_partial:    Callback invoked per decoded segment for streaming UI.
            source_sr:     Sample rate of the input audio (default 16 kHz).
            channels:      Number of channels (hint; auto-detected from shape).
            noise_reduce:  If True, apply spectral-gate noise suppression.
        """
        if not self._whisper_model:
            self.load_whisper()

        # Note: Extensive preprocessing (resampling to 16kHz, mono conversion,
        # volume normalisation, and Silero VAD filtering) is now handled 
        # upstream in ChunkAggregator via audio_preprocessor.py.
        # This function expects `audio` to be a clean 16kHz mono float32 array.

        # Empty array (or pure silence) means skip inference entirely
        if audio is None or len(audio) < 400:
            return {"text": "", "language": language}

        # Apply language config: force or auto-detect, build language-aware prompt
        effective_language = self._resolve_language(language)
        effective_prompt = self._build_prompt(language, prompt)

        if language == "vi":
            logger.debug(f"Vietnamese mode | force={self.force_language} | prompt_len={len(effective_prompt)}")

        is_vietnamese = (effective_language == "vi")

        if is_vietnamese:
            # ── Vietnamese accuracy-first decoding ─────────────────────────
            #
            # beam_size=5              5 candidate paths — critical for tone
            #                          disambiguation (mà/má/mã/mả/ma/mạ)
            # best_of=5               keep best sample per beam
            # temperature=0           greedy — stable, no sampling noise
            # word_timestamps=True    fine-grained alignment for subtitles
            # condition_on_previous_text=True
            #                          carry tonal context across segments
            # repetition_penalty=1.1  suppresses word-jumping/loops
            # vad_filter=False        disabled — Vietnamese tones confuse
            #                          Silero VAD; let Whisper handle it
            #                          internally via no_speech_threshold
            #
            # Phrase Biasing for Tech/IT vocabulary
            tech_hotwords = (
                "YouTube, ChatGPT, GitHub, Python, AI, "
                "machine learning, công nghệ, lập trình"
            )

            _transcribe_kwargs = dict(
                beam_size=5,
                best_of=5,
                temperature=0,
                language=effective_language,
                initial_prompt=effective_prompt,
                hotwords=tech_hotwords,
                word_timestamps=True,
                condition_on_previous_text=True,
                repetition_penalty=1.1,
                vad_filter=False,
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
                compression_ratio_threshold=2.4,
            )
        else:
            # Speed-first greedy decoding for all other languages
            _transcribe_kwargs = dict(
                beam_size=1,
                best_of=1,
                temperature=0,
                language=effective_language,
                initial_prompt=effective_prompt,
                condition_on_previous_text=False,
                vad_filter=False,
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
                compression_ratio_threshold=2.4,
            )
        import re

        try:
            with self._lock:
            if self._batched_pipeline is not None:
                segments, info = self._batched_pipeline.transcribe(
                    audio,
                    batch_size=self._batch_size,
                    **_transcribe_kwargs,
                )
            else:
                segments, info = self._whisper_model.transcribe(
                    audio,
                    **_transcribe_kwargs,
                )

            raw_text_parts: List[str] = []

            for seg in segments:
                seg_text = seg.text.strip()

                # ── Per-segment hallucination filters ─────────────────
                # These run on EVERY segment before it reaches the UI.

                # Filter 1: Empty or too short (< 2 chars)
                if len(seg_text) < 2:
                    logger.debug(f"Segment rejected (too short): {seg_text!r}")
                    continue

                # Filter 2: Low confidence — avg_logprob < -1.0 means
                # Whisper is guessing, typical for hallucinated filler
                if hasattr(seg, 'avg_logprob') and seg.avg_logprob < -1.0:
                    logger.debug(
                        f"Segment rejected (low confidence): "
                        f"avg_logprob={seg.avg_logprob:.2f} | {seg_text!r}"
                    )
                    continue

                # Filter 3: High no-speech probability — segment is likely
                # silence/noise that Whisper tried to transcribe anyway
                if hasattr(seg, 'no_speech_prob') and seg.no_speech_prob > 0.5:
                    logger.debug(
                        f"Segment rejected (no_speech): "
                        f"no_speech_prob={seg.no_speech_prob:.2f} | {seg_text!r}"
                    )
                    continue

                # Filter 4: Repeated words — "ạ ạ ạ", "vâng vâng vâng"
                words = seg_text.lower().split()
                if len(words) >= 2:
                    unique_words = set(words)
                    if len(unique_words) == 1:
                        # Every word is the same → hallucination
                        logger.debug(f"Segment rejected (all repeated): {seg_text!r}")
                        continue
                    # More than 70% of words are the same word → likely hallucination
                    most_common_count = max(words.count(w) for w in unique_words)
                    if most_common_count / len(words) > 0.70 and len(words) >= 3:
                        logger.debug(f"Segment rejected (word repetition): {seg_text!r}")
                        continue

                # Filter 5: Compression ratio — highly repetitive output
                if hasattr(seg, 'compression_ratio') and seg.compression_ratio > 2.4:
                    logger.debug(
                        f"Segment rejected (high compression): "
                        f"ratio={seg.compression_ratio:.2f} | {seg_text!r}"
                    )
                    continue

                # ── Segment passed all filters → emit ─────────────────
                if on_partial:
                    try:
                        on_partial(seg_text)
                    except Exception:
                        pass

                raw_text_parts.append(seg_text)

            raw_text = " ".join(raw_text_parts).strip()
            except RuntimeError as e:
            logger.error(f"CUDA runtime error: {e}")
            torch.cuda.empty_cache()
            return {"text": "", "language": language}
        # ----------------------------------------------------------------
        # Post-processing quality filter (operates on merged text)
        # ----------------------------------------------------------------
        refined_text = raw_text

        # A. Blacklist — YouTube spam + common Whisper hallucinations
        _BLACKLIST = [
            # YouTube channel spam (Vietnamese)
            "ghiền mì gõ", "la la school", "cho kênh", "để không bỏ",
            "hẹn gặp lại", "đăng ký kênh", "ủng hộ kênh", "theo dõi kênh",
            "nhấn like", "nhấn subscribe", "bấm chuông", "chia sẻ video",
            "video hấp dẫn", "kênh của mình",
            # YouTube spam (English)
            "subscribe", "thank you for watching", "like and subscribe",
            "hit the bell", "don't forget to",
            # Common Whisper hallucination patterns
            "cảm ơn bạn", "cảm ơn các bạn", "cảm ơn bạn đã theo dõi",
            "xin chào các bạn", "chào mừng các bạn",
            "thank you", "thanks for watching",
        ]

        low_text = refined_text.lower().strip()

        # B. Blacklist match (short chunks only — long genuine text may contain these)
        if any(junk in low_text for junk in _BLACKLIST) and len(refined_text.split()) < 15:
            refined_text = ""

        # C. Regex-based hallucination patterns on merged text
        if refined_text:
            # Repeated single character/word (e.g. "ạ ạ ạ ạ")
            if re.match(r'^(\S+)(\s+\1){2,}$', refined_text.strip()):
                refined_text = ""
            # Single character or very short nonsense
            elif len(refined_text.strip()) <= 2:
                refined_text = ""
            # Repeated phrases: "ABC ABC" → extract unique part
            elif re.search(r'(.{5,}?)\s*\1', refined_text):
                match = re.match(r'(.{5,}?)\s*\1', refined_text)
                if match:
                    refined_text = match.group(1).strip()

        # D. Duplicate subtitle filter (rolling window)
        if refined_text and not self.ultra_realtime_mode:
            if self._is_duplicate(refined_text):
                refined_text = ""
            else:
                self._recent_texts.append(refined_text)
                self.last_text = refined_text

        # E. Vietnamese Punctuation and Correction Layer
        if refined_text and info.language == "vi":
            try:
                from core.vietnamese_corrector import VietnameseCorrector
                refined_text = VietnameseCorrector.apply_corrections(refined_text)
            except Exception as e:
                logger.error(f"Correction layer error: {e}")

        return {"text": refined_text, "language": info.language}



    def _is_duplicate(self, text: str) -> bool:
        """Return True when *text* is substantially similar to a recent result.

        Three-tier check (in order of cost):
          1. Exact match       — fastest, catches perfect repeats
          2. Containment       — catches short phrases swallowed by longer ones
          3. Token-overlap     — catches paraphrased / partially-shifted repetitions
                                 (overlap > 85 % of the new text's tokens)

        The rolling window (_recent_texts, maxlen=5) covers roughly the last
        10–15 seconds of speech, matching typical subtitle display duration.
        """
        if not self._recent_texts:
            return False
            
        import difflib
        new_norm = text.lower().strip()
        
        for prev in self._recent_texts:
            prev_norm = prev.lower().strip()
            
            # 1. Exact match (O(1))
            if new_norm == prev_norm:
                return True
                
            # 2. Similarity match (O(N^2) but short strings)
            # Suppress if strings are more than 90% similar
            similarity = difflib.SequenceMatcher(None, new_norm, prev_norm).ratio()
            if similarity > 0.90:
                return True
                
            # 3. Containment (catches short phrases swallowed by longer ones)
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


# Rolling context window parameters for StreamingTranscriber.
# Whisper's initial_prompt limit is 224 tokens.  vi_base_prompt consumes ~100,
# leaving ~124 tokens (≈ 150 Vietnamese words) for rolling context.
_CONTEXT_WINDOW_S  = 15.0   # YouTube Mode: Longer context of 15 seconds for robust grammar
_CONTEXT_MAX_WORDS = 150    # Word budget for context within the prompt limit


class StreamingTranscriber:
    """Dispatches audio chunks to Whisper with per-source rolling context memory.

    Each source ('sys', 'mic') maintains an independent deque of
    (timestamp, text) pairs covering the last _CONTEXT_WINDOW_S seconds.
    Before every transcription call the recent transcript is prepended to the
    prompt so Whisper sees sentence continuity — critical for Vietnamese tones
    and cross-chunk word boundaries.

    Context is disabled automatically in ultra_realtime_mode where latency
    takes priority over continuity.
    """

    def __init__(self, ai_engine: AIEngine):
        self.ai = ai_engine
        self.on_sys_text: Optional[Callable[[str], None]] = None
        self.on_mic_text: Optional[Callable[[str], None]] = None

        from core.subtitle_smoother import SubtitleSmoother
        
        # Smoothers delay UI output to combine short fragments and avoid blinking
        self._sys_smoother = SubtitleSmoother(lambda t: self.on_sys_text(t) if self.on_sys_text else None)
        self._mic_smoother = SubtitleSmoother(lambda t: self.on_mic_text(t) if self.on_mic_text else None)

        # Per-source rolling context — deque of (monotonic_timestamp, text)
        # Separate locks because sys and mic workers run on independent threads.
        self._sys_context: deque = deque()
        self._mic_context: deque = deque()
        self._sys_lock = threading.Lock()
        self._mic_lock = threading.Lock()

    def _build_context_prompt(
        self,
        context: deque,
        user_prompt: str,
        now: float,
    ) -> str:
        """Prune stale entries and build a context-enriched prompt string.

        The returned string is passed as ``prompt`` to AIEngine.transcribe(),
        where AIEngine._build_prompt() will prepend vi_base_prompt on top.
        Total prompt order fed to Whisper:
            vi_base_prompt  ·  recent_context  ·  user_keywords

        Must be called while holding the source lock (context is mutated).
        """
        # Drop entries outside the rolling window
        cutoff = now - _CONTEXT_WINDOW_S
        while context and context[0][0] < cutoff:
            context.popleft()

        if not context:
            return user_prompt

        # Concatenate retained transcripts, newest last so Whisper reads forward
        recent = " ".join(text for _, text in context)

        # Truncate to word budget — keep the tail (most recent words)
        words = recent.split()
        if len(words) > _CONTEXT_MAX_WORDS:
            recent = " ".join(words[-_CONTEXT_MAX_WORDS:])

        return f"{recent} {user_prompt}".strip() if user_prompt else recent

    def process_chunk(
        self,
        source: str,
        audio: np.ndarray,
        language: Optional[str] = None,
        prompt: str = "",
    ):
        """Transcribe one aggregated chunk, enriching the prompt with context.

        Partial segments are forwarded immediately so the UI shows rolling text
        before the full chunk completes.
        """
        ctx, lock = (
            (self._sys_context, self._sys_lock)
            if source == "sys"
            else (self._mic_context, self._mic_lock)
        )
        
        smoother = self._sys_smoother if source == "sys" else self._mic_smoother

        now = time.monotonic()

        # Build context prompt under lock so no concurrent writer corrupts the deque
        try:
            with lock:
                # Ultra realtime mode skips context — latency > continuity
                if self.ai.ultra_realtime_mode:
                    context_prompt = prompt
                else:
                    context_prompt = self._build_context_prompt(ctx, prompt, now)
        except Exception as e:
            logger.error(f"StreamingTranscriber preamble error [{source}]: {e}")
            return

        try:
            def _on_partial(seg_text: str):
                smoother.push(seg_text)

            result = self.ai.transcribe(
                audio,
                language=language,
                prompt=context_prompt,
                on_partial=_on_partial,
            )

            text = result.get("text", "")
            if not text:
                return

            # --- Merge partial sentences across chunks ---
            # To handle the 200ms chunk overlap cleanly, we check if the start of
            # the *new* text overlaps with the end of the *previous* text in the context window.
            with lock:
                merged_text = text
                if ctx and not self.ai.ultra_realtime_mode:
                    _, prev_text = ctx[-1]
                    # Look for overlap (up to 30 characters) between end of prev and start of new
                    overlap_amount = 0
                    min_len = min(len(prev_text), len(text))
                    max_overlap = min(min_len, 30)
                    
                    for i in range(max_overlap, 0, -1):
                        if prev_text.endswith(text[:i]):
                            overlap_amount = i
                            break
                    
                    if overlap_amount > 0:
                        merged_text = text[overlap_amount:].strip()
                        # If the result is empty after removing the overlap, it's a complete duplicate
                        if not merged_text:
                            return

                ctx.append((now, text)) # Append the full text to context for future Whisper prompts

            # Emit the *merged* text to the UI to avoid duplicate words via SubtitleSmoother
            smoother.flush(merged_text)

        except Exception as e:
            logger.error(f"StreamingTranscriber error [{source}]: {e}")
