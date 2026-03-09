"""
translation_engine.py
=============================================================================
Local AI Translation Engine — Meta NLLB-200 (facebook/nllb-200-distilled-600M)

Fully offline translation. GPU-accelerated with automatic CPU fallback.

v7 Dual-Pipeline Architecture Role:
    This module is used by BOTH:
      1. TranslationWorker (Pipeline 2 daemon thread) — primary consumer.
         Calls translate(text, src_lang, tgt_lang) synchronously.
         The worker's polling loop (200 ms) naturally serialises calls.

      2. TranslationEngine.translate_async() — kept for backward compat.
         Routes through the internal ThreadPoolExecutor.

    In v7, the primary path is:
        TranscriptBuffer → TranslationWorker → LocalTranslator.translate()

Supported language pairs:
    Vietnamese ↔ English, Japanese, Korean, Chinese, French, German, Spanish
    English ↔ all of the above

Performance targets (Step 8):
    GPU (float16):  < 50 ms per short sentence
    CPU (float32):  < 150 ms per short sentence
    Translation queue maxsize: 20
=============================================================================
"""

import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Union

import torch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# NLLB-200 Language Code Mapping
# ─────────────────────────────────────────────────────────────
NLLB_LANG_MAP: Dict[str, str] = {
    # Primary languages (Section 7 targets)
    "english":    "eng_Latn",
    "vietnamese": "vie_Latn",
    # East Asian (Section 7 targets)
    "japanese":   "jpn_Jpan",
    "chinese":    "zho_Hans",
    "korean":     "kor_Hang",
    # European
    "french":     "fra_Latn",
    "german":     "deu_Latn",
    "spanish":    "spa_Latn",
    "italian":    "ita_Latn",
    "portuguese": "por_Latn",
    "russian":    "rus_Cyrl",
    # Southeast Asian
    "thai":       "tha_Thai",
    "indonesian": "ind_Latn",
    "malay":      "zsm_Latn",
}

_VALID_NLLB_CODES = set(NLLB_LANG_MAP.values())

MODEL_ID = "facebook/nllb-200-distilled-600M"

# Translation queue config (Step 8: queue size = 20)
_TRANSLATION_QUEUE_MAXSIZE = 20   # Bounded — v7 target: < 5/20 usage
_TRANSLATION_TIMEOUT_S     = 15.0  # Max time for a single translation job


def _resolve_lang(lang: str) -> str:
    """Resolve a language string to an NLLB flores code.

    Accepts:
        - Human-readable: "english", "Vietnamese", "JAPANESE"
        - NLLB code:      "eng_Latn", "vie_Latn"
        - Whisper code:   "en", "vi", "ja"

    Raises ValueError if the language is not recognized.
    """
    if not lang:
        raise ValueError("Language code cannot be empty")

    if lang in _VALID_NLLB_CODES:
        return lang

    key = lang.lower().strip()
    if key in NLLB_LANG_MAP:
        return NLLB_LANG_MAP[key]

    _WHISPER_MAP = {
        "en": "eng_Latn", "vi": "vie_Latn", "ja": "jpn_Jpan",
        "zh": "zho_Hans", "ko": "kor_Hang", "fr": "fra_Latn",
        "de": "deu_Latn", "es": "spa_Latn", "it": "ita_Latn",
        "pt": "por_Latn", "ru": "rus_Cyrl", "th": "tha_Thai",
        "id": "ind_Latn", "ms": "zsm_Latn",
    }
    if key in _WHISPER_MAP:
        return _WHISPER_MAP[key]

    raise ValueError(
        f"Unknown language: {lang!r}. "
        f"Accepted: {sorted(NLLB_LANG_MAP.keys())} or NLLB codes like 'eng_Latn'"
    )


# ─────────────────────────────────────────────────────────────
# LocalTranslator — core NLLB-200 wrapper
# ─────────────────────────────────────────────────────────────
class LocalTranslator:
    """Offline NLLB-200 translator with GPU acceleration.

    - Lazy-loads model on first translate() call
    - float16 on CUDA, float32 on CPU
    - Greedy decoding (num_beams=1) for < 100 ms latency
    - Supports single string and batch (list) translation
    """

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model_id = model_id
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.float16 if self._device == "cuda" else torch.float32

        logger.info(
            f"LocalTranslator created | model={model_id} | "
            f"device={self._device.upper()} | dtype={self._dtype}"
        )

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        """Load model and tokenizer on first use. Thread-safe."""
        if self._model is not None:
            return

        with self._lock:
            if self._model is not None:
                return

            logger.info(f"Loading NLLB model: {self.model_id} on {self._device.upper()}...")
            start = time.perf_counter()

            import warnings
            warnings.filterwarnings("ignore", message=".*not sharded.*")
            warnings.filterwarnings("ignore", message=".*unauthenticated.*")
            for name in ("transformers", "transformers.modeling_utils",
                         "transformers.configuration_utils", "huggingface_hub"):
                logging.getLogger(name).setLevel(logging.ERROR)

            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self._model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_id,
                    torch_dtype=self._dtype,
                    low_cpu_mem_usage=True,
                ).to(self._device)
                self._model.eval()

                load_ms = (time.perf_counter() - start) * 1000
                logger.info(
                    f"NLLB model loaded in {load_ms:.0f} ms | "
                    f"{self._device.upper()} | {self._dtype}"
                )
            except Exception as e:
                logger.error(f"Failed to load NLLB on {self._device}: {e}")
                if self._device == "cuda":
                    logger.warning("Falling back to CPU...")
                    self._device = "cpu"
                    self._dtype = torch.float32
                    try:
                        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
                        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                        self._model = AutoModelForSeq2SeqLM.from_pretrained(
                            self.model_id,
                            low_cpu_mem_usage=True,
                        ).to("cpu")
                        self._model.eval()
                        logger.info("NLLB loaded on CPU (fallback).")
                    except Exception as e2:
                        logger.error(f"CPU fallback also failed: {e2}")
                        raise
                else:
                    raise

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------
    def translate(
        self,
        text: Union[str, List[str]],
        src_lang: str,
        tgt_lang: str,
    ) -> Union[str, List[str]]:
        """Translate complete sentence(s) from src_lang to tgt_lang.

        Section 7: Only complete sentences (from SentenceBuilder) should
        be passed here. Never raw ASR fragments.

        Args:
            text:     A string or list of strings to translate.
            src_lang: Source language (name, NLLB code, or Whisper code).
            tgt_lang: Target language (name, NLLB code, or Whisper code).

        Returns:
            Translated string (or list if input was a list).
            Returns original text on error.
        """
        if not text:
            return text

        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        texts = [t for t in texts if t and t.strip()]
        if not texts:
            return "" if is_single else []

        try:
            src_code = _resolve_lang(src_lang)
            tgt_code = _resolve_lang(tgt_lang)
        except ValueError as e:
            logger.error(f"Language resolution error: {e}")
            return text

        if src_code == tgt_code:
            return text

        self._ensure_loaded()
        if not self._model or not self._tokenizer:
            return text

        start = time.perf_counter()

        try:
            self._tokenizer.src_lang = src_code

            inputs = self._tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self._device)

            tgt_lang_id = self._tokenizer.convert_tokens_to_ids(tgt_code)

            with torch.inference_mode():
                gen_tokens = self._model.generate(
                    **inputs,
                    forced_bos_token_id=tgt_lang_id,
                    max_new_tokens=256,
                    num_beams=1,       # Greedy = fastest
                    do_sample=False,
                )

            outputs = self._tokenizer.batch_decode(
                gen_tokens, skip_special_tokens=True
            )

            latency_ms = (time.perf_counter() - start) * 1000
            logger.debug(
                f"Translation: {src_code}→{tgt_code} | "
                f"{len(texts)} sent | {latency_ms:.0f} ms"
            )

            return outputs[0] if is_single else outputs

        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text if is_single else texts

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------
    def free_memory(self) -> None:
        """Release GPU/CPU memory held by the model."""
        with self._lock:
            self._model = None
            self._tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("LocalTranslator memory freed.")


# ─────────────────────────────────────────────────────────────
# TranslationEngine — Singleton + async dispatcher
#
# Section 7 Implementation:
#   - translation_queue: dedicated bounded queue for completed sentences
#   - Worker runs in separate thread (never blocks ASR pipeline)
#   - Language detection from Whisper result.language (passed by caller)
#   - Supported: Vietnamese, English, Japanese, Korean, Chinese (+ more)
# ─────────────────────────────────────────────────────────────
class TranslationEngine:
    """Singleton wrapper providing async and sync translation.

    v7 Dual-Pipeline Architecture:
    - Primary consumer: TranslationWorker calls translate() (synchronous)
      via TranscriptBuffer polling. This is the recommended path.
    - Legacy: translate_async() enqueues sentences to internal executor.
      Kept for backward compatibility with any direct callers.
    - translation_queue bounded at 20 (Step 8: queue < 5/20 target)
    - Language detection: Whisper result.language forwarded by caller
    """

    _instance: Optional["TranslationEngine"] = None
    _init_lock = threading.Lock()

    def __new__(cls) -> "TranslationEngine":
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = super(TranslationEngine, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._lock = threading.Lock()

        self._translator: Optional[LocalTranslator] = None

        # v8: translation_queue — bounded at 20
        self.translation_queue: queue.Queue = queue.Queue(maxsize=_TRANSLATION_QUEUE_MAXSIZE)

        # Single executor worker — NLLB is not concurrency-safe
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="translator"
        )
        self._running = True
        self._dispatcher = threading.Thread(
            target=self._dispatch_loop,
            name="translation-dispatcher",
            daemon=True,
        )
        self._dispatcher.start()

        # Stats
        self._total_translated = 0
        self._total_dropped = 0

        # v8: PRELOAD MODEL AT STARTUP in background thread
        # This ensures the model is warm before the first sentence arrives.
        # No more blocking on the first translate() call during live capture.
        self._preload_thread = threading.Thread(
            target=self._preload_model_bg,
            name="nllb-preload",
            daemon=True,
        )
        self._preload_thread.start()
        logger.info(
            "[TranslationEngine] Preloading NLLB model in background — "
            "translation will be ready before first audio chunk."
        )

    # ------------------------------------------------------------------
    # Internal async machinery
    # ------------------------------------------------------------------
    def _dispatch_loop(self) -> None:
        """Drain the translation_queue and submit complete sentences to executor.

        Section 7: Only complete sentences arrive here (from SentenceBuilder).
        Queue depth is monitored — if full, oldest item is dropped with warning.
        """
        while self._running:
            try:
                item = self.translation_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if item is None:  # poison pill
                break

            text, src_lang, tgt_lang, callback = item
            self._executor.submit(
                self._run_translation, text, src_lang, tgt_lang, callback
            )

    def _run_translation(
        self,
        text: Union[str, List[str]],
        src_lang: str,
        tgt_lang: str,
        callback: Callable[[Union[str, List[str]]], None],
    ) -> None:
        """Execute translation in the executor thread and fire the callback."""
        try:
            translator = self._get_translator()
            result = translator.translate(text, src_lang, tgt_lang)
            self._total_translated += 1
            callback(result)
        except Exception as e:
            logger.error(f"Async translation error: {e}")
            callback(text)  # fall back to original text

    def _preload_model_bg(self) -> None:
        """Run _ensure_loaded() in a background thread at startup.

        v8: Called from __init__ so the model is warm before any sentence arrives.
        The first translate() call returns immediately instead of blocking 5–30 s.
        """
        try:
            t = time.perf_counter()
            self._get_translator()  # triggers LocalTranslator._ensure_loaded()
            elapsed = (time.perf_counter() - t) * 1000
            logger.info(
                f"[TranslationEngine] NLLB model preloaded in {elapsed:.0f}ms — "
                f"translation pipeline ready."
            )
        except Exception as e:
            logger.error(f"[TranslationEngine] Preload failed: {e}")

    def preload_model(self) -> None:
        """Block until the NLLB model is fully loaded.

        Call this from AppController.__init__ after TranslationEngine() is
        instantiated — ensures model is ready before the first capture starts.
        If the background preload is already done this returns immediately.
        """
        if self._preload_thread.is_alive():
            logger.info(
                "[TranslationEngine] Waiting for background NLLB preload to finish..."
            )
            self._preload_thread.join()
        else:
            logger.debug("[TranslationEngine] NLLB model already loaded.")

    def _get_translator(self) -> LocalTranslator:
        """Get or create the LocalTranslator instance (thread-safe)."""
        if self._translator is None:
            with self._lock:
                if self._translator is None:
                    self._translator = LocalTranslator()
        return self._translator

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def translate_async(
        self,
        text: Union[str, List[str]],
        src_lang: str,
        tgt_lang: str,
        callback: Callable[[Union[str, List[str]]], None],
    ) -> None:
        """Non-blocking translation of a COMPLETE sentence.

        Section 7: Only complete sentences from SentenceBuilder should be
        passed here. Do NOT translate raw ASR fragments.

        Enqueues the sentence. When ready, callback is invoked from the
        background executor thread. The caller returns immediately — the
        ASR transcription pipeline is never stalled by translation.

        If the queue is full (NLLB is backed up), the OLDEST item is
        dropped and a warning is logged.
        """
        if not text:
            return

        depth = self.translation_queue.qsize()
        q_pct = depth / _TRANSLATION_QUEUE_MAXSIZE * 100

        if q_pct >= 80:
            logger.warning(
                f"Translation queue at {q_pct:.0f}% "
                f"({depth}/{_TRANSLATION_QUEUE_MAXSIZE}) — "
                "consider checking NLLB performance."
            )

        try:
            self.translation_queue.put_nowait((text, src_lang, tgt_lang, callback))
        except queue.Full:
            # Drop oldest item to make room (ASR pipeline must not stall)
            try:
                dropped = self.translation_queue.get_nowait()
                self._total_dropped += 1
                logger.warning(
                    f"Translation queue full — dropped oldest sentence: {str(dropped[0])[:50]!r} "
                    f"(total dropped: {self._total_dropped})"
                )
            except queue.Empty:
                pass
            try:
                self.translation_queue.put_nowait((text, src_lang, tgt_lang, callback))
            except queue.Full:
                pass

    def translate(
        self,
        text: Union[str, List[str]],
        src_lang: str,
        tgt_lang: str,
    ) -> Union[str, List[str]]:
        """Synchronous translation (kept for one-shot / offline use)."""
        translator = self._get_translator()
        return translator.translate(text, src_lang, tgt_lang)

    def get_queue_stats(self) -> dict:
        """Return queue depth statistics for monitoring."""
        return {
            "depth": self.translation_queue.qsize(),
            "maxsize": _TRANSLATION_QUEUE_MAXSIZE,
            "pct": self.translation_queue.qsize() / _TRANSLATION_QUEUE_MAXSIZE * 100,
            "total_translated": self._total_translated,
            "total_dropped": self._total_dropped,
        }

    # Backward compatibility aliases
    def load_model(self, model_type: str = "nllb") -> None:
        """Pre-load the translation model. Called by AppController."""
        self._get_translator()._ensure_loaded()

    def clear_cache(self) -> None:
        """Free model memory."""
        with self._lock:
            if self._translator:
                self._translator.free_memory()
                self._translator = None
            logger.info("Translation cache cleared.")

    def shutdown(self) -> None:
        """Stop the background dispatcher and executor gracefully."""
        self._running = False
        self.translation_queue.put(None)  # unblock dispatcher
        self._executor.shutdown(wait=True)
        self.clear_cache()
        logger.info(
            f"TranslationEngine shut down. "
            f"Stats: translated={self._total_translated}, "
            f"dropped={self._total_dropped}"
        )
