"""
translation_engine.py
=============================================================================
Local AI Translation Engine — Meta NLLB-200

Fully offline translation using facebook/nllb-200-distilled-600M.
No API calls. GPU-accelerated with automatic CPU fallback.

Supported language pairs:
    Vietnamese ↔ English, Japanese, Korean, Chinese, French, German, Spanish
    English ↔ all of the above

Performance targets:
    GPU (float16):  < 50 ms per short sentence
    CPU (float32):  < 100 ms per short sentence
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
# Maps human-readable names → NLLB flores-200 codes.
# Full list: https://github.com/facebookresearch/flores/blob/main/flores200/README.md
NLLB_LANG_MAP: Dict[str, str] = {
    # Primary languages
    "english":    "eng_Latn",
    "vietnamese": "vie_Latn",
    # East Asian
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

# Also accept NLLB codes directly (pass-through)
_VALID_NLLB_CODES = set(NLLB_LANG_MAP.values())

MODEL_ID = "facebook/nllb-200-distilled-600M"


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

    # Already a valid NLLB code
    if lang in _VALID_NLLB_CODES:
        return lang

    # Human-readable lookup (case-insensitive)
    key = lang.lower().strip()
    if key in NLLB_LANG_MAP:
        return NLLB_LANG_MAP[key]

    # Whisper ISO-639-1 shortcodes
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
    - Greedy decoding (num_beams=1) for < 100ms latency
    - Supports single string and batch (list) translation
    """

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model_id = model_id
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

        # Hardware detection
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
                return  # Double-checked locking

            logger.info(f"Loading NLLB model: {self.model_id} on {self._device.upper()}...")
            start = time.perf_counter()

            # Suppress noisy HuggingFace warnings
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
        """Translate text from src_lang to tgt_lang.

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

        # Skip empty/whitespace-only inputs
        texts = [t for t in texts if t and t.strip()]
        if not texts:
            return "" if is_single else []

        # Resolve language codes
        try:
            src_code = _resolve_lang(src_lang)
            tgt_code = _resolve_lang(tgt_lang)
        except ValueError as e:
            logger.error(f"Language resolution error: {e}")
            return text

        # Skip if source == target
        if src_code == tgt_code:
            return text

        self._ensure_loaded()
        if not self._model or not self._tokenizer:
            return text

        start = time.perf_counter()

        try:
            # Set source language for tokenizer
            self._tokenizer.src_lang = src_code

            # Tokenize
            inputs = self._tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self._device)

            # Get target language token ID
            tgt_lang_id = self._tokenizer.convert_tokens_to_ids(tgt_code)

            # Generate translation — greedy decoding for speed
            with torch.inference_mode():
                gen_tokens = self._model.generate(
                    **inputs,
                    forced_bos_token_id=tgt_lang_id,
                    max_new_tokens=256,
                    num_beams=1,       # Greedy = fastest
                    do_sample=False,
                )

            # Decode
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
# ─────────────────────────────────────────────────────────────
class TranslationEngine:
    """Singleton wrapper providing async and sync translation.

    - Lazy-loads LocalTranslator on first use
    - translate_async() never blocks the caller (speech pipeline)
    - translate() is synchronous for one-shot / offline use
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

        # The single translator instance (lazy-loaded)
        self._translator: Optional[LocalTranslator] = None

        # Async translation infrastructure:
        # - _queue decouples callers from the worker so transcription never blocks.
        # - ThreadPoolExecutor(max_workers=1) serialises model calls (NLLB is not
        #   concurrency-safe) while keeping translation off the main thread.
        self._queue: queue.Queue = queue.Queue()
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

    # ------------------------------------------------------------------
    # Internal async machinery
    # ------------------------------------------------------------------
    def _dispatch_loop(self) -> None:
        """Drain the input queue and submit each job to the executor."""
        while self._running:
            try:
                item = self._queue.get(timeout=0.05)
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
            callback(result)
        except Exception as e:
            logger.error(f"Async translation error: {e}")
            callback(text)  # fall back to original text

    def _get_translator(self) -> LocalTranslator:
        """Get or create the LocalTranslator instance."""
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
        """Non-blocking translation.

        Enqueues *text* for translation. When the result is ready,
        *callback* is invoked from the background executor thread with
        the translated text. The calling thread returns immediately —
        speech transcription is never stalled.
        """
        if not text:
            return
        self._queue.put((text, src_lang, tgt_lang, callback))

    def translate(
        self,
        text: Union[str, List[str]],
        src_lang: str,
        tgt_lang: str,
    ) -> Union[str, List[str]]:
        """Synchronous translation (kept for one-shot / offline use)."""
        translator = self._get_translator()
        return translator.translate(text, src_lang, tgt_lang)

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
        self._queue.put(None)  # unblock dispatcher
        self._executor.shutdown(wait=True)
        self.clear_cache()
        logger.info("TranslationEngine shut down.")
