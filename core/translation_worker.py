"""
translation_worker.py
=============================================================================
Async Translation Worker — Pipeline 2 (v7.1)

Completely decoupled from Pipeline 1 (realtime STT).

Full pipeline:
    Pipeline 1 (STT thread):
        Whisper → SentenceBuilder → TranscriptBuffer.append()
                                          ↑ file write, non-blocking

    Pipeline 2 (this daemon thread):
        TranslationWorker.run()
            → TranscriptBuffer.read_new_lines()    [poll every 200 ms]
            → should translate? (src != tgt check)
                YES → TranslationEngine.translate()  [NLLB-600M]
                    → VietnameseCorrector (only if tgt == "vi")
                NO  → pass-through (same language)
            → callback(src_text, translated_text, src_lang, tgt_lang)

Key fix (v7.1):
    - src_lang and tgt_lang now compared properly to decide whether to
      translate or pass through.
    - VietnameseCorrector applied ONLY when tgt_lang resolves to Vietnamese.
    - Logging added for: detected language, translation result, latency.
    - update_languages() supports hot-swap during live session.

Performance target:
    Poll interval:          200 ms
    NLLB translation:       ~50-300 ms per sentence (GPU) / ~100-600 ms (CPU)
    Effective total latency: 200–600 ms after sentence completion
=============================================================================
"""

import logging
import threading
import time
from typing import Callable, Optional

from core.vietnamese_corrector import VietnameseCorrector

logger = logging.getLogger(__name__)

# v9 req #3: poll every 50 ms (was 200 ms) — sentences picked up within one cycle
_POLL_INTERVAL_S = 0.05   # 50 ms

# NLLB codes considered Vietnamese — used to gate VietnameseCorrector
_VI_NLLB_CODES = {"vie_Latn", "vi", "vietnamese"}


def _is_vietnamese(lang_code: str) -> bool:
    """Return True when the language resolves to Vietnamese."""
    return lang_code.lower().strip() in _VI_NLLB_CODES


def _same_language(src: str, tgt: str) -> bool:
    """Return True when src and tgt resolve to the same language.

    Handles mixed formats: NLLB codes (eng_Latn), Whisper codes (en),
    and human labels (English).
    """
    _NORMALIZE = {
        # Vietnamese
        "vie_latn": "vi", "vi": "vi", "vietnamese": "vi",
        # English
        "eng_latn": "en", "en": "en", "english": "en",
        # Japanese
        "jpn_jpan": "ja", "ja": "ja", "japanese": "ja",
        # Korean
        "kor_hang": "ko", "ko": "ko", "korean": "ko",
        # Chinese
        "zho_hans": "zh", "zh": "zh", "chinese": "zh",
        # French
        "fra_latn": "fr", "fr": "fr", "french": "fr",
        # German
        "deu_latn": "de", "de": "de", "german": "de",
        # Spanish
        "spa_latn": "es", "es": "es", "spanish": "es",
    }
    return _NORMALIZE.get(src.lower(), src.lower()) == _NORMALIZE.get(tgt.lower(), tgt.lower())


class TranslationWorker(threading.Thread):
    """Daemon thread that polls TranscriptBuffer and drives async translation.

    Full processing chain per sentence:
        1. Read new lines from TranscriptBuffer (every 200 ms)
        2. Decide: same language? → pass-through, skip NLLB.
        3. Translate via TranslationEngine (NLLB-600M), log latency.
        4. Post-process: VietnameseCorrector if tgt == Vietnamese.
        5. Fire callback(src_text, translated_text).

    Args:
        buffer:          TranscriptBuffer instance (shared with STT pipeline).
        translator:      TranslationEngine with translate(text, src_lang, tgt_lang).
        callback:        Called with (src_text, translated_text) for every
                         processed sentence. Runs in this daemon thread — the
                         AppController marshals it to Qt via Signal.
        src_lang:        Source NLLB/Whisper code, e.g. "vie_Latn" or "vi".
        tgt_lang:        Target NLLB code, e.g. "eng_Latn".
        poll_interval_s: Polling frequency in seconds (default 200 ms).

    Usage::

        worker = TranslationWorker(
            buffer=transcript_buffer,
            translator=translation_engine,
            callback=on_translation_ready,
            src_lang="vie_Latn",
            tgt_lang="eng_Latn",
        )
        worker.start()

        # Hot-swap languages at any time:
        worker.update_languages("eng_Latn", "vie_Latn")

        # Stop gracefully (≤ 200 ms):
        worker.stop()
    """

    def __init__(
        self,
        buffer,
        translator,
        callback: Callable[[str, str], None],
        src_lang: str = "vie_Latn",
        tgt_lang: str = "eng_Latn",
        poll_interval_s: float = _POLL_INTERVAL_S,
    ) -> None:
        super().__init__(name="TranslationWorker", daemon=True)

        self._buffer     = buffer
        self._translator = translator
        self._callback   = callback
        self._poll_s     = poll_interval_s

        # Guarded by _lang_lock to allow thread-safe hot-swap
        self._lang_lock  = threading.Lock()
        self._src_lang   = src_lang
        self._tgt_lang   = tgt_lang

        self._running    = threading.Event()
        self._running.set()   # set = running; clear() = stop requested

        # Public stats — read by monitoring/debug code
        self.total_translated  = 0
        self.total_passthrough = 0
        self.total_errors      = 0

        logger.info(
            "TranslationWorker created | "
            f"{src_lang} → {tgt_lang} | poll={poll_interval_s * 1000:.0f}ms"
        )

    # ------------------------------------------------------------------
    # Thread entry point
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Poll TranscriptBuffer every 200 ms. For each new line:
        - translate (or pass-through if same language)
        - apply VietnameseCorrector when output is Vietnamese
        - fire callback
        """
        with self._lang_lock:
            src, tgt = self._src_lang, self._tgt_lang
        logger.info(f"TranslationWorker started | {src} → {tgt}")

        while self._running.is_set():
            try:
                lines = self._buffer.read_new_lines()
                for line in lines:
                    if not self._running.is_set():
                        break
                    self._process_line(line)

            except Exception as e:
                logger.error(f"TranslationWorker poll error: {e}")
                self.total_errors += 1

            # Sleep for poll interval — not a busy loop
            self._running.wait(timeout=self._poll_s)

        logger.info(
            f"TranslationWorker stopped | "
            f"translated={self.total_translated} "
            f"passthrough={self.total_passthrough} "
            f"errors={self.total_errors}"
        )

    def _process_line(self, text: str) -> None:
        """Full pipeline for a single sentence:
            read → [translate] → [vi-correct] → callback
        """
        if not text or not text.strip():
            return

        with self._lang_lock:
            src_lang = self._src_lang
            tgt_lang = self._tgt_lang

        t_start = time.perf_counter()

        try:
            # ── Step 1: Same-language check ───────────────────────────────
            if _same_language(src_lang, tgt_lang):
                # No translation needed — pass text through directly.
                # Still run VietnameseCorrector if output is Vietnamese.
                output = text
                self.total_passthrough += 1
                elapsed_ms = (time.perf_counter() - t_start) * 1000
                logger.debug(
                    f"[TranslationWorker] PASS-THROUGH "
                    f"({src_lang}={tgt_lang}) | {elapsed_ms:.1f}ms | "
                    f"{text[:60]!r}"
                )
            else:
                # ── Step 2: NLLB Translation ──────────────────────────────
                logger.debug(
                    f"[TranslationWorker] Translating | "
                    f"{src_lang} → {tgt_lang} | {text[:60]!r}"
                )

                raw_result = self._translator.translate(
                    text,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                )
                output = raw_result if isinstance(raw_result, str) else str(raw_result)

                elapsed_ms = (time.perf_counter() - t_start) * 1000
                self.total_translated += 1
                logger.info(
                    f"[TranslationWorker] ✓ Translated | "
                    f"{src_lang} → {tgt_lang} | {elapsed_ms:.0f}ms | "
                    f"{text[:40]!r} → {output[:40]!r}"
                )

            # ── Step 3: Vietnamese post-correction ────────────────────────
            # Only run when the OUTPUT language is Vietnamese.
            if _is_vietnamese(tgt_lang) and output:
                corrected = VietnameseCorrector.apply_corrections(output)
                if corrected != output:
                    logger.debug(
                        f"[TranslationWorker] ViCorrector: "
                        f"{output[:40]!r} → {corrected[:40]!r}"
                    )
                output = corrected

            # ── Step 4: UI callback ───────────────────────────────────────
            self._callback(text, output)

        except Exception as e:
            self.total_errors += 1
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            logger.error(
                f"[TranslationWorker] Error after {elapsed_ms:.0f}ms "
                f"({src_lang} → {tgt_lang}): {e} | text={text[:60]!r}"
            )
            # Fallback: emit original so UI doesn't go blank
            try:
                self._callback(text, text)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------
    def update_languages(self, src_lang: str, tgt_lang: str) -> None:
        """Hot-swap source/target languages without restarting the thread.

        Thread-safe — takes effect on the next poll cycle (≤ 200 ms).
        Called when the user changes the target language in the UI.
        """
        with self._lang_lock:
            old_src, old_tgt = self._src_lang, self._tgt_lang
            self._src_lang = src_lang
            self._tgt_lang = tgt_lang

        logger.info(
            f"[TranslationWorker] Language updated: "
            f"{old_src}→{old_tgt}  →  {src_lang}→{tgt_lang}"
        )

    def stop(self) -> None:
        """Signal the worker to stop on the next poll cycle.

        Non-blocking — the thread exits within one poll interval (≤ 200 ms)
        without interrupting an in-progress translation call.
        """
        self._running.clear()
        logger.info("[TranslationWorker] Stop requested.")
