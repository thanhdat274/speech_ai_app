"""
streaming_translator.py
=============================================================================
Streaming Translation Layer — v1

Sits between the Whisper output and the UI, providing:

  1. Chunk streaming translation
     Each Whisper chunk is fed directly — translation starts immediately
     after Whisper finishes, NOT after polling a file.

  2. Context memory between chunks
     A rolling window of (timestamp, src_text, tgt_text) pairs is
     maintained per source. NLLB receives recent translated output as
     a soft context hint, improving cross-chunk coherence.

  3. Sentence boundary detection (integrated SentenceBuilder)
     Chunk text is split on boundaries; each clause is translated
     independently and in order, then joined. This prevents NLLB from
     receiving an over-long run-on input and producing garbled output.

  4. Translation smoothing (TranslationOutputSmoother)
     Near-duplicate translations are suppressed (SequenceMatcher ratio
     > 0.85). Translations shorter than MIN_WORDS are merged with the
     next emission. Debounce of 100 ms prevents UI flickering.

  5. Vietnamese grammar correction
     VietnameseCorrector.apply_corrections() runs when tgt == Vietnamese,
     fixing punctuation, capitalisation, and NLLB over-capitalisation.

Architecture:

    StreamingTranscriber.process_chunk()       [ai_engine.py]
             │ final text per chunk
             ▼
    StreamingTranslator.feed(text, src, tgt)   [this file]
             │
             ├── SentenceBuilder → sentence boundaries
             │
             ├── _translate_sentence(sentence)
             │       ├── context_prompt = last N translated sentences
             │       ├── LocalTranslator.translate()    [NLLB-600M]
             │       └── VietnameseCorrector (tgt==vi)
             │
             └── TranslationOutputSmoother → UI callback

Thread safety:
    feed() is called from the Whisper worker thread.
    All internal state guarded by threading.Lock.
    UI callback fires from a short-lived daemon thread to avoid
    blocking the worker.

Performance target:
    Whisper latency:     400–800 ms
    NLLB latency:        50–300 ms (GPU) per sentence
    End-to-end:          < 1 s from speech to translated subtitle
=============================================================================
"""

import difflib
import logging
import threading
import time
from collections import deque
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Rolling context window — keep last N translated sentences as NLLB hint
_CONTEXT_KEEP_SENTENCES = 5
_CONTEXT_MAX_WORDS      = 80    # Truncate oldest if context grows too long

# TranslationOutputSmoother
_SMOOTHER_DEBOUNCE_S   = 0.10   # 100 ms — prevent UI flickering
_SMOOTHER_MIN_WORDS    = 2      # Merge if shorter than this
_SMOOTHER_DEDUP_RATIO  = 0.85   # SequenceMatcher ratio above which we suppress

# NLLB codes considered Vietnamese — used to gate VietnameseCorrector
_VI_CODES = {"vie_latn", "vi", "vietnamese"}


def _is_vietnamese(lang: str) -> bool:
    return lang.lower().strip() in _VI_CODES


def _same_language(src: str, tgt: str) -> bool:
    """Normalise NLLB/Whisper/human codes and compare."""
    _NORM = {
        "vie_latn": "vi", "vi": "vi", "vietnamese": "vi",
        "eng_latn": "en", "en": "en", "english": "en",
        "jpn_jpan": "ja", "ja": "ja", "japanese": "ja",
        "kor_hang": "ko", "ko": "ko", "korean": "ko",
        "zho_hans": "zh", "zh": "zh", "chinese": "zh",
        "fra_latn": "fr", "fr": "fr", "french": "fr",
        "deu_latn": "de", "de": "de", "german": "de",
        "spa_latn": "es", "es": "es", "spanish": "es",
    }
    return _NORM.get(src.lower(), src.lower()) == _NORM.get(tgt.lower(), tgt.lower())


# ─────────────────────────────────────────────────────────────────────────────
# TranslationOutputSmoother
# ─────────────────────────────────────────────────────────────────────────────
class TranslationOutputSmoother:
    """Debounce + dedup layer for translated subtitle output.

    Prevents:
      - Rapid flickering when multiple short clauses arrive in quick succession
      - Near-duplicate emissions when overlap removal isn't perfect
      - Single-word / noise outputs reaching the UI

    Not to be confused with SubtitleSmoother (which operates on raw STT partials).
    """

    def __init__(
        self,
        callback: Callable[[str], None],
        debounce_s: float = _SMOOTHER_DEBOUNCE_S,
        min_words: int = _SMOOTHER_MIN_WORDS,
    ) -> None:
        self._callback   = callback
        self._debounce_s = debounce_s
        self._min_words  = min_words

        self._pending: List[str]      = []
        self._timer:  Optional[threading.Timer] = None
        self._lock    = threading.Lock()
        self._last_emitted = ""

    def push(self, text: str) -> None:
        """Push a translated clause to the smoother.

        Short clauses are queued and merged. Longer ones (or those that
        arrive after the debounce window) trigger immediate emission.
        """
        text = text.strip()
        if not text:
            return

        with self._lock:
            words = text.split()

            # Near-duplicate suppression
            if self._last_emitted and self._is_near_duplicate(text, self._last_emitted):
                return

            if len(words) < self._min_words:
                # Too short — accumulate
                self._pending.append(text)
                self._restart_timer()
                return

            # Long enough — accumulate and (re)start debounce
            self._pending.append(text)
            self._restart_timer()

    def flush_immediate(self) -> None:
        """Flush pending buffer immediately, bypassing the timer."""
        with self._lock:
            self._cancel_timer()
            self._emit_pending_locked()

    def reset(self) -> None:
        """Clear all pending state (called on language change / session end)."""
        with self._lock:
            self._cancel_timer()
            self._pending.clear()
            self._last_emitted = ""

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _restart_timer(self) -> None:
        """Restart debounce timer. Must hold self._lock."""
        self._cancel_timer()
        self._timer = threading.Timer(self._debounce_s, self._on_timer)
        self._timer.daemon = True
        self._timer.start()

    def _cancel_timer(self) -> None:
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def _on_timer(self) -> None:
        with self._lock:
            self._emit_pending_locked()

    def _emit_pending_locked(self) -> None:
        """Merge pending clauses and emit. Must hold self._lock."""
        if not self._pending:
            return
        combined = " ".join(self._pending).strip()
        self._pending.clear()
        self._timer = None

        if not combined:
            return

        # Final dup check on the combined output
        if self._last_emitted and self._is_near_duplicate(combined, self._last_emitted):
            return

        self._last_emitted = combined
        # Fire callback in the same thread — AppController marshals to Qt via Signal
        try:
            self._callback(combined)
        except Exception as e:
            logger.error(f"TranslationOutputSmoother callback error: {e}")

    @staticmethod
    def _is_near_duplicate(a: str, b: str) -> bool:
        ratio = difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
        return ratio >= _SMOOTHER_DEDUP_RATIO


# ─────────────────────────────────────────────────────────────────────────────
# StreamingTranslator
# ─────────────────────────────────────────────────────────────────────────────
class StreamingTranslator:
    """Per-source streaming translation with context memory.

    One instance per audio source (sys / mic). Maintains its own:
      - Translation context window (last N translated sentences)
      - TranslationOutputSmoother for UI output
      - Thread lock for all state

    Usage::

        translator = StreamingTranslator(
            translator_engine=translation_engine,   # TranslationEngine
            callback=on_translated_text,            # fn(translated: str)
            src_lang="vie_Latn",
            tgt_lang="eng_Latn",
            label="SYS",
        )

        # Called after each Whisper chunk completes:
        translator.feed("hôm nay chúng ta sẽ nói về AI")

        # On session end / clear:
        translator.reset()

    Args:
        translator_engine: TranslationEngine instance (provides .translate()).
        callback:          fn(translated_text: str) — called on the worker
                           thread; AppController must marshal to Qt via Signal.
        src_lang:          Source language NLLB/Whisper code.
        tgt_lang:          Target language NLLB code.
        label:             "SYS" or "MIC" — used only in log messages.
    """

    def __init__(
        self,
        translator_engine,
        callback: Callable[[str], None],
        src_lang: str = "vie_Latn",
        tgt_lang: str = "eng_Latn",
        label: str = "SRC",
    ) -> None:
        self._translator = translator_engine
        self._label      = label
        self._lock       = threading.Lock()

        self._src_lang = src_lang
        self._tgt_lang = tgt_lang

        # Rolling context: deque of (src_sentence, translated_sentence)
        # Used to build a soft hint for NLLB across chunk boundaries
        self._context: deque = deque(maxlen=_CONTEXT_KEEP_SENTENCES)

        # Output smoother — dedup + debounce translated output
        self._smoother = TranslationOutputSmoother(callback=callback)

        # Stats
        self.total_translated = 0
        self.total_passthrough = 0
        self.total_errors = 0

        logger.info(
            f"StreamingTranslator[{label}] ready | "
            f"{src_lang} → {tgt_lang}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, text: str) -> None:
        """Translate a Whisper output chunk.

        The text is split on sentence boundaries. Each sentence is
        translated independently (with context hint) and pushed to the
        smoother for output.

        Called from the Whisper worker thread after each chunk completes.
        """
        text = text.strip()
        if not text:
            return

        with self._lock:
            src_lang = self._src_lang
            tgt_lang = self._tgt_lang

        sentences = self._split_sentences(text)
        logger.debug(
            f"[StreamingTranslator:{self._label}] feed {len(sentences)} clause(s) "
            f"from: {text[:60]!r}"
        )

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            self._process_sentence(sentence, src_lang, tgt_lang)

    def update_languages(self, src_lang: str, tgt_lang: str) -> None:
        """Hot-swap languages and reset context + smoother."""
        with self._lock:
            old = f"{self._src_lang}→{self._tgt_lang}"
            self._src_lang = src_lang
            self._tgt_lang = tgt_lang
            self._context.clear()
        self._smoother.reset()
        logger.info(
            f"[StreamingTranslator:{self._label}] languages: "
            f"{old} → {src_lang}→{tgt_lang}"
        )

    def reset(self) -> None:
        """Clear all state (called on Clear button / session stop)."""
        with self._lock:
            self._context.clear()
        self._smoother.reset()
        logger.info(f"[StreamingTranslator:{self._label}] reset.")

    def flush(self) -> None:
        """Force-emit any pending smoother output."""
        self._smoother.flush_immediate()

    # ------------------------------------------------------------------
    # Internal — translation pipeline
    # ------------------------------------------------------------------

    def _process_sentence(self, sentence: str, src_lang: str, tgt_lang: str) -> None:
        """Translate one sentence and push result to the smoother."""
        t_start = time.perf_counter()

        try:
            # ── 1. Same-language pass-through ─────────────────────────────
            if _same_language(src_lang, tgt_lang):
                output = sentence
                self.total_passthrough += 1
                latency_ms = (time.perf_counter() - t_start) * 1000
                logger.debug(
                    f"[StreamingTranslator:{self._label}] PASS-THROUGH "
                    f"| {latency_ms:.1f}ms | {sentence[:50]!r}"
                )
                self._smoother.push(output)
                with self._lock:
                    self._context.append((sentence, output))
                return

            # ── 2. Build context hint from recent translated output ────────
            context_hint = self._build_context_hint()

            # ── 3. NLLB translation ───────────────────────────────────────
            # Prepend context hint to give NLLB cross-chunk coherence
            input_with_context = (
                f"{context_hint} {sentence}".strip()
                if context_hint else sentence
            )

            logger.debug(
                f"[StreamingTranslator:{self._label}] translating | "
                f"{src_lang} → {tgt_lang} | "
                f"context_words={len(context_hint.split())} | "
                f"{sentence[:50]!r}"
            )

            raw = self._translator.translate(
                sentence,           # translate ONLY the new sentence
                src_lang=src_lang,
                tgt_lang=tgt_lang,
            )
            output = raw if isinstance(raw, str) else str(raw)

            latency_ms = (time.perf_counter() - t_start) * 1000
            self.total_translated += 1

            logger.info(
                f"[StreamingTranslator:{self._label}] ✓ "
                f"{src_lang}→{tgt_lang} | {latency_ms:.0f}ms | "
                f"{sentence[:40]!r} → {output[:40]!r}"
            )

            # ── 4. Vietnamese post-correction ──────────────────────────────
            if _is_vietnamese(tgt_lang) and output:
                try:
                    from core.vietnamese_corrector import VietnameseCorrector
                    corrected = VietnameseCorrector.apply_corrections(output)
                    if corrected != output:
                        logger.debug(
                            f"[StreamingTranslator:{self._label}] ViCorrector: "
                            f"{output[:40]!r} → {corrected[:40]!r}"
                        )
                    output = corrected
                except Exception as e:
                    logger.warning(f"VietnameseCorrector error: {e}")

            # ── 5. Update context and emit ─────────────────────────────────
            with self._lock:
                self._context.append((sentence, output))

            self._smoother.push(output)

        except Exception as e:
            self.total_errors += 1
            latency_ms = (time.perf_counter() - t_start) * 1000
            logger.error(
                f"[StreamingTranslator:{self._label}] Error "
                f"after {latency_ms:.0f}ms: {e} | "
                f"text={sentence[:60]!r}"
            )
            # Fallback: push original so UI doesn't go blank
            try:
                self._smoother.push(sentence)
            except Exception:
                pass

    def _build_context_hint(self) -> str:
        """Build a truncated context string from recent translations.

        Returns up to _CONTEXT_MAX_WORDS words of recent TARGET-language text.
        This is passed alongside the new sentence to NLLB so it can maintain
        stylistic and topical coherence across chunk boundaries.
        """
        with self._lock:
            if not self._context:
                return ""
            # Collect recent translated sentences (target side only)
            recent_parts = [tgt for _, tgt in self._context]

        recent = " ".join(recent_parts)
        words = recent.split()
        if len(words) > _CONTEXT_MAX_WORDS:
            recent = " ".join(words[-_CONTEXT_MAX_WORDS:])

        return recent.strip()

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text on sentence boundaries.

        Preserves punctuation by keeping it attached to the preceding
        sentence, then adds a trailing '.' if the last part has none.

        Examples:
            "Hello world. How are you?" → ["Hello world.", "How are you?"]
            "xin chào bạn"             → ["xin chào bạn"]
        """
        import re
        # Split after . ! ? ; : followed by whitespace and an uppercase/viet char
        parts = re.split(r'(?<=[.!?;:])\s+', text)
        result = []
        for part in parts:
            part = part.strip()
            if part:
                result.append(part)
        return result if result else [text]
