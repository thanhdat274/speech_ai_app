"""
sentence_builder.py
=============================================================================
Streaming Sentence Builder for Translation Pipeline

Accumulates partial ASR output and emits complete sentences for translation.
This prevents translating raw audio chunks which causes:
  - broken sentences
  - unstable translation output
  - poor context for NLLB

A sentence is considered "complete" when:
  1. Punctuation detected (. ! ? … ; :)
  2. Silence pause > PAUSE_THRESHOLD_MS (800ms)
  3. Sentence length exceeds MAX_WORDS threshold

Pipeline:
    Faster-Whisper partial → SentenceBuilder → complete sentence → NLLB → UI

Example:
    add_partial("today we will talk about")           → no sentence yet
    add_partial("today we will talk about AI.")        → yields "today we will talk about AI."
    add_partial("and also robotics")                   → buffered
    flush()                                            → yields "and also robotics"
=============================================================================
"""

import re
import threading
import time
from typing import Callable, List, Optional


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
# Sentence boundary detection
_SENTENCE_ENDINGS = re.compile(r'[.!?…;:]\s*$')          # Ends with punctuation
_MID_SENTENCE_SPLIT = re.compile(r'[.!?…;:]\s+')         # Punctuation + space mid-text
_VIETNAMESE_ENDINGS = re.compile(r'[.!?…]\s*$')           # Vietnamese sentence enders

PAUSE_THRESHOLD_S = 0.8    # 800ms silence = force emit buffered text
MAX_WORDS = 25             # Force emit when buffer exceeds this word count
MIN_WORDS_TO_EMIT = 3      # Don't emit fragments shorter than this


class SentenceBuilder:
    """Accumulates streaming ASR partials and emits complete sentences.

    Thread-safe. One instance per source (sys/mic).

    Usage::

        sb = SentenceBuilder(on_sentence=translate_fn)
        # Whisper partials arrive:
        sb.add_partial("hôm nay chúng ta")       # buffered
        sb.add_partial("hôm nay chúng ta sẽ nói về AI.")  # sentence detected → emitted
        # On chunk boundary:
        sb.flush()  # force-emit any remaining buffer
    """

    def __init__(
        self,
        on_sentence: Callable[[str], None],
        pause_threshold_s: float = PAUSE_THRESHOLD_S,
        max_words: int = MAX_WORDS,
    ) -> None:
        self._on_sentence = on_sentence
        self._pause_threshold_s = pause_threshold_s
        self._max_words = max_words

        self._buffer: str = ""
        self._last_update: float = time.monotonic()
        self._lock = threading.Lock()

        # Timer for pause-based emission
        self._timer: Optional[threading.Timer] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_partial(self, text: str) -> None:
        """Add a partial ASR segment.

        The builder detects if the text contains complete sentences and
        emits them immediately. Incomplete trailing text stays buffered.
        """
        text = text.strip()
        if not text:
            return

        with self._lock:
            self._last_update = time.monotonic()

            # ── Overlap detection: if new text is an extension of buffer,
            # replace buffer instead of appending ──────────────────────
            if self._buffer:
                buf_lower = self._buffer.lower()
                txt_lower = text.lower()
                if txt_lower.startswith(buf_lower):
                    # New text extends the buffer → replace
                    self._buffer = text
                elif buf_lower.startswith(txt_lower):
                    # Buffer already has more content → keep buffer
                    pass
                else:
                    # Different content → append with space
                    self._buffer = f"{self._buffer} {text}"
            else:
                self._buffer = text

            # ── Check for complete sentences ──────────────────────────
            self._try_emit_sentences()

            # ── Check word count threshold ────────────────────────────
            if len(self._buffer.split()) >= self._max_words:
                self._emit_buffer()

            # ── Restart pause timer ───────────────────────────────────
            self._restart_timer()

    def flush(self) -> None:
        """Force-emit any remaining buffered text.

        Called at chunk boundaries or when streaming stops.
        """
        with self._lock:
            self._cancel_timer()
            self._emit_buffer()

    def get_buffered(self) -> str:
        """Return current buffer contents without emitting."""
        with self._lock:
            return self._buffer

    def reset(self) -> None:
        """Clear all state."""
        with self._lock:
            self._cancel_timer()
            self._buffer = ""

    # ------------------------------------------------------------------
    # Internal sentence detection
    # ------------------------------------------------------------------
    def _try_emit_sentences(self) -> None:
        """Split buffer on sentence boundaries and emit complete sentences.

        Must be called while holding self._lock.
        """
        if not self._buffer:
            return

        # Split on punctuation followed by space (mid-text boundaries)
        # e.g. "Câu một. Câu hai đang viết" → emit "Câu một.", keep "Câu hai đang viết"
        parts = _MID_SENTENCE_SPLIT.split(self._buffer)

        if len(parts) <= 1:
            # No mid-text split found — check if buffer ends with punctuation
            if _SENTENCE_ENDINGS.search(self._buffer):
                self._emit_buffer()
            return

        # Find where each split point is to reconstruct with punctuation
        sentences: List[str] = []
        remaining = self._buffer

        for match in _MID_SENTENCE_SPLIT.finditer(self._buffer):
            end_pos = match.end()
            sentence = self._buffer[:match.start()] + self._buffer[match.start():match.end()].rstrip()
            if sentence.strip():
                sentences.append(sentence.strip())
            remaining = self._buffer[end_pos:]

        # Emit all complete sentences
        for sentence in sentences:
            if len(sentence.split()) >= MIN_WORDS_TO_EMIT:
                self._on_sentence(sentence)

        # Keep the remaining incomplete part in buffer
        self._buffer = remaining.strip()

    def _emit_buffer(self) -> None:
        """Emit the entire buffer content. Must hold self._lock."""
        if not self._buffer:
            return

        text = self._buffer.strip()
        self._buffer = ""

        if not text or len(text.split()) < MIN_WORDS_TO_EMIT:
            # Too short — put it back, will be merged with next partial
            self._buffer = text
            return

        self._on_sentence(text)

    # ------------------------------------------------------------------
    # Pause timer
    # ------------------------------------------------------------------
    def _restart_timer(self) -> None:
        """Restart the pause-detection timer. Must hold self._lock."""
        self._cancel_timer()
        self._timer = threading.Timer(
            self._pause_threshold_s, self._on_pause_timeout
        )
        self._timer.daemon = True
        self._timer.start()

    def _cancel_timer(self) -> None:
        """Cancel any running timer. Must hold self._lock."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _on_pause_timeout(self) -> None:
        """Called when no new partials arrive for PAUSE_THRESHOLD_S."""
        with self._lock:
            if self._buffer and len(self._buffer.split()) >= MIN_WORDS_TO_EMIT:
                self._emit_buffer()
