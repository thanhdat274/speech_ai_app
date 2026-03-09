"""
sentence_builder.py
=============================================================================
Streaming Sentence Builder for Translation Pipeline

Accumulates partial ASR output and emits COMPLETE sentences for translation.
This prevents translating raw audio fragments which causes:
  - broken sentences
  - unstable translation output
  - poor context for NLLB

A sentence is considered "complete" when:
  1. Punctuation detected (. ! ? … ; :)
  2. Silence pause > PAUSE_THRESHOLD_S (1.2 s per Section 6)
  3. Sentence length exceeds MAX_WORDS threshold

Pipeline:
    Faster-Whisper partial → SentenceBuilder → complete sentence → NLLB → UI

Changes (v6 / Section 6):
  - Silence flush threshold raised to 1.2 s (matches Section 6 spec)
  - Fragment merging: if fragment ends without punctuation → wait for next
  - Flush triggers: punctuation OR 1.2 s silence OR MAX_WORDS

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
_SENTENCE_ENDINGS    = re.compile(r'[.!?…;:]\s*$')      # Ends with punctuation
_MID_SENTENCE_SPLIT  = re.compile(r'[.!?…;:]\s+')       # Punctuation + space mid-text
_VIETNAMESE_ENDINGS  = re.compile(r'[.!?…]\s*$')         # Vietnamese sentence enders

# v9: flush thresholds (req #2)
PAUSE_THRESHOLD_S  = 0.60   # 600 ms silence → flush  (was 1.2 s)
MAX_WORDS          = 6      # Flush when buffer ≥ 6 words (was 30)
MIN_WORDS_TO_EMIT  = 2      # Don't emit single-word fragments (was 3)


class SentenceBuilder:
    """Accumulates streaming ASR partials and emits complete sentences.

    Thread-safe. One instance per source (sys/mic).

    Section 6 Spec:
    - If fragment ends without punctuation → wait for next fragment
    - If silence > 1.2 s → flush sentence
    - This removes subtitle flicker by never emitting mid-sentence fragments

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

        # Timer for pause-based emission (1.2 s)
        self._timer: Optional[threading.Timer] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_partial(self, text: str) -> None:
        """Add a partial ASR segment.

        The builder detects if the text contains complete sentences and
        emits them immediately. Incomplete trailing text stays buffered
        (Section 6: if fragment ends without punctuation → wait).
        """
        text = text.strip()
        if not text:
            return

        with self._lock:
            self._last_update = time.monotonic()

            # ── Overlap detection: if new text extends buffer, replace ──────
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

            # ── Check for complete sentences (punctuation boundary) ──────────
            self._try_emit_sentences()

            # ── Check word count threshold ───────────────────────────────────
            if len(self._buffer.split()) >= self._max_words:
                self._emit_buffer()

            # ── Restart pause timer (1.2 s before flush) ────────────────────
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

        Section 6 rule: fragment without punctuation → stay buffered.
        Only emit when punctuation is detected.

        Must be called while holding self._lock.
        """
        if not self._buffer:
            return

        # Split on punctuation followed by space (mid-text boundaries)
        parts = _MID_SENTENCE_SPLIT.split(self._buffer)

        if len(parts) <= 1:
            # No mid-text split found — check if buffer ends with punctuation
            if _SENTENCE_ENDINGS.search(self._buffer):
                self._emit_buffer()
            # else: no punctuation → buffer and wait (Section 6)
            return

        # There are mid-text splits — collect complete sentences and keep remainder
        sentences: List[str] = []
        remaining = self._buffer

        for match in _MID_SENTENCE_SPLIT.finditer(self._buffer):
            sentence = self._buffer[:match.start()] + self._buffer[match.start():match.end()].rstrip()
            if sentence.strip():
                sentences.append(sentence.strip())
            remaining = self._buffer[match.end():]

        # Emit all complete sentences found
        for sentence in sentences:
            if len(sentence.split()) >= MIN_WORDS_TO_EMIT:
                self._on_sentence(sentence)

        # Keep the remaining incomplete part in buffer (wait for more input)
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
    # Pause timer (1.2 s)
    # ------------------------------------------------------------------
    def _restart_timer(self) -> None:
        """Restart the 1.2 s pause-detection timer. Must hold self._lock."""
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
        """Called when no new partials arrive for PAUSE_THRESHOLD_S (1.2 s).

        Section 6: silence > 1.2 s → flush sentence.
        """
        with self._lock:
            if self._buffer and len(self._buffer.split()) >= MIN_WORDS_TO_EMIT:
                self._emit_buffer()
