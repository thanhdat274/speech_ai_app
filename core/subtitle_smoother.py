"""
subtitle_smoother.py
=============================================================================
Vietnamese Subtitle Stabilization System

Prevents rapid flickering and random word jumps in streaming subtitles.

Strategy (v6 / Section 8):
    1. Accumulate short fragments (< 3 words) using deque(maxlen=5)
    2. Word-overlap deduplication: ≥ 70% word overlap → keep longer version
    3. Debounce UI emission by ~150 ms delay (Section 8 target)
    4. Flush only when:
         - Punctuation detected in incoming text
         - Timeout (150 ms) reached
         - New sentence starts (different from existing buffer)
    5. Vietnamese punctuation correction on final output

Changes (v6):
    - Buffer upgraded to deque(maxlen=5) per Section 8 spec
    - Default delay reduced to 150 ms (Section 8: "delay ~150ms to collect fragments")
    - Flush triggered by punctuation detection in incoming text
    - New sentence detection to avoid mixing unrelated fragments
    - Improved dedup: SequenceMatcher for better similarity detection

Example:
    Input:   "cái này"              (2 words → buffered)
    Input:   "cái này rất hay"      (70%+ overlap → replaces, still buffered)
    Timer:   → after 150 ms: emit "cái này rất hay"  (stable single emission)
=============================================================================
"""

import re
import threading
from collections import deque
from typing import Callable, Optional

import difflib


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
MIN_WORDS_TO_EMIT    = 3      # Segments shorter than this are merged with next
SIMILARITY_THRESHOLD = 0.70   # Word-overlap ratio to consider "same sentence"
# v9 req #4: 80 ms debounce (was 150 ms) — faster subtitle appearance
DEFAULT_DELAY_S      = 0.08   # 80 ms

# Pattern to detect end-of-sentence punctuation
_PUNCTUATION_END = re.compile(r'[.!?…;:]\s*$')
_DEQUE_MAXLEN    = 5          # Section 8: deque(maxlen=5)


def _word_overlap_ratio(a: str, b: str) -> float:
    """Return the fraction of words in the shorter string that appear in the longer.

    Uses set intersection — order-independent, O(N) for Vietnamese word tokens.
    Returns 0.0 when either string is empty.
    """
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    shorter = words_a if len(words_a) <= len(words_b) else words_b
    overlap = words_a & words_b
    return len(overlap) / len(shorter)


def _seq_similarity(a: str, b: str) -> float:
    """SequenceMatcher similarity between two strings."""
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


class SubtitleSmoother:
    """Stabilizes streaming Vietnamese subtitles.

    Thread-safe. Designed to sit between Whisper output and the UI callback.

    Section 8 Implementation:
    - deque(maxlen=5) as the fragment buffer
    - 150 ms debounce delay before emitting
    - Flush on: punctuation detected | timeout | new sentence start

    Usage::

        smoother = SubtitleSmoother(callback=update_ui)
        # Whisper partial segments:
        smoother.push("cái này")         # buffered (< 3 words)
        smoother.push("cái này rất hay") # merged, emitted after 150ms
        # Final flush at end of chunk:
        smoother.flush("Cái này rất hay.")
    """

    def __init__(
        self,
        target_callback: Callable[[str], None],
        delay_s: float = DEFAULT_DELAY_S,
    ):
        self._target_callback = target_callback
        self._delay_s = delay_s

        # Section 8: deque(maxlen=5) instead of plain list
        self._pending: deque = deque(maxlen=_DEQUE_MAXLEN)
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

        # Last emitted text — used for deduplication
        self._last_emitted: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def push(self, text: str) -> None:
        """Push a partial segment from Whisper.

        Section 8 flush triggers:
        - Punctuation detected in text → flush immediately
        - Otherwise: debounce 150 ms timer
        - Short segments (< MIN_WORDS_TO_EMIT) are merged with existing buffer
        """
        text = text.strip()
        if not text:
            return

        with self._lock:
            new_words = text.split()

            if self._pending:
                last = self._pending[-1]

                # ── Similarity check: replace if ≥ 70% overlap ────────────
                ratio = _word_overlap_ratio(last, text)
                if ratio >= SIMILARITY_THRESHOLD:
                    # Keep the longer version — it's the "completed" sentence
                    if len(new_words) >= len(last.split()):
                        self._pending[-1] = text
                    self._restart_timer()
                    return

                # ── Exact duplicate suppression ───────────────────────────
                if last.lower() == text.lower():
                    return

            # ── Short fragment merging ─────────────────────────────────────
            if self._pending and len(self._pending[-1].split()) < MIN_WORDS_TO_EMIT:
                prev = self._pending[-1]
                if text.lower().startswith(prev.lower()):
                    self._pending[-1] = text
                else:
                    self._pending[-1] = f"{prev} {text}"
                self._restart_timer()
                return

            # New independent segment
            self._pending.append(text)

            # Section 8: Flush trigger — punctuation detected
            if _PUNCTUATION_END.search(text):
                self._flush_now()
                return

            self._restart_timer()

    def flush(self, text: str) -> None:
        """Immediately emit final text, bypassing the debounce timer.

        Called at the end of a chunk when the full transcription is ready.
        Cancels any pending timer and clears the buffer.
        """
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._pending.clear()

        if not text or not text.strip():
            return

        text = text.strip()
        self._dedupe_and_emit(text)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _flush_now(self) -> None:
        """Immediately emit pending buffer (called under lock for punctuation trigger)."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if not self._pending:
            return
        combined = " ".join(self._pending)
        self._pending.clear()
        # Release lock before emitting to avoid deadlock with callback
        threading.Thread(
            target=self._dedupe_and_emit,
            args=(combined.strip(),),
            daemon=True,
        ).start()

    def _restart_timer(self) -> None:
        """Cancel existing timer and start a new 150 ms debounce countdown.

        Must be called while holding self._lock.
        """
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(self._delay_s, self._emit_pending)
        self._timer.daemon = True
        self._timer.start()

    def _emit_pending(self) -> None:
        """Timer callback: merge all pending fragments and emit."""
        with self._lock:
            if not self._pending:
                return
            combined = " ".join(self._pending)
            self._pending.clear()
            self._timer = None

        self._dedupe_and_emit(combined.strip())

    def _dedupe_and_emit(self, text: str) -> None:
        """Final deduplication + Vietnamese correction before UI emission.

        Uses both word-overlap ratio and SequenceMatcher for better dedup.
        """
        if not text:
            return

        # ── Suppress if identical or highly similar to last emission ───────
        if self._last_emitted:
            if text.lower() == self._last_emitted.lower():
                return

            # Word-overlap check
            ratio = _word_overlap_ratio(text, self._last_emitted)
            if ratio >= SIMILARITY_THRESHOLD and len(text.split()) <= len(self._last_emitted.split()):
                return

            # SequenceMatcher check for near-duplicates
            sim = _seq_similarity(text, self._last_emitted)
            if sim > 0.85 and len(text.split()) <= len(self._last_emitted.split()):
                return

        self._last_emitted = text

        # ── Vietnamese punctuation/grammar correction ──────────────────────
        try:
            from core.vietnamese_corrector import VietnameseCorrector
            text = VietnameseCorrector.apply_corrections(text)
        except Exception:
            pass

        self._target_callback(text)
