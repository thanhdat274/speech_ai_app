"""
subtitle_smoother.py
=============================================================================
Vietnamese Subtitle Stabilization System

Prevents rapid flickering and random word jumps in streaming subtitles.

Strategy:
    1. Accumulate short fragments (< 3 words) and merge with next segment
    2. Word-overlap deduplication: if two segments share ≥ 70% words,
       keep the longer one (it's the "completed" version)
    3. Debounce UI emission by a configurable delay (default 300 ms)
    4. Vietnamese punctuation correction on final output

Example:
    Input:   "cái này"          (2 words, too short → buffer)
    Input:   "cái này rất hay"  (70%+ overlap with buffer → replace)
    Output:  "cái này rất hay"  (single stable emission)
=============================================================================
"""

import threading
from typing import Callable, List, Optional


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
MIN_WORDS_TO_EMIT  = 3     # Segments shorter than this are merged with next
SIMILARITY_THRESHOLD = 0.70  # Word-overlap ratio to consider "same sentence"
DEFAULT_DELAY_S    = 0.3   # Debounce delay before emitting to UI


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


class SubtitleSmoother:
    """Stabilizes streaming Vietnamese subtitles.

    Thread-safe. Designed to sit between Whisper output and the UI callback.

    Usage::

        smoother = SubtitleSmoother(callback=update_ui)
        # Whisper partial segments:
        smoother.push("cái này")         # buffered (< 3 words)
        smoother.push("cái này rất hay") # merged, emitted after 300ms
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

        self._pending: List[str] = []     # Accumulated short fragments
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

        # Last emitted text — used for deduplication
        self._last_emitted: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def push(self, text: str) -> None:
        """Push a partial segment from Whisper.

        Short segments (< MIN_WORDS_TO_EMIT words) are buffered.
        If the new segment has ≥ 70% word overlap with the buffer,
        the longer version replaces the shorter one (stabilization).
        Restarts the debounce timer on every push.
        """
        text = text.strip()
        if not text:
            return

        with self._lock:
            new_words = text.split()

            if self._pending:
                last = self._pending[-1]

                # ── Similarity check: replace if ≥ 70% overlap ────────
                ratio = _word_overlap_ratio(last, text)
                if ratio >= SIMILARITY_THRESHOLD:
                    # Keep the longer version — it's the "completed" sentence
                    if len(new_words) >= len(last.split()):
                        self._pending[-1] = text
                    # else: new text is shorter/same → ignore (keep existing)
                    self._restart_timer()
                    return

                # ── Exact duplicate suppression ───────────────────────
                if last.lower() == text.lower():
                    return

            # ── Short fragment merging ────────────────────────────────
            # If the previous pending segment is too short, merge with new text
            if self._pending and len(self._pending[-1].split()) < MIN_WORDS_TO_EMIT:
                prev = self._pending[-1]
                # Check if new text already contains the short fragment
                if text.lower().startswith(prev.lower()):
                    # New text is an extension → replace
                    self._pending[-1] = text
                else:
                    # Separate thought → concatenate
                    self._pending[-1] = f"{prev} {text}"
                self._restart_timer()
                return

            # New independent segment
            self._pending.append(text)
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
    def _restart_timer(self) -> None:
        """Cancel existing timer and start a new debounce countdown.

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
        """Final deduplication + Vietnamese correction before UI emission."""
        if not text:
            return

        # ── Suppress if identical or highly similar to last emission ───
        if self._last_emitted:
            if text.lower() == self._last_emitted.lower():
                return
            ratio = _word_overlap_ratio(text, self._last_emitted)
            if ratio >= SIMILARITY_THRESHOLD and len(text.split()) <= len(self._last_emitted.split()):
                # New text is shorter and overlaps heavily → skip (it's a regression)
                return

        self._last_emitted = text

        # ── Vietnamese punctuation/grammar correction ─────────────────
        try:
            from core.vietnamese_corrector import VietnameseCorrector
            text = VietnameseCorrector.apply_corrections(text)
        except Exception:
            pass

        self._target_callback(text)
