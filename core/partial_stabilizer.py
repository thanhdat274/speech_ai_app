"""
partial_stabilizer.py
=============================================================================
Partial Transcript Stabilizer — MacWhisper-style word consensus

Problem:
    Whisper emits overlapping windows. Each window partially overlaps the
    prior one. A word near a chunk boundary may appear in slightly different
    forms across two consecutive passes:
        Pass 1: "Hello, how are"
        Pass 2: "Hello, how are you"
    Naively forwarding every pass creates flickering UI text.

Solution — Consensus Stabilization:
    Track which words have appeared in at least MIN_PASSES consecutive
    Whisper output windows. Only those words are considered "stable" and
    forwarded to the SentenceBuilder. The tail of the buffer (unstable words)
    is shown as a "pending / italic partial" in the UI but not fed to
    translation.

Architecture:
    StreamingTranscriber.process_chunk()
        → PartialStabilizer.push(text)         [per-source, one instance each]
            → on_stable(stable_text)            [→ SentenceBuilder]
            → on_partial(partial_text)          [→ UI partial display]

Thread safety:
    push() is called from the Whisper worker thread.
    Callbacks are dispatched synchronously — callers must be fast or
    dispatch further work to another thread.

Parameters (tunable):
    MIN_PASSES  = 2   — word must appear in 2 consecutive windows to be stable
    MAX_PENDING = 12  — maximum pending (unstable) word count before force-flush
    WINDOW_S    = 4.0 — rolling window of text history to compare against
=============================================================================
"""

import logging
import re
import threading
import time
from collections import deque
from difflib import SequenceMatcher
from typing import Callable, Deque, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Tuning parameters
# ─────────────────────────────────────────────────────────────────────────────
MIN_PASSES   = 2    # A word must appear in this many consecutive passes to stabilize
MAX_PENDING  = 12   # Force flush pending words after this many accumulate
WINDOW_S     = 4.0  # Rolling window of history to compare against (seconds)
_MAX_HISTORY = 6    # Keep last N Whisper outputs for overlap detection


def _tokenize(text: str) -> List[str]:
    """Split text into lowercase word tokens, stripping punctuation."""
    return re.findall(r"[^\s]+", text.lower())


def _longest_common_prefix(a: List[str], b: List[str]) -> int:
    """Return the count of leading words shared by both lists."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


class PartialStabilizer:
    """MacWhisper-style word-level consensus stabilizer.

    Each Whisper window is compared against recent history to determine which
    words have been consistently seen. Stable words are forwarded; unstable
    (pending) words are held back for one more pass.

    Usage::

        def on_stable(text): sentence_builder.add_partial(text)
        def on_partial(text): ui.show_partial(text)

        stab = PartialStabilizer(on_stable=on_stable, on_partial=on_partial)

        # Called from StreamingTranscriber for every Whisper result:
        stab.push("Hello, how are you doing today")
        stab.push("Hello, how are you doing today really")
    """

    def __init__(
        self,
        on_stable:  Optional[Callable[[str], None]] = None,
        on_partial: Optional[Callable[[str], None]] = None,
        min_passes:  int   = MIN_PASSES,
        max_pending: int   = MAX_PENDING,
        window_s:    float = WINDOW_S,
    ) -> None:
        self._on_stable  = on_stable
        self._on_partial = on_partial

        self._min_passes  = min_passes
        self._max_pending = max_pending
        self._window_s    = window_s

        self._lock = threading.Lock()

        # Rolling history of (timestamp, token_list) for recent Whisper outputs
        self._history: Deque[Tuple[float, List[str]]] = deque(maxlen=_MAX_HISTORY)

        # Tokens already emitted as stable (to avoid re-emitting)
        self._emitted_tokens: List[str] = []

        # Pending (seen in last pass but not yet confirmed stable)
        self._pending_tokens: List[str] = []

        # Word vote counters: token → consecutive pass count
        self._vote: dict = {}

        # Timestamp of last push (for forced flush on silence)
        self._last_push_t: float = 0.0
        self._flush_timer: Optional[threading.Timer] = None

        logger.info(
            f"[PartialStabilizer] init | min_passes={min_passes} "
            f"max_pending={max_pending} window_s={window_s}s"
        )

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def push(self, text: str) -> None:
        """Accept a new Whisper window output and update the stabilizer state.

        Called from the Whisper worker thread for every completed window.
        Thread-safe.

        Args:
            text: Raw Whisper transcript text for this window.
        """
        text = text.strip()
        if not text:
            return

        tokens = _tokenize(text)
        if not tokens:
            return

        now = time.monotonic()

        with self._lock:
            self._last_push_t = now

            # Cancel any pending forced-flush timer
            if self._flush_timer is not None:
                self._flush_timer.cancel()
                self._flush_timer = None

            self._update_votes(tokens, now)
            stable_tokens, pending_tokens = self._partition_tokens(tokens)

            self._emit_stable(stable_tokens)
            self._pending_tokens = pending_tokens

            if pending_tokens and self._on_partial:
                try:
                    self._on_partial(" ".join(pending_tokens))
                except Exception as e:
                    logger.error(f"[PartialStabilizer] on_partial error: {e}")

            # Schedule a forced-flush after 600ms silence (matches SentenceBuilder)
            self._flush_timer = threading.Timer(0.6, self._force_flush)
            self._flush_timer.daemon = True
            self._flush_timer.start()

            # Record this window in history
            self._history.append((now, tokens))

    def flush(self) -> None:
        """Force-emit all accumulated tokens immediately (called on stop/clear)."""
        with self._lock:
            if self._flush_timer is not None:
                self._flush_timer.cancel()
                self._flush_timer = None
            pending = self._pending_tokens[:]
            self._pending_tokens = []

        if pending and self._on_stable:
            try:
                self._on_stable(" ".join(pending))
            except Exception as e:
                logger.error(f"[PartialStabilizer] flush on_stable error: {e}")

    def reset(self) -> None:
        """Clear all state — called when session is reset or Clear pressed."""
        with self._lock:
            if self._flush_timer is not None:
                self._flush_timer.cancel()
                self._flush_timer = None
            self._history.clear()
            self._emitted_tokens.clear()
            self._pending_tokens.clear()
            self._vote.clear()
        logger.debug("[PartialStabilizer] reset.")

    # ─────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────

    def _update_votes(self, tokens: List[str], now: float) -> None:
        """Update per-word vote counts based on overlap with previous window.

        A word's vote increments when it appears at the same relative position
        in consecutive Whisper outputs (overlap-aware comparison).
        """
        if not self._history:
            # First window — all words get vote=1
            self._vote = {t: 1 for t in tokens}
            return

        # Get most recent history entry
        _, prev_tokens = self._history[-1]

        # Find where previous and current outputs diverge
        # (they share a common prefix from the overlap region)
        prefix_len = _longest_common_prefix(prev_tokens, tokens)

        # Words in the overlap prefix have been seen again — increment vote
        for i, tok in enumerate(tokens):
            if i < prefix_len:
                # In the stable overlap — increment
                self._vote[tok] = self._vote.get(tok, 0) + 1
            else:
                # New word in this window — start vote at 1
                if tok not in self._vote:
                    self._vote[tok] = 1

        # Decay votes for any word from the previous window not in current
        prev_set = set(prev_tokens)
        curr_set = set(tokens)
        for tok in prev_set - curr_set:
            self._vote.pop(tok, None)

    def _partition_tokens(
        self, tokens: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Split tokens into (stable, pending) based on vote threshold.

        Stable = vote >= MIN_PASSES AND not already emitted.
        Pending = everything else in the new window.
        """
        # Determine the longest prefix of tokens that are stable
        stable_end = 0
        for i, tok in enumerate(tokens):
            if self._vote.get(tok, 0) >= self._min_passes:
                stable_end = i + 1
            else:
                break  # First non-stable word ends the stable prefix

        stable_new = tokens[:stable_end]
        pending    = tokens[stable_end:]

        # Remove already-emitted prefix
        already = len(self._emitted_tokens)
        if stable_end > already:
            stable_emit = stable_new[already:]
        else:
            stable_emit = []

        # Force flush if too many words are pending
        if len(pending) >= self._max_pending:
            stable_emit = stable_emit + pending
            pending = []

        return stable_emit, pending

    def _emit_stable(self, tokens: List[str]) -> None:
        """Emit stable tokens to the SentenceBuilder callback."""
        if not tokens:
            return

        text = " ".join(tokens)
        self._emitted_tokens.extend(tokens)

        logger.debug(f"[PartialStabilizer] stable: {text!r}")

        if self._on_stable:
            try:
                self._on_stable(text)
            except Exception as e:
                logger.error(f"[PartialStabilizer] on_stable error: {e}")

    def _force_flush(self) -> None:
        """Flush pending tokens after silence timeout — called by threading.Timer."""
        with self._lock:
            pending = self._pending_tokens[:]
            self._pending_tokens = []
            self._flush_timer = None

        if pending:
            logger.debug(f"[PartialStabilizer] force-flush after silence: {pending}")
            self._emit_stable(pending)
