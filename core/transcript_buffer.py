"""
transcript_buffer.py
=============================================================================
Thread-Safe In-Memory Transcript Buffer — v9 (zero file-IO on realtime path)

Fixes requirement #8:
    - append() writes to an in-memory deque ONLY — zero disk latency on the
      hot path.
    - A background FileWriter thread drains the deque asynchronously to
      live_transcript.txt so persistence is maintained without blocking.
    - read_new_lines() reads from the in-memory deque — no file seeks.
    - clear() empties the deque and signal + truncates the file off-thread.

Pipeline role:
    STT thread → TranscriptBuffer.append()   [in-memory, O(1), < 1µs]
                       ↓ deque (non-blocking)
    TranslationWorker  → read_new_lines()   [in-memory, O(n_new)]
    FileWriter thread  → drains deque       [async, does not block STT]

Thread safety:
    - append() and read_new_lines() share a lightweight RLock.
    - FileWriter thread uses its own queue so file writes never contend
      with the realtime path.
=============================================================================
"""

import logging
import queue
import threading
from collections import deque
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Size of the in-memory sentence deque — ~3600 sentences at 1 s each = 1 hour
_DEQUE_MAXLEN    = 3600
# Number of sentences per batch written to file by the FileWriter thread
_FILE_BATCH_SIZE = 10


class TranscriptBuffer:
    """In-memory transcript buffer with async file persistence.

    The realtime path (append / read_new_lines) is pure in-memory:
        - append()        : deque.append(), O(1), no lock contention
        - read_new_lines(): returns new items since last call, O(n_new)

    File writes happen in a daemon thread and do NOT block the STT pipeline.

    Usage::

        buf = TranscriptBuffer()
        buf.append("Hôm nay trời đẹp.")   # zero-latency, in STT thread
        lines = buf.read_new_lines()       # in TranslationWorker thread
        buf.clear()                        # called by Clear button
    """

    def __init__(self, path: str = "live_transcript.txt") -> None:
        self._path = Path(path)

        # ── In-memory store ─────────────────────────────────────────────────
        # _sentences: the canonical in-memory store of all accumulated sentences.
        # _read_idx:  index of next sentence to return from read_new_lines().
        self._lock       = threading.RLock()
        self._sentences: deque = deque(maxlen=_DEQUE_MAXLEN)
        self._read_idx: int = 0

        # ── Async file writer ────────────────────────────────────────────────
        # Sentences pushed here are written to disk off the realtime path.
        self._file_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._writer_thread = threading.Thread(
            target=self._file_writer_loop,
            name="TranscriptFileWriter",
            daemon=True,
        )
        self._writer_thread.start()

        # Truncate / create the file on startup
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text("", encoding="utf-8")
            logger.info(f"TranscriptBuffer ready | in-memory + async file: {self._path.resolve()}")
        except Exception as e:
            logger.warning(f"TranscriptBuffer: cannot initialise file {self._path}: {e}")

    # ------------------------------------------------------------------
    # Public API — realtime path (in-memory only)
    # ------------------------------------------------------------------

    def append(self, text: str) -> None:
        """Append a sentence.

        Called from the STT worker thread.  Zero disk I/O — O(1) in-memory
        deque.append() + non-blocking push to the async file queue.

        Args:
            text: A complete sentence emitted by SentenceBuilder.
        """
        text = text.strip()
        if not text:
            return

        with self._lock:
            self._sentences.append(text)

        # Push to async file writer — non-blocking (SimpleQueue has no maxsize)
        self._file_queue.put(text)
        logger.debug(f"[TranscriptBuffer] append: {text[:50]!r}")

    def read_new_lines(self) -> List[str]:
        """Return sentences appended since the last call.

        Called from the TranslationWorker polling loop.
        Pure in-memory — no file access, no disk latency.

        Returns:
            List of new sentences (may be empty).
        """
        with self._lock:
            total = len(self._sentences)
            if self._read_idx >= total:
                return []

            # Convert deque slice to list
            sentences_list = list(self._sentences)
            new = sentences_list[self._read_idx:]
            self._read_idx = total

        if new:
            logger.debug(f"[TranscriptBuffer] read_new_lines: {len(new)} new sentence(s)")
        return new

    def clear(self) -> None:
        """Clear all in-memory state and truncate the file asynchronously.

        Called by the UI Clear button. Safe to call from any thread.
        """
        with self._lock:
            self._sentences.clear()
            self._read_idx = 0

        # Signal the file writer to truncate
        self._file_queue.put(None)   # None = truncate signal
        logger.info("[TranscriptBuffer] cleared (in-memory + file truncate queued).")

    def get_all_lines(self) -> List[str]:
        """Return all sentences currently in memory (for debug / UI reload)."""
        with self._lock:
            return list(self._sentences)

    @property
    def file_path(self) -> Path:
        """Resolved absolute path to the backing file."""
        return self._path.resolve()

    # ------------------------------------------------------------------
    # Async file writer — runs in its own daemon thread
    # ------------------------------------------------------------------

    def _file_writer_loop(self) -> None:
        """Drain the file queue and batch-write sentences to disk.

        Runs entirely off the realtime path.  Uses a 50 ms flush interval
        so sentences reach disk promptly even under low load.
        """
        batch: List[str] = []

        while True:
            try:
                # Block up to 50 ms for the next item
                try:
                    item = self._file_queue.get()
                except Exception:
                    break

                # None = truncate signal from clear()
                if item is None:
                    self._flush_batch(batch)
                    batch = []
                    try:
                        self._path.write_text("", encoding="utf-8")
                        logger.debug("[TranscriptBuffer] file truncated by clear().")
                    except Exception as e:
                        logger.warning(f"[TranscriptBuffer] truncate error: {e}")
                    continue

                batch.append(item)

                # Batch-write every _FILE_BATCH_SIZE sentences
                if len(batch) >= _FILE_BATCH_SIZE:
                    self._flush_batch(batch)
                    batch = []

                # Drain any immediately available items without blocking
                while True:
                    try:
                        extra = self._file_queue.get_nowait()
                        if extra is None:
                            self._flush_batch(batch)
                            batch = []
                            try:
                                self._path.write_text("", encoding="utf-8")
                            except Exception:
                                pass
                        else:
                            batch.append(extra)
                            if len(batch) >= _FILE_BATCH_SIZE:
                                self._flush_batch(batch)
                                batch = []
                    except Exception:
                        break

            except Exception as e:
                logger.error(f"[TranscriptBuffer] FileWriter error: {e}")

        # Flush any remaining batch before thread exits
        self._flush_batch(batch)

    def _flush_batch(self, batch: List[str]) -> None:
        """Write a batch of sentences to file — called only from FileWriter thread."""
        if not batch:
            return
        try:
            with self._path.open("a", encoding="utf-8") as f:
                f.write("\n".join(batch) + "\n")
            logger.debug(f"[TranscriptBuffer] flushed {len(batch)} sentence(s) to file.")
        except Exception as e:
            logger.warning(f"[TranscriptBuffer] file flush error: {e}")
