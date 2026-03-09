"""
audio_engine.py
=============================================================================
AudioEngine Module — Speech-AI v6 Architecture
=============================================================================

Pipeline:

  [Hardware Callback]
       |  never blocks — puts raw float32 frames into RingBuffer
       v
  [RingBuffer]   circular, overwrites oldest on overflow; capture never stalls
       |
       v  (ChunkAggregator thread)
       |  - Silence detection via RMS / Silero VAD
       |  - Dynamic chunk aggregation: 200 ms capture → 800 ms–1.2 s chunks
       |  - Adaptive chunk duration (mode switching based on queue depth)
       |  - 200 ms overlap preserved between consecutive chunks
       |
       v  (ready ~800 ms–1.2 s speech chunks)
  [_ai_queue]   bounded work queue drained by the worker pool
       |          Backpressure: queue > 80% → merge into last chunk instead of enqueue
       v  (Whisper Worker Pool — AI_WORKER_COUNT threads share _ai_queue)
  user callback  (StreamingTranscriber.process_chunk → Whisper)

Key Changes (v8 — stable realtime):
  - ULTRA_CHUNK raised to 1.2 s: fewer Whisper calls, better context per inference
  - BALANCED at 1.0 s, ACCURACY at 1.5 s
  - AI_QUEUE_MAXSIZE reduced to 20 — tight budget prevents queue backlog
  - Backpressure threshold lowered to 70%: earlier merge, avoids cascade overflow
  - Silence detection: RMS gate + Silero VAD neural gate (auto-fallback)
  - Dynamic adaptive sizing with EMA smoothing
  - Auto mode switching: queue >90% → ACCURACY, >70% → BALANCED, <30% → ULTRA
  - Overlap preserved at 200 ms between chunks
"""

import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Callable, List, Optional

import numpy as np
import sounddevice as sd
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16_000             # Whisper requires 16 kHz mono PCM
CHANNELS = 1

# ── Chunk Duration Modes ─────────────────────────────────────────────────────
#
# v9 target:
#   ULTRA_REALTIME: 2.4 s chunk size, 0.6 s stride
#     - Larger window = more context per Whisper call
#     - Stride overlap = smooth sentence continuity across chunks
#     - Fewer total Whisper calls = less GPU contention
#   BALANCED:       1.6 s chunk, 0.5 s stride
#   ACCURACY:       2.4 s chunk, 0.8 s stride
ULTRA_CHUNK_DURATION_S    = 2.40   # v9: 2.4 s window
BALANCED_CHUNK_DURATION_S = 1.60   # v9: 1.6 s
ACCURACY_CHUNK_DURATION_S = 2.40   # v9: same window, longer stride
CHUNK_DURATION_S          = BALANCED_CHUNK_DURATION_S  # exported default

# Capture frame: 200 ms — hardware callback writes this into RingBuffer
CAPTURE_FRAME_S  = 0.20

# v9: stride overlap = 0.6 s (was 0.2 s)
# Overlap prevents clipping words at chunk boundaries
OVERLAP_DURATION_S = 0.60

RING_BUFFER_MAXFRAMES = 300

# v9: queue = 50 (req #5) — prevents backpressure on GTX 1650 Ti
# At 2.4 s chunks: 50 slots = 120 s of headroom
AI_QUEUE_MAXSIZE      = 50

# Backpressure at >80% queue depth (= 40/50)
BACKPRESSURE_THRESHOLD = 0.80
BACKPRESSURE_SLEEP_S   = 0.05

# v9: dual workers for round-robin parallel window processing (req #4)
# Two workers process 2.4s overlapping windows in parallel.
# GTX 1650 Ti handles two float16 Whisper-large-v3 calls interleaved
# at ~800ms each with sufficient VRAM headroom.
AI_WORKER_COUNT = 2

MONITOR_IDLE_S     = 5.0
MONITOR_WARN_S     = 1.0
MONITOR_CRITICAL_S = 0.5

VAD_RMS_THRESHOLD = 0.002

# ── Adaptive chunk sizing ─────────────────────────────────────────────────────
CHUNK_MIN_S = ULTRA_CHUNK_DURATION_S
CHUNK_MAX_S = ACCURACY_CHUNK_DURATION_S
ADAPTIVE_HIGH_THRESHOLD = 0.70
ADAPTIVE_LOW_THRESHOLD  = 0.30

# ── Auto mode switching ───────────────────────────────────────────────────────
MODE_SWITCH_TO_ACCURACY_THRESHOLD = 0.90
MODE_SWITCH_TO_BALANCED_THRESHOLD = 0.70
MODE_SWITCH_TO_ULTRA_THRESHOLD    = 0.30

_WARN_DEPTH         = int(AI_QUEUE_MAXSIZE * 0.50)
_BACKPRESSURE_DEPTH = int(AI_QUEUE_MAXSIZE * BACKPRESSURE_THRESHOLD)



# ---------------------------------------------------------------------------
# RingBuffer
# ---------------------------------------------------------------------------
class RingBuffer:
    """Thread-safe circular audio frame buffer.

    The capture callback is guaranteed to NEVER block:
    - put() is O(1) and always succeeds; the oldest frame is evicted when full.
    - get() sleeps via threading.Event — no busy-polling overhead.
    """

    def __init__(self, maxsize: int = RING_BUFFER_MAXFRAMES) -> None:
        self._buf: deque = deque(maxlen=maxsize)
        self._lock = threading.Lock()
        self._not_empty = threading.Event()

    def put(self, frame: np.ndarray) -> None:
        """Insert a frame (silently evicts the oldest entry when at capacity)."""
        with self._lock:
            self._buf.append(frame)
        self._not_empty.set()

    def get(self, timeout: float = 0.2) -> Optional[np.ndarray]:
        """Return the oldest frame, blocking up to *timeout* seconds."""
        if self._not_empty.wait(timeout=timeout):
            with self._lock:
                if self._buf:
                    item = self._buf.popleft()
                    if not self._buf:
                        self._not_empty.clear()
                    return item
                self._not_empty.clear()
        return None

    def unblock(self) -> None:
        """Wake any thread blocked inside get() — called by stop()."""
        self._not_empty.set()

    @property
    def size(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# SileroVAD  (lazy-loaded module-level singleton, RMS fallback)
# ---------------------------------------------------------------------------
class _SileroVAD:
    """Silero VAD wrapper with automatic RMS energy fallback."""

    def __init__(self) -> None:
        self._model = None
        self._get_ts: Optional[Callable] = None
        self._loaded = False
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            try:
                model, utils = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    force_reload=False,
                    trust_repo=True,
                    verbose=False,
                )
                self._model = model
                self._get_ts = utils[0]
                logger.info("Silero VAD loaded — neural speech detection active.")
            except Exception as exc:
                logger.warning(
                    f"Silero VAD unavailable ({exc}). "
                    "Using RMS energy gate as fallback."
                )
                self._model = None
            finally:
                self._loaded = True

    def is_speech(
        self,
        audio_16k: np.ndarray,
        threshold: float = 0.45,
        noise_gate: float = VAD_RMS_THRESHOLD,
    ) -> bool:
        """Return True if *audio_16k* (16 kHz float32 mono) contains speech."""
        self._ensure_loaded()

        if self._model is None:
            rms  = float(np.sqrt(np.mean(audio_16k ** 2)))
            peak = float(np.max(np.abs(audio_16k)))
            return rms >= noise_gate and peak >= noise_gate

        try:
            tensor = torch.from_numpy(audio_16k).float()
            with self._infer_lock:
                timestamps = self._get_ts(
                    tensor,
                    self._model,
                    sampling_rate=SAMPLE_RATE,
                    threshold=threshold,
                    min_speech_duration_ms=100,
                    min_silence_duration_ms=50,
                )
            return len(timestamps) > 0
        except Exception:
            return float(np.sqrt(np.mean(audio_16k ** 2))) >= noise_gate


_vad = _SileroVAD()


# ---------------------------------------------------------------------------
# BaseCapture
# ---------------------------------------------------------------------------
class BaseCapture(ABC):
    """Abstract base for audio capture sources.

    Subclasses open a hardware stream and call _start_workers() once ready.

    The pipeline is:
        Hardware Callback → RingBuffer → ChunkAggregator → _ai_queue → Workers
    """

    def __init__(
        self,
        callback: Callable[[np.ndarray], None],
        noise_gate: float = 0.001,
    ) -> None:
        self.callback = callback
        self.noise_gate = noise_gate
        self.is_running = False
        self._stream = None
        self.native_rate: int = SAMPLE_RATE

        # Set by AppController/LiveStreamingWorker before start() is called.
        # When True, ChunkAggregator uses the AudioPreprocessor fast path
        # (stereo→mono + resample only, skips DC/RMS/noise gate).
        # Target preprocess latency: < 10 ms instead of standard < 80 ms budget.
        self.ultra_realtime_mode: bool = False

        self._ring_buffer = RingBuffer()
        self._ai_queue: queue.Queue = queue.Queue(maxsize=AI_QUEUE_MAXSIZE)

        # For backpressure merging: reference to last enqueued chunk
        self._last_chunk_lock = threading.Lock()
        self._last_chunk: Optional[np.ndarray] = None

        self._processing_thread: Optional[threading.Thread] = None
        self._dispatch_threads: List[threading.Thread] = []
        self._monitor_thread: Optional[threading.Thread] = None

    @abstractmethod
    def start(self, chunk_duration_s: float = CHUNK_DURATION_S) -> None:
        pass

    def stop(self) -> None:
        self.is_running = False

        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        self._ring_buffer.unblock()
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=2.0)

        for _ in self._dispatch_threads:
            self._ai_queue.put(None)
        for t in self._dispatch_threads:
            if t.is_alive():
                t.join(timeout=2.0)
        self._dispatch_threads.clear()

    # ------------------------------------------------------------------
    # Hardware callback — must NEVER block
    # ------------------------------------------------------------------
    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info, status
    ) -> None:
        if status:
            logger.warning(f"Audio status: {status}")
        if self.is_running:
            self._ring_buffer.put(indata.copy())

    # ------------------------------------------------------------------
    # ChunkAggregator — Section 1 implementation
    #
    # Single thread that:
    #   1. reads 200 ms frames from the RingBuffer
    #   2. accumulates until chunk_samples reached
    #   3. runs silence detection (RMS gate)
    #   4. applies backpressure (Section 2): at >80% queue, merges chunk
    #   5. enqueues speech chunks with 200 ms overlap
    #   6. auto-adjusts chunk duration based on queue depth
    # ------------------------------------------------------------------
    def _processing_worker(self, chunk_duration_s: float) -> None:
        """Dynamic Chunk Builder with Silence Detection and Backpressure.

        Target: accumulate 200 ms captures into 800 ms–1.2 s chunks.
        Silence frames are discarded; speech chunks are enqueued.
        At backpressure (>80% queue) chunks are merged rather than dropped.
        """
        from core.audio_preprocessor import AudioPreprocessor
        preprocessor = AudioPreprocessor.get_instance()

        buffer: List[np.ndarray] = []
        buffered_samples = 0
        current_chunk_s = chunk_duration_s
        chunk_samples   = int(self.native_rate * current_chunk_s)
        overlap_samples = int(self.native_rate * OVERLAP_DURATION_S)

        # Accumulated merged chunk for backpressure (Section 2)
        merge_pending: Optional[np.ndarray] = None

        current_mode = "BALANCED"
        chunks_processed = 0
        last_mode_switch = time.monotonic()

        while self.is_running:
            try:
                frame = self._ring_buffer.get(timeout=0.2)
                if frame is None:
                    continue

                buffer.append(frame)
                buffered_samples += len(frame)

                if buffered_samples < chunk_samples:
                    continue

                chunk_start_t = time.perf_counter()

                # ── Build raw chunk ──────────────────────────────────────
                raw_chunk = np.concatenate(buffer, axis=0).flatten()

                # ── Preprocess: mono → resample → normalize ──────────────
                # ultra_realtime_mode → fast path (skips DC/RMS/noise gate).
                # Cuts preprocess latency from ~1500 ms to ~5-10 ms on 48kHz.
                clean_chunk = preprocessor.preprocess(
                    raw_chunk,
                    source_sr=self.native_rate,
                    ultra_fast=self.ultra_realtime_mode,
                )

                # Silence → keep overlap, skip inference
                if len(clean_chunk) == 0:
                    if overlap_samples > 0 and len(raw_chunk) > overlap_samples:
                        overlap_frame    = raw_chunk[-overlap_samples:].reshape(-1, 1)
                        buffer           = [overlap_frame]
                        buffered_samples = len(overlap_frame)
                    else:
                        buffer           = []
                        buffered_samples = 0
                    merge_pending = None  # speech paused — reset merge buffer
                    continue

                chunk = clean_chunk
                chunks_processed += 1

                # ── Queue status ─────────────────────────────────────────
                depth   = self._ai_queue.qsize()
                q_ratio = depth / AI_QUEUE_MAXSIZE

                # ── SECTION 2: Backpressure control ─────────────────────
                # Queue > 80% → do NOT enqueue immediately.
                # Merge incoming chunk with the pending merge buffer so we
                # keep accumulating audio without stalling or dropping.
                if q_ratio >= BACKPRESSURE_THRESHOLD:
                    if merge_pending is not None:
                        merge_pending = np.concatenate([merge_pending, chunk])
                    else:
                        merge_pending = chunk
                    logger.warning(
                        f"Backpressure — queue {depth}/{AI_QUEUE_MAXSIZE} "
                        f"({q_ratio * 100:.0f}%) — merging chunk "
                        f"(total merged: {len(merge_pending)/SAMPLE_RATE:.2f}s)"
                    )
                    # Carry overlap normally
                    if overlap_samples > 0 and len(raw_chunk) > overlap_samples:
                        overlap_frame    = raw_chunk[-overlap_samples:].reshape(-1, 1)
                        buffer           = [overlap_frame]
                        buffered_samples = len(overlap_frame)
                    else:
                        buffer           = []
                        buffered_samples = 0
                    continue

                # Queue is below threshold — enqueue (flushing any merged audio first)
                if merge_pending is not None:
                    # Flush the accumulated merged chunk first
                    combined = np.concatenate([merge_pending, chunk])
                    merge_pending = None
                    self._safe_enqueue(combined, depth, q_ratio)
                else:
                    self._safe_enqueue(chunk, depth, q_ratio)

                # ── Auto mode switching ──────────────────────────────────
                new_mode = current_mode
                if q_ratio >= MODE_SWITCH_TO_ACCURACY_THRESHOLD:
                    target_s = ACCURACY_CHUNK_DURATION_S
                    new_mode = "ACCURACY"
                elif q_ratio >= MODE_SWITCH_TO_BALANCED_THRESHOLD:
                    target_s = BALANCED_CHUNK_DURATION_S
                    new_mode = "BALANCED"
                elif q_ratio <= MODE_SWITCH_TO_ULTRA_THRESHOLD:
                    target_s = ULTRA_CHUNK_DURATION_S
                    new_mode = "ULTRA_REALTIME"
                else:
                    t_val = (q_ratio - ADAPTIVE_LOW_THRESHOLD) / (
                        ADAPTIVE_HIGH_THRESHOLD - ADAPTIVE_LOW_THRESHOLD
                    )
                    t_val    = max(0.0, min(1.0, t_val))
                    target_s = CHUNK_MIN_S + t_val * (CHUNK_MAX_S - CHUNK_MIN_S)
                    target_s = round(target_s, 2)
                    new_mode = "ADAPTIVE"

                # EMA smoothing to avoid abrupt chunk-size jumps
                alpha = 0.3
                new_s = current_chunk_s * (1 - alpha) + target_s * alpha
                new_s = round(max(CHUNK_MIN_S, min(CHUNK_MAX_S, new_s)), 2)

                if new_mode != current_mode:
                    elapsed = time.monotonic() - last_mode_switch
                    logger.info(
                        f"[PIPELINE] Mode switch: {current_mode} → {new_mode} | "
                        f"chunk: {current_chunk_s:.2f}s → {new_s:.2f}s | "
                        f"queue: {q_ratio * 100:.0f}% | "
                        f"held {current_mode} for {elapsed:.1f}s"
                    )
                    current_mode = new_mode
                    last_mode_switch = time.monotonic()

                if abs(new_s - current_chunk_s) > 0.02:
                    current_chunk_s = new_s
                    chunk_samples   = int(self.native_rate * current_chunk_s)

                proc_ms = (time.perf_counter() - chunk_start_t) * 1000
                if chunks_processed % 10 == 0:
                    logger.debug(
                        f"[PIPELINE] chunk #{chunks_processed} | "
                        f"mode={current_mode} | chunk={current_chunk_s:.2f}s | "
                        f"queue={q_ratio * 100:.0f}% | proc={proc_ms:.0f}ms"
                    )

                # ── 200 ms overlap into next chunk ───────────────────────
                if overlap_samples > 0 and len(raw_chunk) > overlap_samples:
                    overlap_frame    = raw_chunk[-overlap_samples:].reshape(-1, 1)
                    buffer           = [overlap_frame]
                    buffered_samples = len(overlap_frame)
                else:
                    buffer           = []
                    buffered_samples = 0

            except Exception as exc:
                logger.error(f"ChunkAggregator error: {exc}")

    def _safe_enqueue(self, chunk: np.ndarray, depth: int, q_ratio: float) -> None:
        """Enqueue a chunk. Last-resort: if still full drop oldest (should be rare)."""
        if self._ai_queue.full():
            try:
                self._ai_queue.get_nowait()
                logger.warning("Queue full — dropped oldest chunk (last resort).")
            except queue.Empty:
                pass
        try:
            self._ai_queue.put_nowait(chunk)
        except queue.Full:
            logger.warning("Queue put failed — chunk discarded.")

    # ------------------------------------------------------------------
    # Whisper Worker Pool
    # ------------------------------------------------------------------
    def _ai_dispatch_worker(self, worker_id: int) -> None:
        """Pull audio chunks from the shared queue and invoke Whisper callback.

        Two workers run in parallel (round-robin by queue consumption).
        Each worker processes one 2.4s window at a time; while one waits
        for GPU inference, the other can begin preprocessing the next window.
        """
        import time as _t
        while self.is_running:
            try:
                chunk = self._ai_queue.get(timeout=0.2)
                if chunk is None:
                    break
                t0 = _t.perf_counter()
                self.callback(chunk)
                elapsed = (_t.perf_counter() - t0) * 1000
                logger.debug(
                    f"[WhisperWorker-{worker_id}] chunk processed in {elapsed:.0f}ms "
                    f"| queue_depth={self._ai_queue.qsize()}"
                )
            except queue.Empty:
                continue
            except Exception as exc:
                logger.error(f"WhisperWorker-{worker_id} error: {exc}")
                # Don't break — keep worker alive to process next chunk
                import time; time.sleep(0.1)

    # ------------------------------------------------------------------
    # Queue depth monitor
    # ------------------------------------------------------------------
    def _queue_monitor_worker(self, label: str) -> None:
        while self.is_running:
            depth = self._ai_queue.qsize()
            pct   = depth / AI_QUEUE_MAXSIZE * 100
            ring  = self._ring_buffer.size

            if depth >= _BACKPRESSURE_DEPTH:
                interval, log = MONITOR_CRITICAL_S, logger.warning
            elif depth >= _WARN_DEPTH:
                interval, log  = MONITOR_WARN_S, logger.warning
            else:
                interval, log  = MONITOR_IDLE_S, logger.info

            q_ratio = depth / AI_QUEUE_MAXSIZE
            if q_ratio >= MODE_SWITCH_TO_ACCURACY_THRESHOLD:
                mode_label = "ACCURACY"
            elif q_ratio >= MODE_SWITCH_TO_BALANCED_THRESHOLD:
                mode_label = "BALANCED"
            elif q_ratio <= MODE_SWITCH_TO_ULTRA_THRESHOLD:
                mode_label = "ULTRA"
            else:
                mode_label = "ADAPTIVE"

            log(
                f"[PIPELINE:{label}] queue: {depth}/{AI_QUEUE_MAXSIZE} ({pct:.0f}%) | "
                f"ring_buf: {ring} | mode: {mode_label}"
            )
            time.sleep(interval)

    # ------------------------------------------------------------------
    # Launch all background threads
    # ------------------------------------------------------------------
    def _start_workers(self, chunk_duration_s: float, label: str = "Mic") -> None:
        self._processing_thread = threading.Thread(
            target=self._processing_worker,
            args=(chunk_duration_s,),
            daemon=True,
            name=f"{label}ChunkAggregator",
        )
        self._processing_thread.start()

        for i in range(AI_WORKER_COUNT):
            t = threading.Thread(
                target=self._ai_dispatch_worker,
                args=(i,),
                daemon=True,
                name=f"{label}WhisperWorker-{i}",
            )
            t.start()
            self._dispatch_threads.append(t)

        self._monitor_thread = threading.Thread(
            target=self._queue_monitor_worker,
            args=(label,),
            daemon=True,
            name=f"{label}QueueMonitor",
        )
        self._monitor_thread.start()


# ---------------------------------------------------------------------------
# MicrophoneCapture
# ---------------------------------------------------------------------------
class MicrophoneCapture(BaseCapture):
    """Microphone input via sounddevice at 16 kHz mono.

    Opens at SAMPLE_RATE (16 kHz) directly — no resampling overhead.
    blocksize=3200 gives 200 ms frames at 16 kHz (target capture frame size).
    """

    _AUDIO_CONFIG = dict(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=3200,           # 200 ms at 16 kHz — aligns with CAPTURE_FRAME_S
    )

    def __init__(
        self,
        device_index: Optional[int] = None,
        callback: Optional[Callable] = None,
        noise_gate: float = 0.005,
    ) -> None:
        super().__init__(callback, noise_gate)
        self.device_index = device_index

    def start(self, chunk_duration_s: float = CHUNK_DURATION_S) -> None:
        if self.is_running:
            return
        try:
            self.native_rate = SAMPLE_RATE

            self._stream = sd.InputStream(
                device=self.device_index,
                callback=self._audio_callback,
                **self._AUDIO_CONFIG,
            )
            self.is_running = True
            self._stream.start()
            self._start_workers(chunk_duration_s, label="Mic")

            logger.info(
                f"Microphone capture started | device={self.device_index} | "
                f"rate={self.native_rate} Hz | blocksize=3200 (200ms) | "
                f"chunk={chunk_duration_s}s | NO resampling needed"
            )
        except Exception as exc:
            logger.error(f"Failed to start MicrophoneCapture: {exc}")
            raise


# ---------------------------------------------------------------------------
# SystemAudioCapture  (WASAPI Loopback via PyAudioWPatch)
# ---------------------------------------------------------------------------
class SystemAudioCapture(BaseCapture):
    """Windows WASAPI Loopback capture — extends BaseCapture.

    Captures system audio at native rate (typically 48000 Hz).
    The preprocessor handles resampling 48k→16k via scipy polyphase filter.
    Buffer is 9600 samples (~200 ms at 48 kHz) to match CAPTURE_FRAME_S.
    """

    def __init__(
        self,
        callback: Optional[Callable] = None,
        noise_gate: float = 0.001,
    ) -> None:
        super().__init__(callback, noise_gate)
        self.native_rate = 48000
        self._pyaudio    = None

    def start(self, chunk_duration_s: float = CHUNK_DURATION_S) -> None:
        if self.is_running:
            return
        try:
            import pyaudiowpatch as pyaudio

            p = pyaudio.PyAudio()

            try:
                wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
            except OSError:
                raise RuntimeError("WASAPI not available on this system.")

            default_speakers = p.get_device_info_by_index(
                wasapi_info["defaultOutputDevice"]
            )

            loopback_device = None
            for i in range(p.get_device_count()):
                dev = p.get_device_info_by_index(i)
                if dev.get("isLoopbackDevice", False):
                    if default_speakers["name"] in dev["name"]:
                        loopback_device = dev
                        break

            if loopback_device is None:
                for i in range(p.get_device_count()):
                    dev = p.get_device_info_by_index(i)
                    if dev.get("isLoopbackDevice", False):
                        loopback_device = dev
                        break

            if loopback_device is None:
                p.terminate()
                raise RuntimeError("No WASAPI Loopback device found.")

            self.native_rate = int(loopback_device["defaultSampleRate"])
            channels         = loopback_device["maxInputChannels"]

            logger.info(
                f"WASAPI Loopback: '{loopback_device['name']}' | "
                f"{self.native_rate} Hz | {channels} ch"
            )

            # 200 ms frames at native rate: ~9600 samples at 48 kHz
            frames_per_buffer = int(self.native_rate * CAPTURE_FRAME_S)

            def _pyaudio_callback(in_data, frame_count, time_info, flags):
                if self.is_running:
                    audio = np.frombuffer(in_data, dtype=np.float32)
                    if channels >= 2:
                        audio = audio.reshape(-1, channels).mean(axis=1)
                    self._ring_buffer.put(audio.reshape(-1, 1))
                return (None, pyaudio.paContinue)

            self._pyaudio = p
            self._pa_stream = p.open(
                format=pyaudio.paFloat32,
                channels=channels,
                rate=self.native_rate,
                input=True,
                input_device_index=loopback_device["index"],
                frames_per_buffer=frames_per_buffer,
                stream_callback=_pyaudio_callback,
            )

            self.is_running = True
            self._pa_stream.start_stream()
            self._start_workers(chunk_duration_s, label="Sys")

            logger.info(
                f"WASAPI Loopback ACTIVE | buffer={frames_per_buffer} samples (~200ms) | "
                f"chunk={chunk_duration_s}s | "
                f"native={self.native_rate} Hz → resample to {SAMPLE_RATE} Hz"
            )

        except ImportError:
            logger.error("pyaudiowpatch not installed — run: pip install PyAudioWPatch")
            raise
        except Exception as exc:
            logger.error(f"SystemAudioCapture start failed: {exc}")
            raise

    def stop(self) -> None:
        if hasattr(self, "_pa_stream") and self._pa_stream:
            try:
                self._pa_stream.stop_stream()
                self._pa_stream.close()
            except Exception:
                pass
            self._pa_stream = None

        if self._pyaudio:
            try:
                self._pyaudio.terminate()
            except Exception:
                pass
            self._pyaudio = None

        super().stop()


# ---------------------------------------------------------------------------
# AudioEngine  (compatibility wrapper)
# ---------------------------------------------------------------------------
class AudioEngine:
    """Wrapper for managing dual capture sources."""

    def __init__(self) -> None:
        self.mic: Optional[MicrophoneCapture]  = None
        self.sys: Optional[SystemAudioCapture] = None

    @staticmethod
    def list_devices() -> list:
        return [dict(d, index=i) for i, d in enumerate(sd.query_devices())]

    def stop_all(self) -> None:
        if self.mic:
            self.mic.stop()
        if self.sys:
            self.sys.stop()
        self.mic = None
        self.sys = None
