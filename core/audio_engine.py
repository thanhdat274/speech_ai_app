"""
audio_engine.py
=============================================================================
AudioEngine Module — Speech-AI v5 Architecture
=============================================================================

Pipeline:

  [Hardware Callback]
       |  never blocks — puts raw float32 frames into RingBuffer
       v
  [RingBuffer]   circular, overwrites oldest on overflow; capture never stalls
       |
       v  (ChunkAggregator thread)
       |  - accumulates frames to an adaptive window (1.5–2.5 s)
       |  - resamples to 16 kHz for Whisper
       |  - Silero VAD gate — neural speech detection
       |    (automatic RMS energy fallback when Silero is unavailable)
       |
       v  (ready ~2.5 s speech chunks with 200 ms overlap)
  [_ai_queue]   small bounded work queue drained by the worker pool
       |
       v  (Whisper Worker Pool — 5 threads share _ai_queue)
  user callback  (StreamingTranscriber.process_chunk → Whisper)
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

# ── Latency Modes ────────────────────────────────────────────────────────────
#
#   Pipeline:  Audio Capture → Ring Buffer → Dynamic Chunk Builder
#              → Inference Queue → AI Engine → Subtitle Smoother
#
# ULTRA_REALTIME: chunk=0.3s — fastest response, minimal context
# BALANCED:       chunk=0.7s — good balance of latency + Vietnamese accuracy
# ACCURACY:       chunk=1.5s — maximum Vietnamese tone context for YouTube
ULTRA_CHUNK_DURATION_S = 0.25
BALANCED_CHUNK_DURATION_S = 0.5
ACCURACY_CHUNK_DURATION_S = 1.0
CHUNK_DURATION_S = BALANCED_CHUNK_DURATION_S  # exported default for start()

# Overlap: 0.3s between consecutive chunks preserves cross-chunk Vietnamese tones.
OVERLAP_DURATION_S = 0.3

RING_BUFFER_MAXFRAMES = 800      # ~20 s capacity at a 40 fps hardware callback
AI_QUEUE_MAXSIZE = 200            # Large queue — absorbs inference spikes

BACKPRESSURE_THRESHOLD = 0.80
BACKPRESSURE_SLEEP_S   = 0.05

AI_WORKER_COUNT = 3              # Fewer workers = less GPU contention on 4GB cards

MONITOR_IDLE_S     = 5.0
MONITOR_WARN_S     = 1.0
MONITOR_CRITICAL_S = 0.5

VAD_RMS_THRESHOLD = 0.002

# ── Adaptive chunk sizing ────────────────────────────────────────────────────
#   queue > 70% → grow chunk toward CHUNK_MAX_S (fewer, larger chunks)
#   queue < 30% → shrink chunk toward CHUNK_MIN_S (lower latency)
CHUNK_MIN_S = ULTRA_CHUNK_DURATION_S    # 0.3s floor
CHUNK_MAX_S = ACCURACY_CHUNK_DURATION_S # 1.5s ceiling
ADAPTIVE_HIGH_THRESHOLD = 0.70
ADAPTIVE_LOW_THRESHOLD  = 0.30

# ── Automatic mode switching (based on queue depth) ──────────────────────────
#   queue > 90% → switch to ACCURACY (1.5s) — reduce inference load heavily
#   queue > 80% → switch to BALANCED (0.7s) — moderate load reduction
#   queue < 30% → switch to ULTRA_REALTIME (0.3s) — lowest latency
MODE_SWITCH_TO_ACCURACY_THRESHOLD = 0.90
MODE_SWITCH_TO_BALANCED_THRESHOLD = 0.80
MODE_SWITCH_TO_ULTRA_THRESHOLD    = 0.30

# Derived depth thresholds (computed once, referenced in hot loops)
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
        """Return the oldest frame, blocking up to *timeout* seconds.

        Returns None on timeout — mirrors queue.Queue.get() semantics.
        """
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
    """Silero VAD wrapper with automatic RMS energy fallback.

    The model is downloaded once from torch.hub on the first is_speech() call
    and cached as a process-level singleton.  If the hub is unreachable (offline
    environment, incompatible PyTorch version, etc.) every subsequent call
    silently falls back to RMS energy gating — callers see no difference.

    A per-instance inference lock serialises concurrent calls because the
    underlying PyTorch model is not thread-safe.
    """

    def __init__(self) -> None:
        self._model = None
        self._get_ts: Optional[Callable] = None   # get_speech_timestamps fn
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
                self._get_ts = utils[0]   # get_speech_timestamps is first element
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
            # RMS fallback: require both RMS energy and peak above the gate
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
            # Any inference failure → safe RMS fallback so capture is never stalled
            return float(np.sqrt(np.mean(audio_16k ** 2))) >= noise_gate


# Module-level singleton shared across all capture sources
_vad = _SileroVAD()


# ---------------------------------------------------------------------------
# BaseCapture
# ---------------------------------------------------------------------------
class BaseCapture(ABC):
    """Abstract base for audio capture sources.

    Subclasses open a hardware stream and call _start_workers() once ready.
    The RingBuffer → ChunkAggregator → WorkerPool pipeline ensures the hardware
    callback is never delayed by AI inference or VAD processing.
    """

    def __init__(
        self,
        callback: Callable[[np.ndarray], None],
        noise_gate: float = 0.001,
    ) -> None:
        self.callback = callback
        self.noise_gate = noise_gate
        self.is_running = False
        self._stream = None           # sounddevice or PyAudio stream
        self.native_rate: int = SAMPLE_RATE

        # Stage 1: circular buffer — hardware callback writes here, never blocks
        self._ring_buffer = RingBuffer()

        # Stage 2: VAD-passed chunks waiting for the Whisper worker pool
        self._ai_queue: queue.Queue = queue.Queue(maxsize=AI_QUEUE_MAXSIZE)

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

        # Unblock the ChunkAggregator thread immediately
        self._ring_buffer.unblock()
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=2.0)

        # Send one sentinel per worker thread so each unblocks cleanly
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
            # copy() is mandatory — sounddevice reuses the buffer immediately
            self._ring_buffer.put(indata.copy())

    # ------------------------------------------------------------------
    # Dynamic Chunk Builder with Automatic Mode Switching:
    #   Ring Buffer → preprocess → adaptive sizing → inference queue
    #
    # Overlap rule (0.3s):
    #   chunk_1 = 0.0 → 0.7
    #   chunk_2 = 0.4 → 1.1   (0.3s of chunk_1's tail carried over)
    #
    # Auto mode switching:
    #   queue > 90% → ACCURACY mode (1.5s chunks)
    #   queue > 80% → BALANCED mode (0.7s chunks)
    #   queue < 30% → ULTRA_REALTIME mode (0.3s chunks)
    # ------------------------------------------------------------------
    def _processing_worker(self, chunk_duration_s: float) -> None:
        """Accumulate raw frames into overlapping speech windows.

        Dynamic chunk sizing with automatic mode switching:
          queue > 90% → ACCURACY mode (1.5s, max context, fewest calls)
          queue > 80% → BALANCED mode (0.7s, moderate latency)
          queue < 30% → ULTRA_REALTIME mode (0.3s, lowest latency)
          30-80%      → linear interpolation between min and max
        """
        from core.audio_preprocessor import AudioPreprocessor
        preprocessor = AudioPreprocessor.get_instance()

        buffer: List[np.ndarray] = []
        buffered_samples = 0
        current_chunk_s = chunk_duration_s
        chunk_samples   = int(self.native_rate * current_chunk_s)
        overlap_samples = int(self.native_rate * OVERLAP_DURATION_S)

        # Mode tracking for logging
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

                # ── Build raw chunk ──────────────────────────────────
                raw_chunk = np.concatenate(buffer, axis=0).flatten()

                # ── Preprocess: mono → resample → normalize ─────────
                clean_chunk = preprocessor.preprocess(
                    raw_chunk,
                    source_sr=self.native_rate,
                )

                # Empty = silence → keep overlap, skip inference
                if len(clean_chunk) == 0:
                    if overlap_samples > 0 and len(raw_chunk) > overlap_samples:
                        overlap_frame    = raw_chunk[-overlap_samples:].reshape(-1, 1)
                        buffer           = [overlap_frame]
                        buffered_samples = len(overlap_frame)
                    else:
                        buffer           = []
                        buffered_samples = 0
                    continue

                chunk = clean_chunk
                chunks_processed += 1

                # ── Queue status ─────────────────────────────────────
                depth = self._ai_queue.qsize()
                q_ratio = depth / AI_QUEUE_MAXSIZE

                # ── Backpressure (progressive sleep) ─────────────────
                if q_ratio >= BACKPRESSURE_THRESHOLD:
                    sleep_ms = BACKPRESSURE_SLEEP_S * 1000
                    if q_ratio >= 0.95:
                        sleep_ms *= 3   # 150ms sleep at 95%+
                    elif q_ratio >= 0.90:
                        sleep_ms *= 2   # 100ms sleep at 90%+
                    logger.warning(
                        f"Backpressure — queue {depth}/{AI_QUEUE_MAXSIZE} "
                        f"({q_ratio * 100:.0f}%). Pausing {sleep_ms:.0f} ms."
                    )
                    time.sleep(sleep_ms / 1000)

                # Drop oldest when still full (last resort)
                if self._ai_queue.full():
                    try:
                        self._ai_queue.get_nowait()
                        logger.warning("Queue full — dropped oldest chunk.")
                    except queue.Empty:
                        pass
                self._ai_queue.put_nowait(chunk)

                # ── Automatic mode switching ─────────────────────────
                # Higher queue = bigger chunks = fewer inference calls
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
                    # Linear interpolation between min and max
                    t = (q_ratio - ADAPTIVE_LOW_THRESHOLD) / (
                        ADAPTIVE_HIGH_THRESHOLD - ADAPTIVE_LOW_THRESHOLD
                    )
                    t = max(0.0, min(1.0, t))  # clamp to [0, 1]
                    target_s = CHUNK_MIN_S + t * (CHUNK_MAX_S - CHUNK_MIN_S)
                    target_s = round(target_s, 2)
                    new_mode = "ADAPTIVE"

                # Smooth transition — exponential moving average (avoid jumps)
                alpha = 0.3  # smoothing factor: 0.3 = gradual, 1.0 = instant
                new_s = current_chunk_s * (1 - alpha) + target_s * alpha
                new_s = round(max(CHUNK_MIN_S, min(CHUNK_MAX_S, new_s)), 2)

                # Log mode transitions
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

                # ── Processing time logging ──────────────────────────
                proc_ms = (time.perf_counter() - chunk_start_t) * 1000
                if chunks_processed % 10 == 0:  # Log every 10th chunk
                    logger.debug(
                        f"[PIPELINE] chunk #{chunks_processed} | "
                        f"mode={current_mode} | chunk={current_chunk_s:.2f}s | "
                        f"queue={q_ratio * 100:.0f}% | proc={proc_ms:.0f}ms"
                    )

                # ── Carry 0.3s overlap into next chunk ───────────────
                if overlap_samples > 0 and len(raw_chunk) > overlap_samples:
                    overlap_frame    = raw_chunk[-overlap_samples:].reshape(-1, 1)
                    buffer           = [overlap_frame]
                    buffered_samples = len(overlap_frame)
                else:
                    buffer           = []
                    buffered_samples = 0

            except Exception as exc:
                logger.error(f"ChunkAggregator error: {exc}")

    # ------------------------------------------------------------------
    # Whisper Worker Pool — AI_WORKER_COUNT threads share _ai_queue
    # ------------------------------------------------------------------
    def _ai_dispatch_worker(self, worker_id: int) -> None:
        """Pull chunks from the pool queue and invoke the Whisper callback."""
        while self.is_running:
            try:
                chunk = self._ai_queue.get(timeout=0.2)
                if chunk is None:
                    break
                self.callback(chunk)
            except queue.Empty:
                continue
            except Exception as exc:
                logger.error(f"WhisperWorker-{worker_id} error: {exc}")
                break

    # ------------------------------------------------------------------
    # Queue depth monitor with [PIPELINE] logging
    # ------------------------------------------------------------------
    def _queue_monitor_worker(self, label: str) -> None:
        while self.is_running:
            depth = self._ai_queue.qsize()
            pct   = depth / AI_QUEUE_MAXSIZE * 100
            ring  = self._ring_buffer.size

            if depth >= _BACKPRESSURE_DEPTH:
                interval, log = MONITOR_CRITICAL_S, logger.warning
            elif depth >= _WARN_DEPTH:
                interval, log = MONITOR_WARN_S, logger.warning
            else:
                interval, log = MONITOR_IDLE_S, logger.info

            # Determine current effective mode from queue ratio
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

    Opens the hardware stream at SAMPLE_RATE (16 kHz) directly so the
    preprocessor never needs to resample — zero extra latency.
    blocksize=1024 gives ~64 ms frames at 16 kHz for low callback overhead.
    """

    # Audio config — 16 kHz mono float32, small buffer
    _AUDIO_CONFIG = dict(
        samplerate=SAMPLE_RATE,   # 16000 Hz — Whisper native
        channels=CHANNELS,        # 1 (mono)
        dtype="float32",
        blocksize=1024,           # ~64 ms per callback at 16 kHz
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
            # Open at 16 kHz directly — no resampling needed downstream
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
                f"rate={self.native_rate} Hz | blocksize=1024 | "
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

    Captures system audio digitally (identical to how OBS captures audio).
    Works even when speakers are muted.

    IMPORTANT: WASAPI loopback devices MUST be opened at their native sample
    rate (typically 48000 Hz). We cannot force 16 kHz at the hardware level.
    Mono downmix happens in the callback. The preprocessor handles resampling
    48k→16k via cached torchaudio Kaiser filter on CPU (~5 ms).

    Buffer is set to 1024 samples (~21 ms at 48 kHz) for low latency.
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

            # Find the loopback device that mirrors the default speakers
            loopback_device = None
            for i in range(p.get_device_count()):
                dev = p.get_device_info_by_index(i)
                if dev.get("isLoopbackDevice", False):
                    if default_speakers["name"] in dev["name"]:
                        loopback_device = dev
                        break

            # Fallback: use any available loopback device
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

            # Small hardware buffer — 1024 samples for low latency
            # At 48 kHz this is ~21 ms; at 44.1 kHz ~23 ms
            frames_per_buffer = 1024

            def _pyaudio_callback(in_data, frame_count, time_info, flags):
                """PyAudio stream callback — must NEVER block.

                Converts multi-channel float32 to mono in-place.
                """
                if self.is_running:
                    audio = np.frombuffer(in_data, dtype=np.float32)
                    if channels >= 2:
                        # Vectorized mono downmix — no Python loop
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
                f"WASAPI Loopback ACTIVE | buffer=1024 samples | "
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
        # Close the PyAudio stream before delegating to BaseCapture.stop()
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

        super().stop()   # ring_buffer.unblock() + thread joins


# ---------------------------------------------------------------------------
# AudioEngine  (compatibility wrapper)
# ---------------------------------------------------------------------------
class AudioEngine:
    """Wrapper for managing dual capture sources."""

    def __init__(self) -> None:
        self.mic: Optional[MicrophoneCapture]   = None
        self.sys: Optional[SystemAudioCapture]  = None

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
