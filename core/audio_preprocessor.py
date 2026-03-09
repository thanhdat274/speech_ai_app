"""
audio_preprocessor.py
=============================================================================
Ultra Low-Latency Audio Preprocessing — Pure NumPy (no torch/torchaudio)

Two preprocessing paths:

  STANDARD PATH  (ultra_fast=False, default)
  ─────────────────────────────────────────
    input_audio
        ↓  Stereo → Mono          (vectorized mean, ~0.1 ms)
        ↓  Resample → 16 kHz      (scipy.signal.resample_poly, SKIPPED if 16kHz)
        ↓  DC offset removal      (~0.1 ms)
        ↓  Noise gate             (RMS silence detection, ~0.1 ms)
        ↓  RMS normalization      (~0.1 ms)
        ↓  Clip to [-1, 1]
        →  ChunkAggregator / AI engine
    Total: < 5 ms with resample, < 1 ms without

  FAST PATH  (ultra_fast=True — ULTRA_REALTIME mode only)
  ───────────────────────────────────────────────────────
    input_audio
        ↓  Stereo → Mono  (mean axis=1)
        ↓  Resample → 16 kHz  (scipy resample_poly, SKIPPED if 16kHz)
        ↓  float32 cast
        →  AI engine   (no DC removal, no noise gate, no RMS normalization)
    Total: < 10 ms with resample, ~0.5 ms without

SKIPPED in ultra_fast mode:
    - DC offset removal
    - RMS normalization
    - Noise gate (silence threshold)
    - Any noise reduction or spectral gating

No torch, no torchaudio, no librosa, no spectrograms.
All operations are vectorized NumPy with pre-cached filter coefficients.
=============================================================================
"""

import logging
import threading
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
TARGET_SR            = 16_000   # Whisper requires 16 kHz mono
TARGET_RMS           = 0.1      # RMS normalization target (~-20 dBFS)
NOISE_GATE_RMS       = 0.003    # Silence threshold — below this is silence
# v8: tightened latency budgets
LATENCY_WARN_MS      = 30    # Standard path budget (ms)  — was 80
FAST_PATH_WARN_MS    = 5     # Ultra-fast path budget (ms) — was 30


class AudioPreprocessor:
    """Thread-safe, ultra-low-latency audio preprocessor — pure NumPy.

    Singleton pattern — use AudioPreprocessor.get_instance().

    Usage::

        prep = AudioPreprocessor.get_instance()

        # Standard path (default) — full quality, silence detection:
        clean = prep.preprocess(raw_audio, source_sr=48000)

        # Ultra-fast path — skips DC/RMS/noise gate, ~5-10ms:
        clean = prep.preprocess(raw_audio, source_sr=48000, ultra_fast=True)

    Standard path:
        1. float32 cast + ensure C-contiguous
        2. Stereo → Mono (channel mean)
        3. Resample to 16 kHz (SKIPPED if already 16 kHz)
        4. DC offset removal (subtract mean)
        5. Noise gate (silence detection via RMS threshold)
        6. RMS normalization
        7. Clip to [-1, 1]

    Ultra-fast path (ULTRA_REALTIME mode only):
        1. Stereo → Mono (mean axis)
        2. Resample to 16 kHz (scipy polyphase, SKIPPED if 16 kHz)
        3. float32 cast
        Skips: DC removal, noise gate, RMS normalization

    Latency budget (1 s chunk @ 48 kHz stereo → 16 kHz mono):
        Standard:     < 5 ms  (with resample) / < 1 ms (16 kHz mic)
        Ultra-fast:   < 10 ms (with resample) / < 0.5 ms (16 kHz mic)
    """

    _instance: Optional["AudioPreprocessor"] = None
    _init_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "AudioPreprocessor":
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    def __init__(self) -> None:
        self._ratio_cache: dict = {}

        # Pre-import scipy in background to avoid cold-start on first standard-path chunk
        # ultra_fast path does NOT use scipy (stride downsampling only)
        self._resample_fn = None
        self._resample_ready = False
        self._import_thread = threading.Thread(
            target=self._ensure_resample, daemon=True, name="SciPyLoader"
        )
        self._import_thread.start()

        # Stride lookup: exact integer ratios for common hardware rates
        # key = source_sr, value = stride step (source_sr / TARGET_SR)
        self._stride_map: dict = {}
        for sr in (48000, 44100, 32000, 24000, 22050):
            step = sr / TARGET_SR
            if abs(step - round(step)) < 0.01:  # within 1% of integer
                self._stride_map[sr] = int(round(step))
        # 44100 approx: 44100/16000 = 2.756 — not integer, falls back to scipy

        logger.info(
            f"AudioPreprocessor ready | stride_map={self._stride_map} | "
            f"target_sr={TARGET_SR} Hz"
        )

    # ------------------------------------------------------------------
    # Lazy scipy import + ratio cache
    # ------------------------------------------------------------------
    def _ensure_resample(self) -> None:
        """Import scipy.signal.resample_poly once on first use."""
        if self._resample_ready:
            return
        try:
            from scipy.signal import resample_poly
            self._resample_fn = resample_poly
            self._resample_ready = True
            logger.info("scipy.signal.resample_poly loaded for resampling")
        except ImportError:
            self._resample_fn = None
            self._resample_ready = True
            logger.warning(
                "scipy not available — using numpy interp for resampling "
                "(install scipy for better quality: pip install scipy)"
            )

    def _get_ratio(self, source_sr: int) -> tuple:
        """Return (up, down) factors for resample_poly, cached by source_sr."""
        cached = self._ratio_cache.get(source_sr)
        if cached is not None:
            return cached

        from math import gcd
        g = gcd(TARGET_SR, source_sr)
        up, down = TARGET_SR // g, source_sr // g
        self._ratio_cache[source_sr] = (up, down)
        logger.info(f"Resample ratio cached: {source_sr} → {TARGET_SR} Hz (up={up}, down={down})")
        return up, down

    def _stride_downsample(self, audio: np.ndarray, source_sr: int) -> np.ndarray:
        """Integer-stride downsampling — the fastest possible resampler.

        For 48 kHz → 16 kHz: step=3 → take every 3rd sample.
        Latency: ~0.1 ms for 1.2 s @48kHz (pure NumPy slice, no allocation).

        Falls back to scipy/interp for non-integer ratios (e.g. 44.1 kHz).
        """
        if source_sr == TARGET_SR:
            return audio

        step = self._stride_map.get(source_sr)
        if step is not None:
            # Pure NumPy stride — zero-copy when possible
            return np.ascontiguousarray(audio[::step], dtype=np.float32)
        else:
            # Non-integer ratio: fall back to scipy or linear interp
            return self._resample(audio, source_sr)

    # ------------------------------------------------------------------
    # Resample (only when source_sr != TARGET_SR)
    # ------------------------------------------------------------------
    def _resample(self, audio: np.ndarray, source_sr: int) -> np.ndarray:
        """Resample audio from source_sr to TARGET_SR using scipy or linear interp."""
        if source_sr == TARGET_SR:
            return audio

        if not self._resample_ready and self._import_thread.is_alive():
            self._import_thread.join(timeout=2.0)
        if not self._resample_ready:
            self._ensure_resample()

        if self._resample_fn is not None:
            up, down = self._get_ratio(source_sr)
            return self._resample_fn(audio, up, down).astype(np.float32)
        else:
            n_out = int(len(audio) * TARGET_SR / source_sr)
            indices = np.linspace(0, len(audio) - 1, n_out)
            return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def preprocess(
        self,
        audio: np.ndarray,
        source_sr: int = TARGET_SR,
        channels: int = 1,
        noise_reduce: bool = False,   # kept for API compat — ignored
        ultra_fast: bool = False,
    ) -> np.ndarray:
        """Preprocess raw audio for Whisper inference.

        Args:
            audio:       Raw numpy array, shape (N,), (N,C) or (C,N), any dtype.
            source_sr:   Input sample rate. Skips resampling when == TARGET_SR.
            channels:    Hint — actual shape is auto-detected.
            noise_reduce: Ignored. Kept for API compatibility.
            ultra_fast:  When True, uses the minimal fast path:
                         stereo→mono + resample only. Skips DC removal,
                         noise gate, and RMS normalization.
                         Latency: ~5-10 ms vs ~1-5 ms for standard path.
                         Use in ULTRA_REALTIME mode.

        Returns:
            float32 numpy array at 16 kHz, mono.
            Standard path: RMS-normalised; empty array if pure silence.
            Ultra-fast path: raw amplitude; never empty (no silence gate).
        """
        start = time.perf_counter()

        # ══════════════════════════════════════════════════════════════════
        # ULTRA-FAST PATH — ULTRA_REALTIME mode
        # Skips: DC removal, noise gate, RMS normalization, heavy filtering.
        # Only does: stereo→mono, resample to 16 kHz, float32 cast.
        # Target latency: < 10 ms (< 0.5 ms when source is already 16 kHz).
        # ══════════════════════════════════════════════════════════════════
        if ultra_fast:
            t0 = time.perf_counter()

            # Step 1: Stereo → Mono
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            t1 = time.perf_counter()

            # Step 2: Resample — STRIDE downsampling (pure NumPy, < 0.1 ms)
            # Avoids scipy entirely; only falls back for non-integer ratios.
            if source_sr != TARGET_SR:
                audio = self._stride_downsample(audio, source_sr)

            # Step 3: Ensure float32 contiguous
            result = np.ascontiguousarray(audio, dtype=np.float32)

            latency = (time.perf_counter() - t0) * 1000
            resample_ms = (time.perf_counter() - t1) * 1000
            if latency > FAST_PATH_WARN_MS:
                logger.warning(
                    f"[Preprocess:FAST] {latency:.2f}ms > {FAST_PATH_WARN_MS}ms budget "
                    f"| stereo→mono={( t1-t0)*1000:.2f}ms resample={resample_ms:.2f}ms"
                )
            else:
                logger.debug(
                    f"[Preprocess:FAST] {latency:.2f}ms "
                    f"| sr={source_sr} stride={self._stride_map.get(source_sr, 'scipy')}"
                )
            return result

        # ══════════════════════════════════════════════════════════════════
        # STANDARD PATH — balanced latency + quality
        # ══════════════════════════════════════════════════════════════════

        # ── Step 0: Ensure float32 contiguous ──────────────────────────
        t_cast = time.perf_counter()
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32, copy=False)
        if not audio.flags["C_CONTIGUOUS"]:
            audio = np.ascontiguousarray(audio)
        cast_ms = (time.perf_counter() - t_cast) * 1000

        # ── Step 1: Stereo → Mono ──────────────────────────────────
        t_mono = time.perf_counter()
        if audio.ndim == 2:
            if audio.shape[1] > 1:
                audio = audio.mean(axis=1)
            else:
                audio = audio.ravel()
        elif audio.ndim > 2:
            audio = audio.ravel()

        if audio.ndim != 1:
            audio = audio.ravel()
        mono_ms = (time.perf_counter() - t_mono) * 1000

        # ── Step 2: Resample to 16 kHz (SKIPPED if already 16 kHz) ────
        t_resample = time.perf_counter()
        if source_sr != TARGET_SR:
            audio = self._resample(audio, source_sr)
        resample_ms = (time.perf_counter() - t_resample) * 1000

        # ── Step 3: DC offset removal ────────────────────────────────
        t_dc = time.perf_counter()
        audio = audio - np.mean(audio)
        dc_ms = (time.perf_counter() - t_dc) * 1000

        # ── Step 4: Noise gate (silence detection) ─────────────────────
        t_gate = time.perf_counter()
        rms = float(np.sqrt(np.mean(audio * audio)))
        if rms < NOISE_GATE_RMS:
            latency = (time.perf_counter() - start) * 1000
            logger.debug(f"[Preprocess:STD] silence | rms={rms:.5f} | {latency:.1f}ms")
            return np.array([], dtype=np.float32)
        gate_ms = (time.perf_counter() - t_gate) * 1000

        # ── Step 5: RMS normalization ──────────────────────────────────
        t_norm = time.perf_counter()
        scale = TARGET_RMS / rms
        scale = min(scale, 10.0)        # clamp — don't over-amplify noise
        audio = audio * scale
        np.clip(audio, -1.0, 1.0, out=audio)
        norm_ms = (time.perf_counter() - t_norm) * 1000

        latency = (time.perf_counter() - start) * 1000
        if latency > 30: # LATENCY_WARN_MS
            logger.warning(
                f"[Preprocess:STD] {latency:.2f}ms > 30ms budget " # LATENCY_WARN_MS
                f"| cast={cast_ms:.1f}ms mono={mono_ms:.1f}ms resample={resample_ms:.1f}ms "
                f"dc={dc_ms:.1f}ms gate={gate_ms:.1f}ms norm={norm_ms:.1f}ms rms={rms:.4f}"
            )
        else:
            logger.debug(
                f"[Preprocess:STD] {latency:.2f}ms "
                f"| cast={cast_ms:.1f}ms mono={mono_ms:.1f}ms resample={resample_ms:.1f}ms "
                f"dc={dc_ms:.1f}ms gate={gate_ms:.1f}ms norm={norm_ms:.1f}ms rms={rms:.4f} scale={scale:.2f}"
            )

        return audio
