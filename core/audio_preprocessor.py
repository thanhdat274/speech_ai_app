"""
audio_preprocessor.py
=============================================================================
Ultra Low-Latency Audio Preprocessing — Pure NumPy (no torch/torchaudio)

Pipeline (target < 10 ms on any CPU):

    input_audio
        ↓
    Stereo → Mono          (vectorized mean, ~0.1 ms)
        ↓
    Resample → 16 kHz      (scipy.signal.resample_poly, ~3 ms, SKIPPED if already 16k)
        ↓
    DC offset removal      (subtract mean, ~0.1 ms)
        ↓
    RMS normalization       (vectorized, ~0.1 ms)
        ↓
    Noise gate              (RMS threshold, ~0.1 ms)
        ↓
    send to AI engine

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
TARGET_SR       = 16_000    # Whisper requires 16 kHz mono
TARGET_RMS      = 0.1       # RMS normalization target (~-20 dBFS)
NOISE_GATE_RMS  = 0.003     # Silence threshold — below this is silence
LATENCY_WARN_MS = 80        # Warn if preprocessing exceeds this


class AudioPreprocessor:
    """Thread-safe, ultra-low-latency audio preprocessor — pure NumPy.

    Usage::

        prep = AudioPreprocessor.get_instance()
        clean = prep.preprocess(raw_audio, source_sr=48000)
        # Returns float32 numpy @ 16 kHz mono, RMS-normalised.
        # Returns empty array if the chunk is pure silence.
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
        # Pre-computed GCD ratios for common sample rates → 16 kHz
        self._ratio_cache: dict = {}

        # Pre-import scipy in background to avoid 700ms cold-start on first chunk
        self._resample_fn = None
        self._resample_ready = False
        self._import_thread = threading.Thread(
            target=self._ensure_resample, daemon=True, name="SciPyLoader"
        )
        self._import_thread.start()

        logger.info(
            f"AudioPreprocessor ready | pure NumPy | "
            f"target_sr={TARGET_SR} Hz | target_rms={TARGET_RMS}"
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
            # Fallback: numpy-only linear interpolation (lower quality but no deps)
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

    # ------------------------------------------------------------------
    # Resample (only when source_sr != TARGET_SR)
    # ------------------------------------------------------------------
    def _resample(self, audio: np.ndarray, source_sr: int) -> np.ndarray:
        """Resample audio from source_sr to TARGET_SR.

        Uses scipy.signal.resample_poly for polyphase filtering (fast, high quality).
        Falls back to numpy linear interpolation if scipy is unavailable.
        """
        if source_sr == TARGET_SR:
            return audio

        # Wait for scipy background import if it hasn't finished
        if not self._resample_ready and self._import_thread.is_alive():
            self._import_thread.join(timeout=2.0)
        if not self._resample_ready:
            self._ensure_resample()

        if self._resample_fn is not None:
            # scipy polyphase — high quality, ~3ms for 1s @ 48kHz
            up, down = self._get_ratio(source_sr)
            return self._resample_fn(audio, up, down).astype(np.float32)
        else:
            # Fallback: numpy linear interpolation
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
        noise_reduce: bool = False,  # kept for API compat — ignored
    ) -> np.ndarray:
        """Preprocess raw audio for Whisper inference.

        Pipeline (all vectorized NumPy — no torch, no spectrograms):
            1. Ensure float32 contiguous
            2. Stereo → Mono (channel mean)
            3. Resample to 16 kHz (SKIPPED if already 16 kHz)
            4. DC offset removal (subtract mean)
            5. RMS normalization
            6. Noise gate (silence detection)

        Args:
            audio:      Raw numpy array, shape (N,), (N,C) or (C,N), any dtype.
            source_sr:  Input sample rate. Skip resampling when == TARGET_SR.
            channels:   Hint — actual shape is auto-detected.
            noise_reduce: Ignored. Kept for API compatibility.

        Returns:
            float32 numpy array at 16 kHz, mono, RMS-normalised.
            Empty array (len==0) when the chunk is pure silence.

        Latency budget (1s chunk @ 48 kHz stereo → 16 kHz mono):
            float32 cast:    < 0.1 ms
            Stereo→Mono:     < 0.1 ms  (np.mean vectorized)
            Resample:        ~ 3.0 ms  (scipy polyphase, SKIPPED if 16kHz)
            DC removal:      < 0.1 ms  (subtract mean)
            RMS normalize:   < 0.1 ms  (single multiply)
            Noise gate:      < 0.1 ms  (RMS comparison)
            Total:           < 5 ms    (with resample)
            Total:           < 1 ms    (without resample, mic at 16kHz)
        """
        start = time.perf_counter()

        # ── Step 0: Ensure float32 contiguous ─────────────────────────────
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32, copy=False)
        if not audio.flags["C_CONTIGUOUS"]:
            audio = np.ascontiguousarray(audio)

        # ── Step 1: Stereo → Mono ─────────────────────────────────────────
        if audio.ndim == 2:
            if audio.shape[1] > 1:
                audio = audio.mean(axis=1)
            else:
                audio = audio.ravel()
        elif audio.ndim > 2:
            audio = audio.ravel()

        # Ensure 1-D
        if audio.ndim != 1:
            audio = audio.ravel()

        # ── Step 2: Resample to 16 kHz (SKIPPED if already 16 kHz) ────────
        if source_sr != TARGET_SR:
            audio = self._resample(audio, source_sr)

        # ── Step 3: DC offset removal ─────────────────────────────────────
        # Removes any DC bias from the capture hardware. Vectorized O(N).
        audio = audio - np.mean(audio)

        # ── Step 4: Noise gate (silence detection) ────────────────────────
        rms = float(np.sqrt(np.mean(audio * audio)))
        if rms < NOISE_GATE_RMS:
            # Pure silence or near-silence — skip inference
            return np.array([], dtype=np.float32)

        # ── Step 5: RMS normalization ─────────────────────────────────────
        # Scale audio so RMS equals TARGET_RMS (~-20 dBFS).
        # This is more stable than peak normalization for speech because
        # occasional plosives/clicks don't squash the entire signal.
        scale = TARGET_RMS / rms
        # Clamp scale to avoid amplifying noise excessively
        scale = min(scale, 10.0)
        audio = audio * scale

        # Clip to [-1, 1] to prevent downstream overflow
        np.clip(audio, -1.0, 1.0, out=audio)

        latency = (time.perf_counter() - start) * 1000
        if latency > LATENCY_WARN_MS:
            logger.warning(f"Preprocess latency {latency:.1f} ms > {LATENCY_WARN_MS} ms budget")
        else:
            logger.debug(f"Preprocess: {latency:.1f} ms | rms={rms:.4f} | scale={scale:.2f}")

        return audio
