"""
audio_engine.py
=============================================================================
AudioEngine Module - Multi-source Capture (Mic & System Audio)
=============================================================================
"""

import logging
import queue
import threading
from abc import ABC, abstractmethod
from typing import Optional, Callable, List, Dict

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1


class BaseCapture(ABC):
    """Abstract base for any audio capture source."""
    def __init__(self, callback: Callable[[np.ndarray], None], noise_gate: float = 0.001):
        self.callback = callback
        self.noise_gate = noise_gate
        self.is_running = False
        self._stream: Optional[sd.InputStream] = None
        self._queue = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self.native_rate = SAMPLE_RATE

    @abstractmethod
    def start(self):
        pass

    def stop(self):
        self.is_running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._worker_thread and self._worker_thread.is_alive():
            self._queue.put(None)
            self._worker_thread.join(timeout=1.0)

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            logger.warning(f"Audio status: {status}")
        if self.is_running:
            self._queue.put(indata.copy())

    def _processing_worker(self, chunk_duration_s: float):
        """Processes raw audio from queue, resamples, and emits chunks."""
        buffer = []
        buffered_samples = 0
        
        while self.is_running:
            try:
                frame = self._queue.get(timeout=0.2)
                if frame is None: break
                
                buffer.append(frame)
                buffered_samples += len(frame)
                
                # Check if we have enough for a chunk based on native rate
                if buffered_samples >= int(self.native_rate * chunk_duration_s):
                    raw_chunk = np.concatenate(buffer, axis=0).flatten()
                    
                    # Resample if native rate != 16000
                    if self.native_rate != SAMPLE_RATE:
                        duration = len(raw_chunk) / self.native_rate
                        target_len = int(duration * SAMPLE_RATE)
                        chunk = np.interp(
                            np.linspace(0, duration, target_len, endpoint=False),
                            np.linspace(0, duration, len(raw_chunk), endpoint=False),
                            raw_chunk
                        ).astype(np.float32)
                    else:
                        chunk = raw_chunk

                    # Simple Amplitude Noise Gate
                    vol = np.max(np.abs(chunk))
                    if vol > 0.0001: # Avoid logging absolute silence
                        logger.info(f"CAPTURING [Audio Power]: {vol:.4f}")

                    if vol >= self.noise_gate:
                        self.callback(chunk)

                    # Maintain 0.2s overlap in native rate
                    overlap_samples = int(self.native_rate * 0.2)
                    overlap_frame = raw_chunk[-overlap_samples:].reshape(-1, 1)
                    buffer = [overlap_frame]
                    buffered_samples = len(overlap_frame)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Capture worker error: {e}")
                break


class MicrophoneCapture(BaseCapture):
    """Standard Microphone input capture."""
    def __init__(self, device_index=None, callback=None, noise_gate=0.005):
        super().__init__(callback, noise_gate)
        self.device_index = device_index

    def start(self, chunk_duration_s=1.0):
        if self.is_running: return
        try:
            device_info = sd.query_devices(self.device_index, 'input')
            self.native_rate = int(device_info.get('default_samplerate', 44100))
            
            self._stream = sd.InputStream(
                samplerate=self.native_rate,
                channels=CHANNELS,
                device=self.device_index,
                callback=self._audio_callback,
                dtype='float32'
            )
            self.is_running = True
            self._stream.start()
            
            self._worker_thread = threading.Thread(
                target=self._processing_worker, args=(chunk_duration_s,), daemon=True
            )
            self._worker_thread.start()
            logger.info(f"Microphone capture started on device {self.device_index}")
        except Exception as e:
            logger.error(f"Failed to start Mic: {e}")
            raise


class SystemAudioCapture(BaseCapture):
    """Windows WASAPI Loopback / Stereo Mix capture."""
    def __init__(self, callback=None, noise_gate=0.005):
        super().__init__(callback, noise_gate)

    def _find_wasapi_loopback(self) -> Optional[int]:
        """Find the device index for digital loopback using WASAPI."""
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
            
            wasapi_idx = -1
            for i, h in enumerate(hostapis):
                if "WASAPI" in h['name']:
                    wasapi_idx = i
                    break
            
            if wasapi_idx == -1: return None

            # Phase 1: Search for 'Loopback' in inputs (Best Case)
            for i, d in enumerate(devices):
                if d['hostapi'] == wasapi_idx and d['max_input_channels'] > 0:
                    if "loopback" in d['name'].lower():
                        logger.info(f"Using Direct Loopback: {d['name']}")
                        return i
            
            # Phase 2: Search for the couple of 'Speakers' or 'Output' on WASAPI
            host_info = sd.query_hostapis(wasapi_idx)
            default_out_idx = host_info.get('default_output_device')
            if default_out_idx is not None and default_out_idx != -1:
                # Find an input with a similar name OR Stereo Mix as last resort
                for i, d in enumerate(devices):
                    if d['hostapi'] == wasapi_idx and d['max_input_channels'] > 0:
                        name = d['name'].lower()
                        if any(kw in name for kw in ["stereo mix", "wave out", "loopback"]):
                             logger.info(f"Mapping System Audio to: {d['name']}")
                             return i
            
            # Phase 3: Last resort WASAPI input
            for i, d in enumerate(devices):
                if d['hostapi'] == wasapi_idx and d['max_input_channels'] > 0:
                    return i
                    
            return None
        except Exception:
            return None

    def start(self, chunk_duration_s=1.0):
        if self.is_running: return
        device_idx = self._find_wasapi_loopback()
        
        if device_idx is None:
            raise RuntimeError("WASAPI Loopback device not found.")
            
        try:
            device_info = sd.query_devices(device_idx, 'input')
            self.native_rate = int(device_info.get('default_samplerate', 44100))
            
            # This is the 'Magic' for WASAPI Loopback on Windows
            try:
                from sounddevice import WasapiSettings
                extra_settings = WasapiSettings(loopback=True)
            except: extra_settings = None

            self._stream = sd.InputStream(
                samplerate=self.native_rate,
                channels=CHANNELS,
                device=device_idx,
                callback=self._audio_callback,
                dtype='float32',
                extra_settings=extra_settings
            )
            self.is_running = True
            self._stream.start()
            
            self._worker_thread = threading.Thread(
                target=self._processing_worker, args=(chunk_duration_s,), daemon=True
            )
            self._worker_thread.start()
            logger.info(f"System Audio Capture ACTIVE on device {device_idx}")
        except Exception as e:
            logger.error(f"System Audio Start Failed: {e}")
            raise

    def start(self, chunk_duration_s=1.0):
        if self.is_running: return
        device_idx = self._find_wasapi_loopback()
        if device_idx is None:
            raise RuntimeError("No WASAPI Loopback device found.")
            
        try:
            device_info = sd.query_devices(device_idx, 'input')
            self.native_rate = int(device_info.get('default_samplerate', 44100))
            
            self._stream = sd.InputStream(
                samplerate=self.native_rate,
                channels=CHANNELS,
                device=device_idx,
                callback=self._audio_callback,
                dtype='float32'
            )
            self.is_running = True
            self._stream.start()
            
            self._worker_thread = threading.Thread(
                target=self._processing_worker, args=(chunk_duration_s,), daemon=True
            )
            self._worker_thread.start()
            logger.info(f"System Audio capture started on device {device_idx}")
        except Exception as e:
            logger.error(f"Failed to start Sys Audio: {e}")
            raise


class AudioEngine:
    """Wrapper to maintain compatibility and manage dual sources."""
    def __init__(self):
        self.mic = None
        self.sys = None

    @staticmethod
    def list_devices():
        return [dict(d, index=i) for i, d in enumerate(sd.query_devices())]

    def stop_all(self):
        if self.mic: self.mic.stop()
        if self.sys: self.sys.stop()
        self.mic = None
        self.sys = None
