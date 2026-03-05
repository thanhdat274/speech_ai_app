"""
ai_engine.py
=============================================================================
AIEngine Module - Core Faster-Whisper AI Engine
=============================================================================
"""

import logging
import threading
import queue
from typing import Optional, Dict, Any, List, Callable

import torch
import numpy as np
from faster_whisper import WhisperModel

# Configure module-level logger
logger = logging.getLogger(__name__)


class AIEngine:
    """
    Singleton-based thread-safe AI Engine managing Faster-Whisper.
    """
    _instance: Optional["AIEngine"] = None
    _init_lock = threading.Lock()

    def __new__(cls) -> "AIEngine":
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = super(AIEngine, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        with self._init_lock:
            if getattr(self, "_initialized", False):
                return
            
            self._initialized = True
            self._lock = threading.Lock()

            # Hardware State
            self.device: str = "cpu"
            self.compute_type: str = "int8"
            self.vram_gb: float = 0.0

            # Models
            self._whisper_model: Optional[WhisperModel] = None
            self._whisper_model_name: Optional[str] = None
            
            self._detect_hardware()

    def _detect_hardware(self) -> None:
        if torch.cuda.is_available():
            try:
                torch.cuda.init()
                self.device = "cuda"
                mem_bytes = torch.cuda.get_device_properties(0).total_memory
                self.vram_gb = mem_bytes / (1024 ** 3)
                # Optimized for GTX 1650 (4GB)
                self.compute_type = "int8_float16" if self.vram_gb < 6 else "float16"
                logger.info(f"Hardware: GPU ({self.vram_gb:.1f}GB). Compute: {self.compute_type}")
            except Exception as e:
                logger.warning(f"CUDA Init failed: {e}")
                self._fallback_cpu()
        else:
            self._fallback_cpu()

    def _fallback_cpu(self):
        self.device = "cpu"
        self.compute_type = "int8"
        logger.info("Hardware: CPU (int8)")

    def _auto_select_whisper_model(self) -> str:
        """Determine optimal model based on detected hardware."""
        if self.device == "cpu":
            return "base"
        
        # Optimized for modern GPUs
        if self.vram_gb < 3.5:
            return "tiny"
        elif self.vram_gb < 5.5:
            return "base"
        elif self.vram_gb < 8.5:
            return "small"
        elif self.vram_gb < 11.5:
            return "medium"
        else:
            return "large-v3-turbo"

    def load_whisper(self, model_size: str = "base") -> None:
        """Load or reload model, handling 'Auto (Recommended)' input."""
        # Sanitize input from UI
        if not model_size or any(x in model_size for x in ["Auto", "Recommended", "auto"]):
            model_size = self._auto_select_whisper_model()
            
        with self._lock:
            if self._whisper_model and self._whisper_model_name == model_size:
                return
            
            logger.info(f"Loading Whisper model '{model_size}' on {self.device}...")
            try:
                self._whisper_model = WhisperModel(
                    model_size, 
                    device=self.device, 
                    compute_type=self.compute_type
                )
                self._whisper_model_name = model_size
                logger.info("Faster-Whisper model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load Whisper model '{model_size}': {e}")
                # Fallback to tiny if everything fails
                if model_size != "tiny":
                    self.load_whisper("tiny")
                else:
                    raise

    def transcribe(self, audio: np.ndarray, language: Optional[str] = None) -> Dict[str, Any]:
        """Thread-safe transcription call."""
        if not self._whisper_model:
            self.load_whisper()
            
        with self._lock:
            # Increase beam_size for 'large' models to improve accuracy significantly.
            # beam_size=1 is fast but prone to errors in tonal languages.
            current_beam = 1
            if "large" in (self._whisper_model_name or ""):
                current_beam = 5 # Maximum accuracy for Vietnamese
            elif "medium" in (self._whisper_model_name or ""):
                current_beam = 3

            # condition_on_previous_text=False prevents the AI from getting stuck in loops.
            segments, info = self._whisper_model.transcribe(
                audio, 
                beam_size=current_beam, 
                language=language,
                vad_filter=False, # Disable VAD to ensure AI catches everything
                initial_prompt="Real-time transcription for system audio. Vietnamese and English preferred.",
                condition_on_previous_text=False,
                vad_parameters=dict(min_silence_duration_ms=100)
            )
            
            full_text = "".join([s.text for s in list(segments)]).strip()
            return {"text": full_text, "language": info.language}


    def free_memory(self):
        with self._lock:
            self._whisper_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class StreamingTranscriber:
    """Manages real-time transcription for multiple sources."""
    def __init__(self, ai_engine: AIEngine):
        self.ai = ai_engine
        self.on_sys_text: Optional[Callable[[str], None]] = None
        self.on_mic_text: Optional[Callable[[str], None]] = None

    def process_chunk(self, source: str, audio: np.ndarray, language: Optional[str] = None):
        """Processes a single chunk from either 'sys' or 'mic'."""
        try:
            result = self.ai.transcribe(audio, language=language)
            text = result.get("text", "")
            if not text: return

            if source == "sys" and self.on_sys_text:
                self.on_sys_text(text) # Remove prefix here, handled in AppController
            elif source == "mic" and self.on_mic_text:
                self.on_mic_text(text)
        except Exception as e:
            logger.error(f"StreamingTranscriber error [{source}]: {e}")
