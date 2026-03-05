"""
app_controller.py
=============================================================================
Application Controller (Middle Tier)
=============================================================================
"""

import os
from typing import Optional
from PySide6.QtCore import QObject, QThread, Signal, Slot
from config import setup_logger, SUPPORTED_LANGUAGES, TRANSLATION_MAPPING

# Core Backend
from core.ai_engine import AIEngine, StreamingTranscriber
from core.audio_engine import AudioEngine, MicrophoneCapture, SystemAudioCapture
from core.translation_engine import TranslationEngine

logger = setup_logger("AppController")

class LiveStreamingWorker(QObject):
    """
    Handles simultaneous dual-source capture and transcription.
    """
    chunk_result = Signal(dict)
    error = Signal(str)

    def __init__(self, ai_engine, audio_engine, translation_engine, 
                 model_name, src_lang, tgt_lang, 
                 enable_sys=True, enable_mic=True, 
                 mic_device_index=None, noise_gate=0.005):
        super().__init__()
        self._ai = ai_engine
        self._audio_engine = audio_engine # Just for listing devices if needed
        self._translator = translation_engine
        
        self.model_name = model_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        self.enable_sys = enable_sys
        self.enable_mic = enable_mic
        self.mic_device_index = mic_device_index
        self.noise_gate = noise_gate
        
        self._is_running = False
        self._transcriber = StreamingTranscriber(self._ai)
        
        # Sources
        self.sys_capture: Optional[SystemAudioCapture] = None
        self.mic_capture: Optional[MicrophoneCapture] = None

    def start(self):
        try:
            logger.info("Initializing Dual-Mode Live Streaming...")
            self._ai.load_whisper(self.model_name)
            
            # Setup Callbacks
            self._transcriber.on_sys_text = lambda text: self._handle_text("SYS", text)
            self._transcriber.on_mic_text = lambda text: self._handle_text("MIC", text)
            
            self._is_running = True
            lang_code = SUPPORTED_LANGUAGES.get(self.src_lang)

            # Start System Audio
            if self.enable_sys:
                self.sys_capture = SystemAudioCapture(
                    callback=lambda chunk: self._transcriber.process_chunk("sys", chunk, lang_code),
                    noise_gate=self.noise_gate
                )
                self.sys_capture.start(chunk_duration_s=0.4)

            # Start Microphone
            if self.enable_mic:
                self.mic_capture = MicrophoneCapture(
                    device_index=self.mic_device_index,
                    callback=lambda chunk: self._transcriber.process_chunk("mic", chunk, lang_code),
                    noise_gate=self.noise_gate
                )
                self.mic_capture.start(chunk_duration_s=0.4)

        except Exception as e:
            logger.error(f"Live Start Error: {e}")
            self.error.emit(str(e))

    def _handle_text(self, source: str, text: str):
        if not self._is_running: return
        
        clean_text = text.strip()
        if len(clean_text) < 2: return

        # Format with prefix
        display_text = f"[{source}] {clean_text}"
        
        # Translation logic
        target_lang_code = TRANSLATION_MAPPING.get(self.tgt_lang)
        if target_lang_code:
            try:
                src_nllb = "eng_Latn"
                if self.src_lang == "Vietnamese": src_nllb = "vie_Latn"
                trans = self._translator.translate(clean_text, src_lang=src_nllb, tgt_lang=target_lang_code)
                display_text = f"[{source}] {clean_text}\n   └ [VIE] {trans}"
            except Exception:
                pass

        self.chunk_result.emit({
            "speakers": [(source, display_text)],
            "text": display_text
        })

    def stop(self):
        self._is_running = False
        if self.sys_capture: self.sys_capture.stop()
        if self.mic_capture: self.mic_capture.stop()
        logger.info("Streaming sources stopped.")


class AppController(QObject):
    controller_status = Signal(str, str)
    transcript_progress = Signal(int)
    transcript_result = Signal(dict)
    streaming_chunk_ready = Signal(dict)
    
    def __init__(self):
        super().__init__()
        self.ai_engine = AIEngine()
        self.audio_engine = AudioEngine()
        self.translation_engine = TranslationEngine()
        self.live_worker: Optional[LiveStreamingWorker] = None

    @Slot(str, str, str, str, bool, bool, object, float)
    def start_live_streaming(self, model_name: str, src_lang: str, trans_target: str, 
                             compute_mode: str, enable_sys: bool, enable_mic: bool, 
                             device_index=None, noise_gate=0.005):
        
        if self.live_worker: return

        # Apply compute mode
        if compute_mode == "Force CPU":
            self.ai_engine.device = "cpu"
            self.ai_engine.compute_type = "int8"
        elif compute_mode == "Force GPU (NVIDIA)":
            import torch
            if torch.cuda.is_available():
                self.ai_engine.device = "cuda"
                self.ai_engine.compute_type = "int8_float16"

        self.live_worker = LiveStreamingWorker(
            self.ai_engine, self.audio_engine, self.translation_engine,
            model_name, src_lang, trans_target, 
            enable_sys=enable_sys, enable_mic=enable_mic,
            mic_device_index=device_index, noise_gate=noise_gate
        )
        self.live_worker.chunk_result.connect(self.streaming_chunk_ready.emit)
        self.live_worker.error.connect(lambda e: self.controller_status.emit(e, "error"))
        
        self.live_worker.start()
        status_msg = "Capture ACTIVE: "
        if enable_sys: status_msg += "[SYS] "
        if enable_mic: status_msg += "[MIC] "
        self.controller_status.emit(status_msg, "success")

    @Slot()
    def stop_live_streaming(self):
        if self.live_worker:
            self.live_worker.stop()
            self.live_worker = None
            self.controller_status.emit("Streaming Stopped.", "idle")

    def cancel_transcription(self):
        """Abort any ongoing background tasks immediately."""
        logger.info("Cancelling all background tasks...")
        self.stop_live_streaming()

    def graceful_shutdown(self):
        """Release all hardware resources (GPU/Mic) before exit."""
        self.cancel_transcription()
        self.ai_engine.free_memory()
