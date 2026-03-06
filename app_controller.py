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
from core.audio_engine import (
    AudioEngine, MicrophoneCapture, SystemAudioCapture,
    CHUNK_DURATION_S, ULTRA_CHUNK_DURATION_S,
    BALANCED_CHUNK_DURATION_S, ACCURACY_CHUNK_DURATION_S,
)
from core.translation_engine import TranslationEngine
from core.config_manager import ConfigManager

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
                 mic_device_index=None, noise_gate=0.005, prompt=""):
        super().__init__()
        self._ai = ai_engine
        self._audio_engine = audio_engine
        self._translator = translation_engine
        
        self.model_name = model_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.prompt = prompt
        
        self.enable_sys = enable_sys
        self.enable_mic = enable_mic
        self.mic_device_index = mic_device_index
        self.noise_gate = noise_gate
        
        self._is_running = False
        self._transcriber = StreamingTranscriber(self._ai)
        
        # Sources
        self.sys_capture: Optional[SystemAudioCapture] = None
        self.mic_capture: Optional[MicrophoneCapture] = None

        # ── Sentence Builders (one per source) ────────────────────────
        # Accumulate partial ASR output → emit complete sentences → NLLB
        # This prevents translating raw chunks which causes broken output.
        from core.sentence_builder import SentenceBuilder

        target_lang_code = TRANSLATION_MAPPING.get(self.tgt_lang)
        if target_lang_code:
            src_nllb = "vie_Latn" if self.src_lang == "Vietnamese" else "eng_Latn"
            label = self.tgt_lang[:3].upper()

            self._sys_sentence_builder = SentenceBuilder(
                on_sentence=lambda text: self._translate_sentence("SYS", text, src_nllb, target_lang_code, label)
            )
            self._mic_sentence_builder = SentenceBuilder(
                on_sentence=lambda text: self._translate_sentence("MIC", text, src_nllb, target_lang_code, label)
            )
        else:
            self._sys_sentence_builder = None
            self._mic_sentence_builder = None

    def start(self):
        try:
            # Choose chunk size based on latency mode
            # Pipeline: Audio Capture → Ring Buffer → Dynamic Chunk Builder
            #           → Inference Queue → AI Engine → Subtitle Smoother
            #
            # ULTRA_REALTIME → 0.3s (fast response)
            # BALANCED       → 0.7s (default — good latency + Vietnamese accuracy)
            # ACCURACY       → 1.5s (maximum Vietnamese tone context)
            if self._ai.ultra_realtime_mode:
                chunk_s    = ULTRA_CHUNK_DURATION_S
                mode_label = "ULTRA_REALTIME (0.3s)"
            else:
                chunk_s    = BALANCED_CHUNK_DURATION_S
                mode_label = "BALANCED (0.7s)"
            logger.info(f"Initializing Dual-Mode Live Streaming — mode={mode_label}")

            self._ai.load_whisper(self.model_name)

            # Setup Callbacks
            self._transcriber.on_sys_text = lambda text: self._handle_text("SYS", text)
            self._transcriber.on_mic_text = lambda text: self._handle_text("MIC", text)

            self._is_running = True
            lang_code = SUPPORTED_LANGUAGES.get(self.src_lang)

            # Start System Audio
            if self.enable_sys:
                self.sys_capture = SystemAudioCapture(
                    callback=lambda chunk: self._transcriber.process_chunk("sys", chunk, lang_code, prompt=self.prompt),
                    noise_gate=self.noise_gate
                )
                self.sys_capture.start(chunk_duration_s=chunk_s)

            # Start Microphone
            if self.enable_mic:
                self.mic_capture = MicrophoneCapture(
                    device_index=self.mic_device_index,
                    callback=lambda chunk: self._transcriber.process_chunk("mic", chunk, lang_code, prompt=self.prompt),
                    noise_gate=self.noise_gate
                )
                self.mic_capture.start(chunk_duration_s=chunk_s)

        except Exception as e:
            logger.error(f"Live Start Error: {e}")
            self.error.emit(str(e))

    def _handle_text(self, source: str, text: str):
        """Handle transcribed text from Whisper.

        Pipeline:
            1. Emit transcription to UI immediately (never wait)
            2. Feed text to SentenceBuilder for translation buffering
               → SentenceBuilder accumulates partials
               → emits complete sentences to _translate_sentence()
        """
        if not self._is_running:
            return

        clean_text = text.strip()
        if len(clean_text) < 2:
            return

        # Emit transcription immediately — user sees it without delay
        self.chunk_result.emit({
            "speakers": [(source, clean_text)],
            "text": clean_text,
        })

        # Feed to SentenceBuilder for buffered translation
        sb = (
            self._sys_sentence_builder if source == "SYS"
            else self._mic_sentence_builder
        )
        if sb is not None:
            sb.add_partial(clean_text)

    def _translate_sentence(
        self, source: str, sentence: str,
        src_nllb: str, tgt_code: str, label: str,
    ):
        """Called by SentenceBuilder when a complete sentence is ready.

        Fires async translation — never blocks the ASR pipeline.
        """
        if not self._is_running:
            return

        def _on_translated(result: str, _src=source, _lbl=label):
            if not self._is_running:
                return
            translated = f"   └ [{_lbl}] {result}"
            self.chunk_result.emit({
                "speakers": [(_src, translated)],
                "text": translated,
            })

        self._translator.translate_async(
            sentence, src_nllb, tgt_code, _on_translated
        )

    def stop(self):
        self._is_running = False
        # Flush any remaining buffered sentences before stopping
        if self._sys_sentence_builder:
            self._sys_sentence_builder.flush()
        if self._mic_sentence_builder:
            self._mic_sentence_builder.flush()
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

        # Sync AI engine config from persisted settings
        _cfg = ConfigManager()
        self.ai_engine.force_language = _cfg.get("force_language", True)
        self.ai_engine.vi_base_prompt = _cfg.get("vi_base_prompt", self.ai_engine.vi_base_prompt)
        self.ai_engine.ultra_realtime_mode = _cfg.get("ultra_realtime_mode", False)
        if self.ai_engine.ultra_realtime_mode:
            logger.info("Ultra Realtime Mode ENABLED — chunk=0.3s, aggressive VAD, no context memory.")

    @Slot(str, str, str, str, bool, bool, object, float, str)
    def start_live_streaming(self, model_name: str, src_lang: str, trans_target: str, 
                             compute_mode: str, enable_sys: bool, enable_mic: bool, 
                             device_index=None, noise_gate=0.005, prompt=""):
        
        if self.live_worker: return

        # Apply compute mode BEFORE loading model
        need_reload = False
        if compute_mode == "Force CPU":
            if self.ai_engine.device != "cpu":
                need_reload = True
            self.ai_engine.device = "cpu"
            self.ai_engine.compute_type = "int8"
        elif compute_mode == "Force GPU (NVIDIA)":
            import torch
            if torch.cuda.is_available():
                if self.ai_engine.device != "cuda":
                    need_reload = True
                self.ai_engine.device = "cuda"
                # Optimized for GTX 1650 (4GB VRAM)
                vram = self.ai_engine.vram_gb
                self.ai_engine.compute_type = "int8_float16" if vram < 6 else "float16"
                logger.info(f"Forcing GPU: device=cuda, compute={self.ai_engine.compute_type}")

        # Force reload model on new device if device changed
        if need_reload:
            self.ai_engine.load_whisper(model_name, force_reload=True)

        self.live_worker = LiveStreamingWorker(
            self.ai_engine, self.audio_engine, self.translation_engine,
            model_name, src_lang, trans_target, 
            enable_sys=enable_sys, enable_mic=enable_mic,
            mic_device_index=device_index, noise_gate=noise_gate,
            prompt=prompt
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
