"""
app_controller.py
=============================================================================
Application Controller (Middle Tier) — v8 Streaming Translation Architecture
=============================================================================

Pipeline (v8):

  Audio → AudioPreprocessor (ultra_fast in ULTRA mode)
        → Whisper (small / int8_float16)
        → SubtitleSmoother    ─── partial ──→ UI (raw STT, white text)
        → SentenceBuilder     ─── complete sentence ──→
        → StreamingTranslator (per source)
               │  context window (last 5 sentences)
               │  NLLB-600M translate()
               │  VietnameseCorrector (tgt == vi)
               │  TranslationOutputSmoother (dedup + 100ms debounce)
               └──────────────────────────────→ UI (translated, teal text)

v8 vs v7 differences:
  - StreamingTranslator replaces file-polling TranslationWorker for LIVE
    sessions. Translation starts immediately after Whisper finishes — no
    200ms file-poll hop.
  - Context memory: per-source deque of last 5 (src, tgt) sentence pairs.
    NLLB receives recent context → better cross-chunk coherence.
  - TranslationOutputSmoother: 100ms debounce + SequenceMatcher dedup.
  - TranscriptBuffer + TranslationWorker retained for fallback/file mode.
  - Clear button resets StreamingTranslators + buffer + worker.

Performance targets:
  STT latency:         < 400ms
  Translation latency: 50–300ms (GPU) on top of STT
  Queue usage:         < 5/20
=============================================================================
"""

import logging
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
from core.transcript_buffer import TranscriptBuffer
from core.translation_worker import TranslationWorker
from core.streaming_translator import StreamingTranslator
from core.config_manager import ConfigManager

logger = setup_logger("AppController")

# ── Performance settings ──────────────────────────────────────────────────────
_DEFAULT_MODEL    = "small"
_CHUNK_DURATION_S = ULTRA_CHUNK_DURATION_S  # v8: 1.2 s default (was 0.8 s)


class LiveStreamingWorker(QObject):
    """Manages dual-source realtime capture → STT → streaming translation.

    v8 Pipeline per source (SYS / MIC):

        Hardware audio
          → AudioPreprocessor (ultra_fast path in ULTRA_REALTIME mode)
          → Whisper (small, int8_float16)
          → SubtitleSmoother  ──→ chunk_result Signal  (raw STT partial, white)
          → SentenceBuilder   ──→ complete sentence
          → StreamingTranslator
                ├── context_window (last 5 sentences, src+tgt)
                ├── NLLB translate (with context hint)
                ├── VietnameseCorrector (if tgt == vi)
                └── TranslationOutputSmoother
                        └──→ translation_result Signal (translated, teal)

    Two StreamingTranslator instances — one per source — so SYS and MIC
    contexts are never mixed.
    """

    chunk_result       = Signal(dict)   # raw STT partial → UI (white)
    translation_result = Signal(dict)   # translated text → UI (teal)
    error              = Signal(str)

    def __init__(
        self,
        ai_engine,
        audio_engine,
        translation_engine,
        transcript_buffer: TranscriptBuffer,
        model_name: str,
        src_lang: str,
        tgt_lang: str,
        enable_sys: bool = True,
        enable_mic: bool = True,
        mic_device_index=None,
        noise_gate: float = 0.005,
        prompt: str = "",
    ):
        super().__init__()
        self._ai               = ai_engine
        self._audio_engine     = audio_engine
        self._translator_eng   = translation_engine
        self._transcript_buf   = transcript_buffer

        self.model_name        = model_name
        self.src_lang          = src_lang
        self.tgt_lang          = tgt_lang
        self.prompt            = prompt

        self.enable_sys        = enable_sys
        self.enable_mic        = enable_mic
        self.mic_device_index  = mic_device_index
        self.noise_gate        = noise_gate

        self._is_running  = False
        self._transcriber = StreamingTranscriber(self._ai)

        # Audio capture handles
        self.sys_capture: Optional[SystemAudioCapture] = None
        self.mic_capture: Optional[MicrophoneCapture]  = None

        # ── Sentence Builders — one per source ───────────────────────────
        from core.sentence_builder import SentenceBuilder

        self._sys_sentence_builder = SentenceBuilder(
            on_sentence=lambda text: self._on_sentence_ready("SYS", text)
        )
        self._mic_sentence_builder = SentenceBuilder(
            on_sentence=lambda text: self._on_sentence_ready("MIC", text)
        )

        # ── StreamingTranslators — one per source ─────────────────────────
        # Created here; activated in start() once we know src/tgt NLLB codes.
        self._sys_streaming_translator: Optional[StreamingTranslator] = None
        self._mic_streaming_translator: Optional[StreamingTranslator] = None

    # ------------------------------------------------------------------
    # Start / Stop
    # ------------------------------------------------------------------
    def start(self) -> None:
        try:
            # ── Chunk size by latency mode ────────────────────────────────
            if self._ai.ultra_realtime_mode:
                chunk_s    = ULTRA_CHUNK_DURATION_S   # 1.2 s
                mode_label = "ULTRA_REALTIME (1.2s, stride-preprocess)"
            else:
                chunk_s    = BALANCED_CHUNK_DURATION_S  # 1.0 s
                mode_label = f"BALANCED ({BALANCED_CHUNK_DURATION_S}s)"

            logger.info(f"[LiveWorker] Starting — mode={mode_label}")

            self._ai.load_whisper(self.model_name)

            # STT callbacks
            self._transcriber.on_sys_text = lambda t: self._handle_text("SYS", t)
            self._transcriber.on_mic_text = lambda t: self._handle_text("MIC", t)

            self._is_running = True
            lang_code = SUPPORTED_LANGUAGES.get(self.src_lang)

            # ── Resolve NLLB codes ────────────────────────────────────────
            target_nllb = TRANSLATION_MAPPING.get(self.tgt_lang)
            src_nllb = _src_to_nllb(self.src_lang)

            if target_nllb and not _nllb_same(src_nllb, target_nllb):
                # Create per-source StreamingTranslators
                self._sys_streaming_translator = StreamingTranslator(
                    translator_engine = self._translator_eng,
                    callback = lambda t: self._on_translation("SYS", t),
                    src_lang = src_nllb,
                    tgt_lang = target_nllb,
                    label    = "SYS",
                )
                self._mic_streaming_translator = StreamingTranslator(
                    translator_engine = self._translator_eng,
                    callback = lambda t: self._on_translation("MIC", t),
                    src_lang = src_nllb,
                    tgt_lang = target_nllb,
                    label    = "MIC",
                )
                logger.info(
                    f"[LiveWorker] StreamingTranslators active: "
                    f"{src_nllb} → {target_nllb}"
                )
            else:
                logger.info(
                    "[LiveWorker] Translation disabled "
                    f"(target={self.tgt_lang}, src={self.src_lang})"
                )

            # ── System Audio ──────────────────────────────────────────────
            if self.enable_sys:
                self.sys_capture = SystemAudioCapture(
                    callback=lambda chunk: self._transcriber.process_chunk(
                        "sys", chunk, lang_code, prompt=self.prompt
                    ),
                    noise_gate=self.noise_gate,
                )
                self.sys_capture.ultra_realtime_mode = self._ai.ultra_realtime_mode
                self.sys_capture.start(chunk_duration_s=chunk_s)
                logger.info("[LiveWorker] System audio capture started.")

            # ── Microphone ────────────────────────────────────────────────
            if self.enable_mic:
                self.mic_capture = MicrophoneCapture(
                    device_index=self.mic_device_index,
                    callback=lambda chunk: self._transcriber.process_chunk(
                        "mic", chunk, lang_code, prompt=self.prompt
                    ),
                    noise_gate=self.noise_gate,
                )
                self.mic_capture.ultra_realtime_mode = self._ai.ultra_realtime_mode
                self.mic_capture.start(chunk_duration_s=chunk_s)
                logger.info("[LiveWorker] Microphone capture started.")

        except Exception as e:
            logger.error(f"LiveStreamingWorker.start error: {e}")
            self.error.emit(str(e))

    def stop(self) -> None:
        self._is_running = False

        # Flush sentence builders — emit any buffered partial sentences
        for sb in [self._sys_sentence_builder, self._mic_sentence_builder]:
            if sb:
                try:
                    sb.flush()
                except Exception:
                    pass

        # Flush translation smoothers
        for st in [self._sys_streaming_translator, self._mic_streaming_translator]:
            if st:
                try:
                    st.flush()
                except Exception:
                    pass

        if self.sys_capture:
            self.sys_capture.stop()
        if self.mic_capture:
            self.mic_capture.stop()

        logger.info("[LiveWorker] STT capture stopped.")

    def reset_translators(self) -> None:
        """Clear context and smoother state without stopping capture."""
        for st in [self._sys_streaming_translator, self._mic_streaming_translator]:
            if st:
                st.reset()

    # ------------------------------------------------------------------
    # STT text handler — raw Whisper output
    # ------------------------------------------------------------------
    def _handle_text(self, source: str, text: str) -> None:
        """Raw Whisper output → UI + SentenceBuilder.

        Stage latency logged at DEBUG level.
        NO disk I/O here — realtime hot path must stay lightweight (req #8/#9).
        """
        import time as _time
        if not self._is_running:
            return

        clean = text.strip()
        if len(clean) < 2:
            return

        t_recv = _time.perf_counter()

        # ── Immediate raw STT emission → UI (white text, no delay) ──────────
        self.chunk_result.emit({
            "speakers": [(source, clean)],
            "text":     clean,
            "type":     "partial",
        })

        # ── SentenceBuilder: accumulate until boundary ────────────────────
        # transcript_buf.append() is called in _on_sentence_ready (complete
        # sentences only) — NOT on every partial chunk. (req #8/#9)
        sb = (
            self._sys_sentence_builder if source == "SYS"
            else self._mic_sentence_builder
        )
        if sb is not None:
            sb.add_partial(clean)

        logger.debug(
            f"[Pipeline:{source}] STT recv | "
            f"words={len(clean.split())} | "
            f"emit+sb_feed={(_time.perf_counter()-t_recv)*1000:.1f}ms | "
            f"{clean[:40]!r}"
        )

    def _on_sentence_ready(self, source: str, sentence: str) -> None:
        """Complete sentence from SentenceBuilder → TranscriptBuffer + StreamingTranslator.

        v9: transcript_buf.append() happens HERE (complete sentences only)
        so the realtime STT hot path has zero I/O. The in-memory buffer is
        O(1), and the async FileWriter thread handles disk persistence.
        """
        if not self._is_running:
            return

        sentence = sentence.strip()
        if not sentence:
            return

        logger.debug(f"[LiveWorker] Sentence ready [{source}]: {sentence!r}")

        # ── Persist sentence to in-memory buffer (async file write off-thread) ─
        self._transcript_buf.append(sentence)

        st = (
            self._sys_streaming_translator if source == "SYS"
            else self._mic_streaming_translator
        )

        if st is not None:
            # Run translation in a short-lived thread so it never blocks
            # the Whisper worker thread (which is processing the next chunk)
            import threading
            threading.Thread(
                target=st.feed,
                args=(sentence,),
                daemon=True,
                name=f"StreamTrans-{source}",
            ).start()

    def _on_translation(self, source: str, translated: str) -> None:
        """Translation result from StreamingTranslator → Signal → UI."""
        if not self._is_running or not translated.strip():
            return
        self.translation_result.emit({
            "source":     source,
            "translated": translated,
        })


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _src_to_nllb(src_lang: str) -> str:
    """Map UI source language string → NLLB code."""
    _MAP = {
        "Vietnamese": "vie_Latn",
        "English":    "eng_Latn",
        "Japanese":   "jpn_Jpan",
        "Korean":     "kor_Hang",
        "Chinese":    "zho_Hans",
        "French":     "fra_Latn",
        "German":     "deu_Latn",
        "Spanish":    "spa_Latn",
        "Auto-Detect": "eng_Latn",  # fallback
    }
    return _MAP.get(src_lang, "eng_Latn")


def _nllb_same(src: str, tgt: str) -> bool:
    """True if both codes resolve to the same language."""
    _NORM = {
        "vie_latn": "vi", "eng_latn": "en", "jpn_jpan": "ja",
        "kor_hang": "ko", "zho_hans": "zh", "fra_latn": "fr",
        "deu_latn": "de", "spa_latn": "es",
    }
    return _NORM.get(src.lower(), src) == _NORM.get(tgt.lower(), tgt)


# ─────────────────────────────────────────────────────────────────────────────
# AppController
# ─────────────────────────────────────────────────────────────────────────────
class AppController(QObject):
    """Top-level controller wiring both pipelines.

    Signals:
        controller_status(str, str):   status message + level key
        transcript_progress(int):      0-100 for file processing
        transcript_result(dict):       file transcription result
        streaming_chunk_ready(dict):   live STT raw partial (Pipeline 1)
        translation_ready(dict):       translated output (StreamingTranslator)
    """

    controller_status     = Signal(str, str)
    transcript_progress   = Signal(int)
    transcript_result     = Signal(dict)
    streaming_chunk_ready = Signal(dict)
    translation_ready     = Signal(dict)

    def __init__(self) -> None:
        super().__init__()

        # ── Core singletons ────────────────────────────────────────────────
        self.ai_engine          = AIEngine()
        self.audio_engine       = AudioEngine()
        self.translation_engine = TranslationEngine()

        # ── Live worker ────────────────────────────────────────────────────
        self.live_worker: Optional[LiveStreamingWorker] = None

        # ── Transcript buffer (file log + fallback polling) ────────────────
        self.transcript_buffer = TranscriptBuffer(
            path=os.path.join(os.path.dirname(__file__), "live_transcript.txt")
        )

        # TranslationWorker retained as fallback / file-mode translation
        self._translation_worker: Optional[TranslationWorker] = None

        # ── Persisted settings ─────────────────────────────────────────────
        _cfg = ConfigManager()
        self.ai_engine.force_language      = _cfg.get("force_language", True)
        self.ai_engine.vi_base_prompt      = _cfg.get("vi_base_prompt", self.ai_engine.vi_base_prompt)
        self.ai_engine.ultra_realtime_mode = _cfg.get("ultra_realtime_mode", False)

        if self.ai_engine.ultra_realtime_mode:
            logger.info(
                "Ultra Realtime Mode ENABLED — "
                "chunk=1.2s, stride-preprocess, no context memory."
            )

        # v8: NLLB model preload was triggered in TranslationEngine.__init__.
        # By the time the user clicks Start (~30s+ after app launch), the model
        # will already be loaded. No blocking on first translate() call.
        logger.info(
            "[AppController] NLLB background preload started — "
            "model will be ready before first live session."
        )

    # ------------------------------------------------------------------
    # Live streaming — start / stop
    # ------------------------------------------------------------------
    @Slot(str, str, str, str, bool, bool, object, float, str)
    def start_live_streaming(
        self,
        model_name: str,
        src_lang: str,
        trans_target: str,
        compute_mode: str,
        enable_sys: bool,
        enable_mic: bool,
        device_index=None,
        noise_gate: float = 0.005,
        prompt: str = "",
    ) -> None:
        if self.live_worker:
            return

        # ── Apply compute mode ────────────────────────────────────────────
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
                vram = self.ai_engine.vram_gb
                self.ai_engine.compute_type = "int8_float16" if vram < 6 else "float16"
                logger.info(f"Forcing GPU | compute={self.ai_engine.compute_type}")

        if need_reload:
            self.ai_engine.load_whisper(model_name or _DEFAULT_MODEL, force_reload=True)

        # ── Create LiveStreamingWorker (v8 — includes StreamingTranslators) ─
        self.live_worker = LiveStreamingWorker(
            ai_engine         = self.ai_engine,
            audio_engine      = self.audio_engine,
            translation_engine = self.translation_engine,
            transcript_buffer = self.transcript_buffer,
            model_name        = model_name or _DEFAULT_MODEL,
            src_lang          = src_lang,
            tgt_lang          = trans_target,
            enable_sys        = enable_sys,
            enable_mic        = enable_mic,
            mic_device_index  = device_index,
            noise_gate        = noise_gate,
            prompt            = prompt,
        )

        # Raw STT partials → UI (white text)
        self.live_worker.chunk_result.connect(self.streaming_chunk_ready.emit)

        # Translated output → UI (teal text)
        self.live_worker.translation_result.connect(self._on_translation_ready)

        self.live_worker.error.connect(
            lambda e: self.controller_status.emit(e, "error")
        )
        self.live_worker.start()

        status_msg = "Capture ACTIVE: "
        if enable_sys: status_msg += "[SYS] "
        if enable_mic: status_msg += "[MIC] "
        self.controller_status.emit(status_msg, "success")

    @Slot()
    def stop_live_streaming(self) -> None:
        if self.live_worker:
            self.live_worker.stop()
            self.live_worker = None
        if self._translation_worker:
            self._translation_worker.stop()
            self._translation_worker = None
        self.controller_status.emit("Streaming Stopped.", "idle")

    # ------------------------------------------------------------------
    # Translation callback — marshalled to Qt via Signal
    # ------------------------------------------------------------------
    def _on_translation_ready(self, data: dict) -> None:
        """Forwarded from LiveStreamingWorker.translation_result Signal.

        data = {"source": "SYS"/"MIC", "translated": str}
        Re-emits as translation_ready for main.py to consume.
        """
        translated = data.get("translated", "").strip()
        source     = data.get("source", "TR")
        if not translated:
            return
        self.translation_ready.emit({
            "src":        "",
            "translated": translated,
            "text":       translated,
            "source":     source,
        })

    # ------------------------------------------------------------------
    # Clear — wipes transcript + resets translator context
    # ------------------------------------------------------------------
    @Slot()
    def clear_all(self) -> None:
        """Clear transcript file and reset all translation context.

        1. Reset StreamingTranslator context windows (src+tgt sentence history)
        2. Clear live_transcript.txt + read pointer
        3. Signal UI to clear subtitle display
        """
        logger.info("clear_all: resetting translation context and transcript buffer.")

        # Reset per-source StreamingTranslator context
        if self.live_worker:
            self.live_worker.reset_translators()

        # Clear file-based transcript
        self.transcript_buffer.clear()

        # Tell UI to clear subtitle display
        self.translation_ready.emit({
            "src": "", "translated": "", "text": "", "clear": True,
        })
        self.controller_status.emit("Cleared.", "idle")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def cancel_transcription(self) -> None:
        logger.info("Cancelling all background tasks...")
        self.stop_live_streaming()

    def graceful_shutdown(self) -> None:
        """Release all hardware resources (GPU/Mic/NLLB) before exit."""
        self.cancel_transcription()
        self.ai_engine.free_memory()
        self.translation_engine.shutdown()
