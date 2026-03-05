# Project Architecture

This document describes the Clean Architecture design for the Local AI Speech
Recognition + Translation App. The system is highly modular, ensuring that the
User Interface (UI), Business Logic (Core), and External Integrations (AI,
Hardware, Audio) remain decoupled.

## General Dependency Rules

1. **Inner circles have no knowledge of outer circles:** The `core/` module
   dictates interfaces but knows nothing about the concrete implementations
   inside `ui/` or `ai/`.
2. **Dependency Injection:** Core services receive their dependencies (like the
   speech engine or translator) via interfaces rather than direct instantiation.
3. **No Circular Imports:** Imports flow strictly downward from `ui/` -> `core/`
   -> Infrastructure (`ai/`, `hardware/`, etc.) -> `config/` & `utils/`.

---

## 1. `config/` (Configuration Management)

**Responsibilities:** Provides a centralized way to manage application settings,
user preferences, and constant values. It acts as the single source of truth for
paths, model choices, and UI states.

**Classes:**

- `AppConfig`: Singleton or global instance that loads and saves application
  state.

**Public Interfaces:**

- `get(key: str) -> Any`: Retrieve a configuration value.
- `set(key: str, value: Any) -> None`: Update a configuration value.
- `save() -> None`: Persist current config to a file (JSON/YAML).
- `load() -> dict`: Load config from disk.

**Dependency Rules:**

- Can import from: strictly `None` (or pure standard library/`utils/`).
- Depended on by: practically every other module.

---

## 2. `core/` (Business Logic & Use Cases)

**Responsibilities:** The heart of the application. It orchestrates the
interactions between audio capture, AI transcription, and translation. It
defines the "What" without caring "How" things are implemented.

**Classes:**

- `TranscriptionPipeline`: Orchestrates the flow of acquiring audio, sending it
  to the AI engine, and receiving text.
- `DiarizationManager`: Manages combining transcription timelines with speaker
  segmentation.
- `TaskManager`: Manages threading and asynchronous queues so the UI string
  operations don't freeze the main thread.

**Public Interfaces:**

- `start_live_session(audio_source)`
- `stop_live_session()`
- `process_audio_file(file_path: str)`
- `subscribe_to_results(callback_function)`

**Dependency Rules:**

- Can import from: `config/`, `utils/` (and abstract interfaces of `ai/`,
  `audio/`, `translation/`).
- Must **NOT** import from: `ui/`.

---

## 3. `hardware/` (Resource Detection)

**Responsibilities:** Detects and monitors the underlying host machine
properties. Answers questions about available RAM, CPU cores, and GPU
capabilities.

**Classes:**

- `HardwareDetector`: Scans the system for OpenCL/CUDA acceleration.
- `ResourceMonitor`: Polls the active usage of CPU/GPU and RAM.

**Public Interfaces:**

- `get_system_info() -> dict`: Returns total RAM, CPU model, GPU model.
- `is_cuda_available() -> bool`
- `get_vram_capacity() -> float`

**Dependency Rules:**

- Can import from: `utils/`.
- Depended on by: `core/`, `ai/` (to decide which model size to load based on
  VRAM).

---

## 4. `ai/` (Local Speech & Diarization Engine)

**Responsibilities:** Encapsulates all logic related to PyTorch, Whisper, and
pyannote. It handles model loading, inference execution, and VRAM management.

**Classes:**

- `SpeechRecognizer`: Wrapper around the Whisper model.
- `SpeakerSegmenter`: Wrapper around the pyannote.audio diarization pipeline.
- `ModelLoader`: Factory class that dynamically instantiates and loads weights
  based on hardware limits.

**Public Interfaces:**

- `load_model(model_size: str, compute_mode: str)`
- `transcribe_audio(audio_chunk: np.ndarray) -> str`
- `segment_speakers(audio_file: str) -> List[SpeakerSegment]`

**Dependency Rules:**

- Can import from: `config/`, `hardware/`, `utils/`.
- Must **NOT** import from: `ui/`, `core/`.

---

## 5. `audio/` (Audio Capture & Pre-processing)

**Responsibilities:** Interacts directly with system sound interfaces
(Microphone, Stereo Mix). It formats, chunks, and queues the raw byte streams
into NumPy arrays suitable for the AI engine.

**Classes:**

- `MicrophoneStream`: Handles live audio input using `sounddevice`.
- `SystemAudioCapture`: Captures loopback audio (Windows WASAPI).
- `AudioFileProcessor`: Loads and normalizes `.mp3`/`.wav`/`.mp4` using
  `ffmpeg`.

**Public Interfaces:**

- `start_stream(callback: callable)`: Begins pushing raw audio frames to a
  callback or Queue.
- `stop_stream()`
- `extract_audio_from_video(video_path: str) -> str`

**Dependency Rules:**

- Can import from: `config/`, `utils/`.
- Generates data consumed by: `core/`.

---

## 6. `translation/` (Language Translation)

**Responsibilities:** Handles translation of text sequences using local
HuggingFace NLP models (e.g., MarianMT, NLLB).

**Classes:**

- `TranslationEngine`: Loads the neural translation models and pipelines.
- `LanguageDetector`: (Optional) detects the language of a pure text string if
  Whisper didn't provide strict confidence.

**Public Interfaces:**

- `load_translator(source_lang: str, target_lang: str)`
- `translate_text(text: str) -> str`

**Dependency Rules:**

- Can import from: `config/`, `hardware/`, `utils/`.
- Consumed purely by: `core/` (after transcription is complete).

---

## 7. `utils/` (Shared Utilities)

**Responsibilities:** Provides generic helper functions that do not belong to
any specific domain. Includes formatting, logging, and generic file I/O
operations.

**Classes:**

- `Logger`: Centralized logging configuration (e.g., to console and `.log`
  file).
- `FileExporter`: Turns JSON transcription objects into `.txt`, `.docx`, or
  `.srt` files.
- `TimeFormatter`: Formats total seconds into standard `HH:MM:SS` strings.

**Public Interfaces:**

- `export_srt(transcription_data, output_path)`
- `setup_logging(level)`
- `format_timestamp(seconds: float) -> str`

**Dependency Rules:**

- Can import from: Strictly standard libraries.
- Depended on by: **All** modules.

---

## 8. `ui/` (User Interface - PySide6)

**Responsibilities:** The visual presentation layer. It displays settings,
transcription streams, and hardware info to the user. It translates user
interactions (clicks, toggles) into commands for the `core/`.

**Classes:**

- `MainWindow`: The primary shell holding all tabs.
- `TranscribePanel`: The real-time chat/subtitle viewer.
- `SettingsPanel`: Sliders and dropdowns to modify the `config/`.
- `SignalManager`: Maps PySide6 QThread signals from background workers to main
  thread GUI updates.

**Public Interfaces:**

- `update_transcript_view(text: str)`
- `update_hardware_dashboard(stats: dict)`
- `show_error_dialog(message: str)`

**Dependency Rules:**

- Can import from: **All** other modules (as it is the top-level orchestration
  layer holding the instances of `core/` and initializing the app).
- Must **NOT** be imported by **ANY** other module (to keep business logic
  decoupled from PySide6).
