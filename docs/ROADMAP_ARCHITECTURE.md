# Architecture Roadmap & Scalability Guide

## 1. Goal

To evolve Speech AI from a monolithic standalone Desktop App into a scalable,
extensible ecosystem enabling dynamic AI updates and offline-first
functionalities without re-compiling the main application.

## 2. Core Upgrades

### Auto Model Downloader

**Current state:** Users manually resolve PyTorch/HF pipelines on
initialization. **Future Architecture:**

- Submodule `ModelManager` orchestrates asynchronous chunked downloads from
  HuggingFace via `HTTPX`.
- Downloads are verified via SHA256 hashes.
- Resume-on-fail mechanisms ensure massive files (2-6GB) aren't corrupted by
  network drops.
- `QProgressBar` connects directly to the download stream in the UI.

### Plugin System (Dynamic Extension)

**Architecture:**

- A `plugins/` directory allows users or third-parties to drop `.py` modules.
- On startup, the `AppController` uses `importlib.util` and
  `pkgutil.walk_packages` to iterate plugins dynamically.
- Plugins register themselves to predefined UI slots (e.g., adding a new button
  to the Sidebar) and expose execution interfaces (`IPlugin` abstract class).
- Immediate use case: Video Subtitle Export (`.srt` parsing engine).

### Subtitle Generator (`.srt` / `.vtt`) integration

- Abstract `AIEngine.transcribe_audio()` to return chunk-level timestamps
  natively.
- Send timestamps + decoded text to `SubtitleEngine`.
- Provide generic `.write(format="srt")` function directly hooked into UI's
  "Export" dialog.
- Scalable because it leverages the existing metadata rather than re-inventing
  the inference loop.

### AI Summarization Module

- Introduction of an embedded LLM architecture (e.g., `llama-cpp-python` or
  `Transformers` running lightweight models like Phi-3 or Qwen).
- Positioned as a separate tab in `QStackedWidget`.
- Operates strictly after the transcript resolves.
- Given the system already manages GPU/RAM (`AudioEngine` and
  `TranslationEngine`), the summarization logic securely taps into
  `AIEngine.free_memory()` before loading its LLM into VRAM to prevent OOM
  conflicts.

### Dark/Light Theme Engine

- Expand `Theme` configuration into an injected Qt `QPalette`.
- Establish CSS template configurations where primary, accent, and background
  colors reference standard hex tokens.
- Add toggle switch inside `SettingsPanel` connected to
  `AppController.set_theme()`.

### Offline Smart-Updates

- Instead of forcing an app reinstall, implement a `Updater` service.
- The service downloads a signed `core.zip` containing patched `.pyc` logic
  files.
- Replaces backend Python scripts natively and restarts the embedded python
  interpreter (useful when delivering AI bug fixes independently of GUI
  updates).

## 3. Scalability Conclusion

By adhering to strict MVC structure defined in `ui_main.py` and maintaining
decoupled singleton backends (`AIEngine`, `AudioEngine`, `TranslationEngine`),
future expansion teams can iterate the AI models independently from the PySide6
UI without breaking the production builds. The event-driven Qt architecture
ensures threads communicate seamlessly regardless of compute weight.
