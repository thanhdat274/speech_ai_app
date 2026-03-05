# PySide6 UI Architecture Plan

## 1. Overview

The PySide6 UI replaces the CustomTkinter prototype to provide a highly
professional, native-feeling, and scalable desktop application. It enforces a
strict separation between the Presentation Layer (UI) and the Business Logic
(AI, Audio) using the Controller/Service pattern and Qt's Signal/Slot mechanism.

## 2. Core Components

### `ui/main_window.py`

**Class: `MainWindow(QMainWindow)`**

- **Role**: The primary application container.
- **Layout**: Consists of a Sidebar (left) and a `QStackedWidget` (right) to
  switch between different main views (Transcript, Settings, Hardware Info).
- **Widgets**:
  - `SidebarWidget`: Navigation buttons (`QListWidget` or styled
    `QPushButton`s).
  - `TranscriptPanel`: `QTextEdit` displaying ongoing transcription, formatted
    with HTML/CSS for speaker highlighting.
  - `SettingsPanel`: UI for selecting compute mode, performance mode, language,
    and translation target.
  - `HardwarePanel`: Real-time hardware stats (CPU/GPU/RAM usage).
  - `StatusBar`: Indicates status (Idle, Recording, Transcribing) and
    application version.

### `core/app_controller.py`

**Class: `AppController(QObject)`**

- **Role**: Acts as the intermediary between the PySide6 UI and the backend
  modules (`AIEngine`, `AudioEngine`, `TranslationEngine`).
- **Mechanism**: The UI never imports or calls AI modules directly. It emits
  signals (e.g., `start_recording_requested`, `file_selected_signal`) to the
  `AppController`.
- **Background Processing**: Manages `Worker` threads using `QThread` and
  `QRunnable` to ensure the UI thread is never blocked during model loading,
  transcribing, or translating.

### `ui/workers.py`

**Class: `TranscriptionWorker(QThread)` / `AudioStreamWorker(QThread)`**

- **Role**: Executes long-running tasks.
- **Signals**: Emits `progress`, `result_ready`, or `error` signals back to the
  Controller, which then updates the UI.

## 3. Communication Flow (Signal/Slot)

1. **User Action**: User clicks "Start Transcription" in `TranscriptPanel`.
2. **UI Signal**: `TranscriptPanel` emits `transcription_requested(file_path)`.
3. **Controller Slot**: `AppController` catches the signal and spawns a
   `TranscriptionWorker` inside a `QThreadPool` or standalone `QThread`.
4. **Worker Execution**: `Worker` calls `AIEngine.transcribe_audio()`.
5. **Worker Signal**: Upon completion, `Worker` emits
   `transcription_finished(result_dict)`.
6. **Controller Slot**: `AppController` receives the raw dictionary, checks
   translations, formats it, and emits `update_transcript_ui(html_text)`.
7. **UI Update**: `TranscriptPanel` receives the formatted text and appends it
   to the `QTextEdit`.

## 4. Why This Architecture?

- **No Blocking**: Qt GUI thread runs at 60fps unaffected by Heavy AI loads.
- **Separation of Concerns**: UI developers can work on `ui/` while AI engineers
  work on `ai/` without merge conflicts.
- **Scalability**: Adding a new "AI Summarization" panel is as simple as adding
  a new widget to the `QStackedWidget` and a signal to the Controller.
