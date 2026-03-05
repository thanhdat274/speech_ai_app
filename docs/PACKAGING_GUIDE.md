# PyInstaller Packaging Guide (Windows)

## Introduction

Packaging an AI application like Speech AI on Windows requires careful handling
of heavy libraries like PyTorch, Pyannote, HuggingFace transformers, and ffmpeg
dependencies. Use this guide to create a standalone binary.

## 1. Preparing the Environment

Ensure your virtual environment is clean. Heavy development dependencies (like
`jupyter`, `pytest`) bloat the executable.

```powershell
pip install pyinstaller auto-py-to-exe
```

## 2. One-Folder vs One-File Approach

**Recommendation: One-Folder (`-D` or `--onedir`)**

- **Pros of One-File (`-F`)**: Clean distribution (a single `.exe`).
- **Cons of One-File**: Huge startup delay (it extracts gigabytes of PyTorch
  DLLs to `C:\Temp` every launch). High risk of anti-virus false positives.
- **Pros of One-Folder**: Instant startup. Easy update patching (replace
  individual `.pyc` files).
- **Cons**: User sees a cluttered folder if not wrapped in an InnoSetup
  installer.

## 3. The PyInstaller Spec File Setup

Run the following base command to generate `app.spec`:

```powershell
pyinstaller --name "SpeechAI" --windowed --icon "assets/icon.ico" ui_main.py
```

_Note: We package `ui_main.py` (PySide6) as the entry point._

### Fixing Hidden Imports & DLLs

Edit the generated `app.spec` to manually include AI backends that PyInstaller's
AST parser misses:

```python
hiddenimports=[
    'whisper', 'torch', 'torchaudio', 'torchvision',
    'pyannote.audio', 'transformers', 'accelerate',
    'scipy', 'sounddevice', 'numpy', 'psutil'
],
```

### Exclusions (Optimizing File Size)

Many libraries ship with test suites and unnecessary bloat (like Qt WebEngine if
unused). Remove them:

```python
excludes=['matplotlib', 'IPython', 'notebook', 'PySide6.QtWebEngine', 'PySide6.QtWebEngineWidgets', 'PySide6.QtNetwork'],
```

### Bundling FFMPEG

To avoid requiring users to install FFMPEG manually, bundle it:

1. Download a static FFMPEG Windows build (`ffmpeg.exe`, `ffprobe.exe`).
2. Place them in a folder `bins/`.
3. Add to `app.spec`:

```python
datas=[
    ('bins/ffmpeg.exe', 'bins/'),
    ('bins/ffprobe.exe', 'bins/')
],
```

_In your `audio_engine.py`, ensure paths dynamically adjust to `sys._MEIPASS`
when running as an executable._

### HuggingFace Models Handling

**Strategy:** Avoid bundling models inside the EXE. Reason: A 2GB EXE is
terrible for updates. Models should be side-loaded. Set your system to download
models to a user-writable path:

```python
import os
os.environ["HF_HOME"] = os.path.join(os.path.expanduser("~"), ".speech_ai", "models")
```

## 4. Final Build Command

Once `app.spec` is perfected:

```powershell
pyinstaller --clean app.spec
```

## 5. Deployment Post-Steps

1. Take the output in `dist/SpeechAI/`.
2. Run tools like `Enigma Virtual Box` or `InnoSetup` to create an installer
   wizard (`SpeechAI_Setup_v3.exe`).
3. Add an uninstaller inside `InnoSetup`.
4. Sign the final installer with a Code Signing Certificate (Authenticode) to
   prevent Windows SmartScreen blocks.
