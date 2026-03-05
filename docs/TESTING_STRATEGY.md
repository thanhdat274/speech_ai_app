# Speech AI Desktop App - Testing Strategy

## Pre-Release Checklist (QA Protocol)

### 1. Hardware Variation Tests

- [ ] **High-End GPU Machine (RTX 4090 / 3090, 24GB VRAM)**
  - Load `large-v3` model. Ensure no OOM errors.
  - Run inference with NLLB Translation + Pyannote simultaneous execution.
  - VRAM tracking stays under 12GB.
- [ ] **Mid-Range GPU Machine (RTX 3060 / 2060, 6GB-8GB VRAM)**
  - Load `medium` model. Ensure no OOM.
  - Test Pyannote -> OOM -> CPU fallback behavior implicitly works if forced.
  - Ensure VRAM does not overflow OS threshold.
- [ ] **CPU-Only / Integrated Graphics Machine (Intel UHD / Iris Xe)**
  - Ensure `AIEngine.device == "cpu"`.
  - Application defaults to `tiny` or `base` for transcription.
  - UI remains fully responsive during pure-CPU 100% load (threading works).
- [ ] **Low RAM Machine (< 8GB System RAM)**
  - Ensure model loads securely without swapping OS to death.
  - Test application memory consumption on idle (< 300MB).

### 2. Audio Payload Tests

- [ ] **Microphone Realtime Stream**
  - Connect standard generic microphone. Verify recording callbacks are injected
    properly.
  - Ensure words are not truncated horizontally (`overlap` mechanics logic works
    correctly).
- [ ] **System Audio / WASAPI Capture**
  - Verify app captures YouTube audio running in Chrome.
- [ ] **Long Audio File Test (3+ hours)**
  - Verify `ffmpeg` can load a large MP4/MP3.
  - Memory usage must remain stable across transcription (no accumulating
    footprint leak).
  - Verify UI progress bar updates sequentially rather than stalling.
- [ ] **Multi-Speaker Diarization Test**
  - Test a known podcast with 3+ speakers overlapping.
  - Verify Pyannote identifies distinct speaker segments correctly in the
    Transcript UI.

### 3. Edge-case & Error Handling

- [ ] **Invalid File Dropout**
  - Drag and drop `.pdf` or corrupted `.WAV` file. UI should catch error via
    signal without exploding backend.
- [ ] **Memory Leak Test (Repetitive Execution)**
  - Run transcription on the same file 10 times consecutively.
  - VRAM and RAM footprint must not exceed 20% growth over total session
    duration (verify `empty_cache()` hooks work).
- [ ] **Abrupt Cancellation (Interrupt)**
  - Test clicking "Cancel Transcription" mid-way. Ensure threads are gracefully
    destroyed (no orphaned processes/zombie PyTorch ops).
- [ ] **Missing FFMPEG Error**
  - Uninstall `ffmpeg` from system PATH and run. App must show user-friendly
    error dialog referencing PATH configuration, no unhandled stack trace.
