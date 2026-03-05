# Hardware Auto-Optimization Logic

## 1. Objective

To ensure the Speech AI application runs efficiently across diverse Windows
hardware configurations (from high-end RTX 4090 desktops to low-end Intel
Celeron laptops) without user intervention, while preventing Out-of-Memory (OOM)
crashes.

## 2. Detection Mechanism

Upon initialization, `AIEngine._detect_hardware()` executes the following
checks:

### CPU Detection

- **Method**: `psutil.cpu_count(logical=False)` (Physical Cores) &
  `logical=True` (Threads).
- **Purpose**: Determines the number of background threads for operations like
  `ffmpeg` audio extraction and CPU-fallback inference (`PyTorch` intra-op
  threads).

### System RAM Detection

- **Method**: `psutil.virtual_memory().available`
- **Purpose**: Defines the maximum size of models that can be loaded into system
  memory.
  - **Rule**: Never consume > 70% of available RAM to avoid OS pagefile
    thrashing.

### GPU VRAM Detection

- **Method**: `torch.cuda.is_available()` and
  `torch.cuda.get_device_properties(0).total_memory`
- **Purpose**: Checks if a CUDA-enabled GPU is present and measures exact VRAM
  space.
  - VRAM is the primary bottleneck for LLMs/Whisper models.

---

## 3. Auto-Selection Logic

The `AIEngine._auto_select_whisper_model()` method applies strict thresholds.

### Compute Mode & Precision

```python
if gpu_vram_gb >= 2.0:
    device = "cuda"
    compute_type = torch.float16 # Uses Tensor Cores, half VRAM footprint
else:
    # No GPU or sub-2GB GPU -> CPU Fallback
    device = "cpu"
    compute_type = torch.float32 # CPUs generally lack robust FP16 support natively
```

### Whisper Model Heuristic

Assuming FP16 precision on GPU (or FP32 on CPU), models consume VRAM/RAM as
follows (approx):

- `tiny`: ~0.5 GB
- `base`: ~0.7 GB
- `small`: ~1.5 GB
- `medium`: ~3.5 GB
- `large-v3`: ~6.0 GB

**Selection Rules:**

1. **Low End** (VRAM < 2GB or RAM < 4GB): Load `tiny` or `base`. High risk of
   swapping; prioritize stability over extreme accuracy.
2. **Medium End** (VRAM 2GB - 4GB): Load `small`. Good balance of speed and
   accuracy.
3. **High End** (VRAM 4GB - 7GB): Load `medium`. Very accurate, standard for
   production.
4. **Enthusiast** (VRAM > 7GB): Load `large-v3`. Best-in-class accuracy.

### Fallback Mitigation (CUDA OOM)

Even with heuristics, Diarization (Pyannote) and Translation (NLLB) pipelines
consume memory. If a pipeline starts and `torch.cuda.OutOfMemoryError` is
thrown:

1. `AIEngine` catches the exception.
2. `AIEngine.free_memory()` is called (`torch.cuda.empty_cache()`).
3. The engine permanently downgrades the offending model to `device="cpu"` and
   retries without crashing the UI.

## 4. Scalable Design & User Override

- **Config Injection**: Heuristics are stored in a centralized configuration
  dictionary/file, not hardcoded into the pipeline execution. This allows
  updating thresholds remotely if new optimized models are released (e.g.,
  `distil-whisper`).
- **User Preference**: The `SettingsPanel` (UI) exposes "Performance Mode"
  (Speed, Balanced, Quality) and "Compute Mode" (Auto, Force GPU, Force CPU).
- **Override Trigger**: If a user selects "Force GPU" despite having 2GB VRAM
  and requests the `large-v3` model, the system attempts it but falls back
  cleanly per the mitigation rules, alerting the user via the `AppController`.
