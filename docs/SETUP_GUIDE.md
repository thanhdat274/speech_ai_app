# Local AI Speech App - Windows Environment Setup Guide

Follow this step-by-step checklist to prepare a professional development
environment on Windows.

## 1. Prerequisites

- [ ] Install **Python 3.10** or **3.11** (64-bit) from the
      [official Python website](https://www.python.org/downloads/).
  - _Important:_ Check the box **"Add Python to PATH"** during installation.
- [ ] Install **FFmpeg** and add its `bin` folder to the system `PATH`
      environment variable.

## 2. Virtual Environment Setup

Open PowerShell or Command Prompt in your project root directory and run:

- [ ] Create a virtual environment named `venv`:
  ```bash
  python -m venv venv
  ```
- [ ] Activate the virtual environment:
  ```bash
  .\venv\Scripts\activate
  ```
- [ ] Upgrade the core Python packaging tools:
  ```bash
  python -m pip install --upgrade pip setuptools wheel
  ```

## 3. Library Installation

With the virtual environment activated, install the required packages:

- [ ] Install **PyTorch and TorchAudio** with CUDA support (adjust the CUDA
      version link if necessary based on your Nvidia driver, this uses CUDA 12.1
      as default):
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```
- [ ] Install the primary **AI Engines** and dependencies:
  ```bash
  pip install openai-whisper pyannote.audio transformers accelerate
  ```
- [ ] Install the **UI Framework** and hardware interaction libraries:
  ```bash
  pip install PySide6 sounddevice ffmpeg-python
  ```

## 4. Verification Checklists

After installation, run these diagnostic scripts to ensure the environment is
correctly configured.

- [ ] **Verify PyTorch Compatibility and CUDA Availability:**

  ```python
  python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
  ```

  _(Expected output: CUDA Available: True)_

- [ ] **GPU Detection Test:**

  ```python
  python -c "import torch; print(f'GPU Name: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else print('No GPU detected.')"
  ```

  _(Expected output: Prints your Nvidia GPU model name like "NVIDIA GeForce RTX
  4070")_

- [ ] **CPU Fallback Test:**
  ```python
  python -c "import torch; cpu_device = torch.device('cpu'); x = torch.rand(2, 2).to(cpu_device); print('CPU Tensor compute success!')"
  ```
  _(Expected output: "CPU Tensor compute success!")_
