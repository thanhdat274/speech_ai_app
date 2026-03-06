"""
test_audio_diagnostic.py
==========================
Chạy file này để chẩn đoán: Âm thanh hệ thống có vào được app không?
Và AI có nhận diện được tiếng Việt không?

Cách chạy: Mở video tiếng Việt, rồi chạy file này.
"""
import sys
import time
import wave
import struct
import numpy as np
import sounddevice as sd

print("=" * 60)
print("  SPEECH AI - AUDIO DIAGNOSTIC TOOL")
print("=" * 60)

# ==========================================
# STEP 1: Liệt kê tất cả thiết bị âm thanh
# ==========================================
print("\n[STEP 1] Listing all audio devices...\n")
devices = sd.query_devices()
hostapis = sd.query_hostapis()

wasapi_idx = -1
for i, h in enumerate(hostapis):
    if "WASAPI" in h['name']:
        wasapi_idx = i
        break

print(f"  WASAPI Host API Index: {wasapi_idx}")
if wasapi_idx >= 0:
    host_info = sd.query_hostapis(wasapi_idx)
    print(f"  WASAPI Default Output: {host_info.get('default_output_device')}")
    print(f"  WASAPI Default Input:  {host_info.get('default_input_device')}")

print(f"\n  {'Idx':<5} {'API':<10} {'In':<5} {'Out':<5} {'Name'}")
print(f"  {'-'*60}")

wasapi_inputs = []
for i, d in enumerate(devices):
    api_name = hostapis[d['hostapi']]['name'][:8]
    in_ch = d['max_input_channels']
    out_ch = d['max_output_channels']
    marker = ""
    if d['hostapi'] == wasapi_idx and in_ch > 0:
        wasapi_inputs.append(i)
        marker = " <-- WASAPI INPUT"
    if d['hostapi'] == wasapi_idx and out_ch > 0:
        marker = " <-- WASAPI OUTPUT"
    print(f"  {i:<5} {api_name:<10} {in_ch:<5} {out_ch:<5} {d['name']}{marker}")

# ==========================================
# STEP 2: Thu âm thanh hệ thống 5 giây
# ==========================================
print("\n" + "=" * 60)
print("[STEP 2] Recording 5 seconds of system audio...")
print("  >> Make sure a VIDEO or MUSIC is playing NOW! <<")
print("=" * 60)

# Try each WASAPI input device
test_device = None
test_rate = 44100
audio_data = []

# First try Stereo Mix (it was known to work)
for idx in wasapi_inputs:
    name = devices[idx]['name'].lower()
    if "stereo mix" in name:
        test_device = idx
        test_rate = int(devices[idx]['default_samplerate'])
        print(f"\n  Testing device [{idx}]: {devices[idx]['name']} @ {test_rate}Hz")
        break

if test_device is None and len(wasapi_inputs) > 0:
    test_device = wasapi_inputs[0]
    test_rate = int(devices[test_device]['default_samplerate'])
    print(f"\n  Testing device [{test_device}]: {devices[test_device]['name']} @ {test_rate}Hz")

if test_device is None:
    print("\n  ERROR: No WASAPI input device found!")
    sys.exit(1)

# Record
try:
    recording = sd.rec(int(5 * test_rate), samplerate=test_rate, channels=1, 
                       device=test_device, dtype='float32')
    for i in range(5, 0, -1):
        print(f"  Recording... {i}s remaining")
        time.sleep(1)
    sd.wait()
    
    audio_flat = recording.flatten()
    max_vol = np.max(np.abs(audio_flat))
    mean_vol = np.mean(np.abs(audio_flat))
    
    print(f"\n  [RESULT] Max Volume:  {max_vol:.5f}")
    print(f"  [RESULT] Mean Volume: {mean_vol:.5f}")
    
    if max_vol < 0.0001:
        print("\n  ❌ NO AUDIO DETECTED! The device is silent.")
        print("  SOLUTION: Try un-muting your speakers or use a different device.")
        print("  Try running again with device index argument.")
    elif max_vol < 0.01:
        print("\n  ⚠️  VERY WEAK audio detected. Signal might be too quiet for AI.")
    else:
        print(f"\n  ✅ AUDIO CAPTURED SUCCESSFULLY! (Volume: {max_vol:.3f})")
    
except Exception as e:
    print(f"\n  ERROR recording: {e}")
    sys.exit(1)

# ==========================================
# STEP 3: Nhận diện bằng Whisper
# ==========================================
if max_vol > 0.001:
    print("\n" + "=" * 60)
    print("[STEP 3] Running Whisper AI on captured audio...")
    print("=" * 60)
    
    # Resample to 16000Hz for Whisper
    duration = len(audio_flat) / test_rate
    target_len = int(duration * 16000)
    audio_16k = np.interp(
        np.linspace(0, duration, target_len, endpoint=False),
        np.linspace(0, duration, len(audio_flat), endpoint=False),
        audio_flat
    ).astype(np.float32)
    
    # Save WAV for debugging
    wav_path = "test_capture.wav"
    with wave.open(wav_path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes((audio_16k * 32767).astype(np.int16).tobytes())
    print(f"  Saved audio to: {wav_path}")
    
    try:
        from faster_whisper import WhisperModel
        
        print("  Loading Whisper model 'base'...")
        model = WhisperModel("base", device="cuda", compute_type="int8_float16")
        
        # Test 1: Auto-detect language
        print("\n  --- Test 1: Auto-Detect Language ---")
        segments, info = model.transcribe(audio_16k, beam_size=5, vad_filter=True)
        text = "".join([s.text for s in segments]).strip()
        print(f"  Detected Language: {info.language} (prob: {info.language_probability:.2f})")
        print(f"  Text: {text}")
        
        # Test 2: Force Vietnamese
        print("\n  --- Test 2: Force Vietnamese ---")
        segments, info = model.transcribe(audio_16k, beam_size=5, language="vi", vad_filter=True)
        text = "".join([s.text for s in segments]).strip()
        print(f"  Text (vi): {text}")
        
        # Test 3: Force English
        print("\n  --- Test 3: Force English ---")
        segments, info = model.transcribe(audio_16k, beam_size=5, language="en", vad_filter=True)
        text = "".join([s.text for s in segments]).strip()
        print(f"  Text (en): {text}")
        
        print("\n" + "=" * 60)
        print("  DIAGNOSTIC COMPLETE!")
        print("=" * 60)
        
    except Exception as e:
        print(f"  Whisper Error: {e}")

else:
    print("\n  Skipping AI test due to no audio.")
    
print("\nDone. Press Enter to exit.")
input()
