"""
test_loopback2.py - Test PyAudioWPatch WASAPI Loopback
"""
import time
import wave
import numpy as np

print("=" * 60)
print("  PyAudioWPatch WASAPI LOOPBACK TEST")
print("=" * 60)

try:
    import pyaudiowpatch as pyaudio
    print("✅ pyaudiowpatch imported successfully!")
except ImportError:
    print("❌ pyaudiowpatch not installed!")
    input()
    exit()

p = pyaudio.PyAudio()

# Find WASAPI
try:
    wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    print(f"\n✅ WASAPI Host API found!")
    print(f"   Default Output: index {wasapi_info['defaultOutputDevice']}")
except OSError:
    print("❌ WASAPI not available!")
    p.terminate()
    input()
    exit()

# Get default speakers
default_speakers = p.get_device_info_by_index(wasapi_info['defaultOutputDevice'])
print(f"   Speakers: {default_speakers['name']}")

# Find loopback devices
print("\n  All Loopback Devices:")
loopback_device = None
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if dev.get('isLoopbackDevice', False):
        match = "✅ MATCH" if default_speakers['name'] in dev['name'] else ""
        print(f"   [{i}] {dev['name']} (ch:{dev['maxInputChannels']}, rate:{dev['defaultSampleRate']}) {match}")
        if default_speakers['name'] in dev['name'] and loopback_device is None:
            loopback_device = dev

if loopback_device is None:
    # Use first loopback
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev.get('isLoopbackDevice', False):
            loopback_device = dev
            break

if loopback_device is None:
    print("\n❌ No loopback device found!")
    p.terminate()
    input()
    exit()

print(f"\n  Using: {loopback_device['name']}")
print(f"  Rate: {int(loopback_device['defaultSampleRate'])}Hz, Channels: {loopback_device['maxInputChannels']}")

# Record 5 seconds
print("\n>> Make sure VIDEO/MUSIC is playing! Speaker CAN be MUTED! <<\n")

native_rate = int(loopback_device['defaultSampleRate'])
channels = loopback_device['maxInputChannels']
frames = []

def callback(in_data, frame_count, time_info, flags):
    audio = np.frombuffer(in_data, dtype=np.float32)
    if channels >= 2:
        audio = audio.reshape(-1, channels).mean(axis=1)
    frames.append(audio)
    return (None, pyaudio.paContinue)

stream = p.open(
    format=pyaudio.paFloat32,
    channels=channels,
    rate=native_rate,
    input=True,
    input_device_index=loopback_device['index'],
    frames_per_buffer=int(native_rate * 0.1),
    stream_callback=callback
)

stream.start_stream()
for i in range(5, 0, -1):
    print(f"  Recording... {i}s")
    time.sleep(1)

stream.stop_stream()
stream.close()
p.terminate()

if frames:
    audio = np.concatenate(frames)
    max_vol = np.max(np.abs(audio))
    mean_vol = np.mean(np.abs(audio))
    
    print(f"\n  Max Volume:  {max_vol:.5f}")
    print(f"  Mean Volume: {mean_vol:.5f}")
    
    if max_vol < 0.001:
        print("\n  ❌ No audio captured through loopback.")
    else:
        print(f"\n  ✅ LOOPBACK WORKS! Audio captured at volume {max_vol:.3f}")
        
        # Resample to 16kHz
        duration = len(audio) / native_rate
        target_len = int(duration * 16000)
        audio_16k = np.interp(
            np.linspace(0, duration, target_len, endpoint=False),
            np.linspace(0, duration, len(audio), endpoint=False),
            audio
        ).astype(np.float32)
        
        # Save
        with wave.open("test_loopback_capture.wav", 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes((audio_16k * 32767).astype(np.int16).tobytes())
        print("  Saved: test_loopback_capture.wav")
        
        # Whisper test
        print("\n  Running Whisper AI...")
        try:
            from faster_whisper import WhisperModel
            model = WhisperModel("base", device="cuda", compute_type="int8_float16")
            
            segs, info = model.transcribe(audio_16k, beam_size=5, language="vi", vad_filter=True)
            text = "".join([s.text for s in segs]).strip()
            print(f"  Language: {info.language}")
            print(f"  Text: {text}")
        except Exception as e:
            print(f"  Whisper Error: {e}")
else:
    print("\n  ❌ No frames captured")

print("\nDone. Press Enter to exit.")
input()
