"""Record a 25 s test clip to sample.wav for model comparison."""
import sys
import time
import wave

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
DURATION = 25

print(f"Recording {DURATION}s — speak technical sentences, mumble some on purpose...", flush=True)
audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
for i in range(DURATION):
    time.sleep(1)
    print(f"  {DURATION - i - 1}s left", flush=True)
sd.wait()

pcm = (np.clip(audio[:, 0], -1, 1) * 32767).astype(np.int16)
with wave.open(r"C:\dev\Talkie-Putty\sample.wav", "wb") as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(SAMPLE_RATE)
    f.writeframes(pcm.tobytes())
print(f"Saved sample.wav, peak level {np.abs(audio).max():.3f}", flush=True)
