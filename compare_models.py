"""Decode sample.wav with all candidate models, print transcripts + timing."""
import os
import pathlib
import time
import wave

import numpy as np

APP_DIR = pathlib.Path(__file__).parent
VENV_NVIDIA = APP_DIR / ".venv" / "Lib" / "site-packages" / "nvidia"
dll_dirs = [str(d) for d in VENV_NVIDIA.rglob("*")
            if d.is_dir() and any(f.suffix == ".dll" for f in d.iterdir() if f.is_file())]
os.environ["PATH"] = os.pathsep.join(dll_dirs) + os.pathsep + os.environ["PATH"]

import sherpa_onnx
from faster_whisper import WhisperModel

WAV = str(APP_DIR / "sample.wav")
with wave.open(WAV) as f:
    n = f.getnframes()
    sr = f.getframerate()
    samples = np.frombuffer(f.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
dur = n / sr
print(f"sample.wav: {dur:.1f}s\n")

PROMPT = (APP_DIR / "prompt.txt").read_text(encoding="utf-8").strip()


def bench(name, fn, runs=2):
    best, text = None, ""
    for _ in range(runs):
        t0 = time.perf_counter()
        text = fn()
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    print(f"=== {name} ===  {best:.2f}s  RTF {best/dur:.3f}")
    print(text.strip(), "\n")


# 1. current: distil-large-v3.5 GPU (with vocab prompt, as in the app)
m1 = WhisperModel("distil-whisper/distil-large-v3.5-ct2", device="cuda", compute_type="float16")
bench("distil-large-v3.5 (GPU, current)", lambda: " ".join(
    s.text for s in m1.transcribe(samples, language="en", beam_size=5,
                                  condition_on_previous_text=False,
                                  initial_prompt=PROMPT)[0]))
del m1

# 2. whisper large-v3-turbo GPU
m2 = WhisperModel("deepdml/faster-whisper-large-v3-turbo-ct2", device="cuda", compute_type="float16")
bench("large-v3-turbo (GPU)", lambda: " ".join(
    s.text for s in m2.transcribe(samples, language="en", beam_size=5,
                                  condition_on_previous_text=False,
                                  initial_prompt=PROMPT)[0]))
del m2

# 3. parakeet-tdt-0.6b-v2 int8 CPU (sherpa-onnx offline transducer)
PK = APP_DIR / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
rec = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder=str(PK / "encoder.int8.onnx"),
    decoder=str(PK / "decoder.int8.onnx"),
    joiner=str(PK / "joiner.int8.onnx"),
    tokens=str(PK / "tokens.txt"),
    num_threads=8,
    model_type="nemo_transducer",
)

def parakeet():
    st = rec.create_stream()
    st.accept_waveform(sr, samples)
    rec.decode_stream(st)
    return st.result.text

bench("parakeet-tdt-0.6b-v2 int8 (CPU)", parakeet)
