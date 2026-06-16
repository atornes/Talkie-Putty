"""Timed live mic test: capture 15 s, print each partial as its own line."""
import time
from pathlib import Path

import sounddevice as sd
import sherpa_onnx

MODEL_DIR = Path(__file__).parent / "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"

recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens=str(MODEL_DIR / "tokens.txt"),
    encoder=str(MODEL_DIR / "encoder-epoch-99-avg-1.int8.onnx"),
    decoder=str(MODEL_DIR / "decoder-epoch-99-avg-1.int8.onnx"),
    joiner=str(MODEL_DIR / "joiner-epoch-99-avg-1.int8.onnx"),
    num_threads=4,
    sample_rate=16000,
    feature_dim=80,
    enable_endpoint_detection=True,
    rule2_min_trailing_silence=0.8,
)

SAMPLE_RATE = 16000
CHUNK = 1600
DURATION = 15

print(sd.query_devices(kind="input"), flush=True)
stream = recognizer.create_stream()
last_text = ""
utterance = 0
start = time.monotonic()

print(f"Recording {DURATION}s...", flush=True)
with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                    blocksize=CHUNK) as mic:
    peak = 0.0
    while time.monotonic() - start < DURATION:
        chunk, _ = mic.read(CHUNK)
        level = float(abs(chunk[:, 0]).max())
        peak = max(peak, level)
        if int((time.monotonic() - start) * 10) % 10 == 0:
            print(f"  level: {level:.4f} (peak so far {peak:.4f})", flush=True)
        stream.accept_waveform(SAMPLE_RATE, chunk[:, 0])
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)

        text = recognizer.get_result(stream)
        if text != last_text:
            t = time.monotonic() - start
            print(f"  [{t:5.2f}s] {text}", flush=True)
            last_text = text

        if recognizer.is_endpoint(stream):
            if last_text:
                utterance += 1
                print(f"FINAL [{utterance}]: {last_text}", flush=True)
            recognizer.reset(stream)
            last_text = ""

if last_text:
    print(f"FINAL [tail]: {last_text}", flush=True)
print("Done.", flush=True)
