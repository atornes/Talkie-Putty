"""Simulate real-time streaming: feed a test wav in 100ms chunks,
print words as they are emitted, measure per-chunk processing time."""
import time
import wave
from pathlib import Path

import numpy as np
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
)

wav_path = MODEL_DIR / "test_wavs" / "0.wav"
with wave.open(str(wav_path)) as f:
    assert f.getframerate() == 16000, f.getframerate()
    samples = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
samples = samples.astype(np.float32) / 32768.0

CHUNK = 1600  # 100 ms at 16 kHz
stream = recognizer.create_stream()
last_text = ""
proc_times = []

for i in range(0, len(samples), CHUNK):
    chunk = samples[i : i + CHUNK]
    t0 = time.perf_counter()
    stream.accept_waveform(16000, chunk)
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    proc_times.append(time.perf_counter() - t0)

    text = recognizer.get_result(stream)
    if text != last_text:
        new = text[len(last_text):]
        audio_pos = (i + len(chunk)) / 16000
        print(f"[{audio_pos:6.2f}s audio] +{new!r}  (chunk proc {proc_times[-1]*1000:.1f} ms)")
        last_text = text

print()
print(f"Final: {last_text}")
print(f"Chunks: {len(proc_times)}, avg proc {np.mean(proc_times)*1000:.1f} ms, "
      f"max {np.max(proc_times)*1000:.1f} ms per 100 ms chunk")
print(f"Real-time factor: {np.sum(proc_times) / (len(samples)/16000):.3f} (must be < 1.0)")
