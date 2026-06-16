"""Streaming Parakeet 240ms: file sim, word emission timing + CPU load check."""
import time
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx

PK = Path(__file__).parent / "sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-streaming-240ms"

recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens=str(PK / "tokens.txt"),
    encoder=str(PK / "encoder.int8.onnx"),
    decoder=str(PK / "decoder.int8.onnx"),
    joiner=str(PK / "joiner.int8.onnx"),
    num_threads=8,
    sample_rate=16000,
    feature_dim=80,
    enable_endpoint_detection=True,
    model_type="nemo_transducer",
)

wav_path = Path(__file__).parent / "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17" / "test_wavs" / "0.wav"
with wave.open(str(wav_path)) as f:
    samples = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
samples = samples.astype(np.float32) / 32768.0
dur = len(samples) / 16000

CHUNK = 1600
stream = recognizer.create_stream()
last = ""
proc = []
for i in range(0, len(samples), CHUNK):
    t0 = time.perf_counter()
    stream.accept_waveform(16000, samples[i:i + CHUNK])
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    proc.append(time.perf_counter() - t0)
    text = recognizer.get_result(stream)
    if text != last:
        print(f"[{(i+CHUNK)/16000:5.2f}s] {text!r}")
        last = text

print()
print(f"avg {np.mean(proc)*1000:.0f} ms / 100 ms chunk, max {np.max(proc)*1000:.0f} ms, "
      f"RTF {np.sum(proc)/dur:.3f}")
