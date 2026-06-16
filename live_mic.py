"""Live word-by-word STT from microphone. Ctrl+C to stop.
Prints partial words as you speak; endpoint detection finalizes each utterance."""
import sys
from pathlib import Path

import numpy as np
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
    rule1_min_trailing_silence=2.4,   # endpoint after 2.4 s silence, no speech yet
    rule2_min_trailing_silence=0.8,   # endpoint after 0.8 s silence following speech
)

SAMPLE_RATE = 16000
CHUNK = 1600  # 100 ms

stream = recognizer.create_stream()
last_text = ""
utterance = 0

print("Listening... speak now (Ctrl+C to stop)")
with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                    blocksize=CHUNK) as mic:
    try:
        while True:
            chunk, _ = mic.read(CHUNK)
            stream.accept_waveform(SAMPLE_RATE, chunk[:, 0])
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)

            text = recognizer.get_result(stream)
            if text != last_text:
                sys.stdout.write("\r" + " " * 100 + "\r")
                sys.stdout.write(text)
                sys.stdout.flush()
                last_text = text

            if recognizer.is_endpoint(stream):
                if last_text:
                    utterance += 1
                    sys.stdout.write(f"\r[{utterance}] {last_text}\n")
                    sys.stdout.flush()
                recognizer.reset(stream)
                last_text = ""
    except KeyboardInterrupt:
        print("\nStopped.")
