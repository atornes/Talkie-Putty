"""Kokoro TTS smoke test: synth speed + playback."""
import time
from pathlib import Path

import sounddevice as sd
import sherpa_onnx

K = Path(__file__).parent / "kokoro-int8-multi-lang-v1_1"

t0 = time.perf_counter()
tts = sherpa_onnx.OfflineTts(
    sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=str(K / "model.int8.onnx"),
                voices=str(K / "voices.bin"),
                tokens=str(K / "tokens.txt"),
                data_dir=str(K / "espeak-ng-data"),
                dict_dir=str(K / "dict"),
                lexicon=f"{K / 'lexicon-us-en.txt'},{K / 'lexicon-zh.txt'}",
            ),
            num_threads=8,
        ),
        max_num_sentences=1,
    )
)
print(f"load: {time.perf_counter()-t0:.1f}s, voices: {tts.num_speakers}")

TEXT = ("Ready to dictate. The merge conflict in the CI pipeline is resolved, "
        "and all end to end tests are passing.")
t0 = time.perf_counter()
audio = tts.generate(TEXT, sid=0, speed=1.0)
dt = time.perf_counter() - t0
dur = len(audio.samples) / audio.sample_rate
print(f"synth {dur:.1f}s audio in {dt:.2f}s  RTF {dt/dur:.3f}")

sd.play(audio.samples, audio.sample_rate)
sd.wait()
print("playback done")
