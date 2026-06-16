import os, pathlib, subprocess, time

base = pathlib.Path(r"C:\dev\Talkie-Putty\.venv\Lib\site-packages\nvidia")
dirs = [str(d) for d in base.rglob("*") if d.is_dir()
        and any(f.suffix == ".dll" for f in d.iterdir() if f.is_file())]
os.environ["PATH"] = os.pathsep.join(dirs) + os.pathsep + os.environ["PATH"]

from faster_whisper import WhisperModel


def vram():
    return int(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    ).decode().strip())


wav = r"C:\dev\Talkie-Putty\sherpa-onnx-streaming-zipformer-en-20M-2023-02-17\test_wavs\0.wav"
for name in ["distil-whisper/distil-large-v3.5-ct2",
             "deepdml/faster-whisper-large-v3-turbo-ct2"]:
    t0 = time.perf_counter()
    m = WhisperModel(name, device="cuda", compute_type="float16")
    load = time.perf_counter() - t0
    times = []
    for run in range(3):
        t0 = time.perf_counter()
        segs, _ = m.transcribe(wav, language="en", beam_size=1)
        text = " ".join(s.text for s in segs)
        times.append(time.perf_counter() - t0)
    print(f"{name}: load {load:.0f}s, vram {vram()} MiB, "
          f"warm {min(times):.2f}s RTF {min(times)/6.625:.3f}")
    print("  text:", text.strip())
    del m
