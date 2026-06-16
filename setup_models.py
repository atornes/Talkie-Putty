"""Download and extract the sherpa-onnx models Talkie-Putty needs.

Run once after cloning:  python setup_models.py

GPU Whisper backends are not handled here — faster-whisper auto-downloads those
from Hugging Face on first use. This script only fetches the local sherpa-onnx
model directories (streaming pass-1, CPU parakeet pass-2, and Piper TTS).

Pure standard library — no extra pip installs required.
"""
import pathlib
import sys
import tarfile
import tempfile
import urllib.request

APP_DIR = pathlib.Path(__file__).parent
BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download"

# (extracted dir name, archive url, required-for-core-app?)
MODELS = [
    ("sherpa-onnx-streaming-zipformer-en-20M-2023-02-17",
     f"{BASE}/asr-models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2",
     True),
    ("sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
     f"{BASE}/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2",
     True),
    ("vits-piper-en_US-libritts_r-medium",
     f"{BASE}/tts-models/vits-piper-en_US-libritts_r-medium.tar.bz2",
     False),
]


def _progress(done, total):
    if total <= 0:
        sys.stdout.write(f"\r    {done / 1e6:6.1f} MB")
    else:
        pct = done / total * 100
        bar = "#" * int(pct // 4)
        sys.stdout.write(f"\r    [{bar:<25}] {pct:5.1f}%  {done / 1e6:6.1f} MB")
    sys.stdout.flush()


def fetch_and_extract(url, dest_root):
    with tempfile.NamedTemporaryFile(suffix=".tar.bz2", delete=False) as tmp:
        tmp_path = pathlib.Path(tmp.name)
    try:
        with urllib.request.urlopen(url) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            with open(tmp_path, "wb") as f:
                while True:
                    chunk = resp.read(1 << 20)  # 1 MB
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    _progress(done, total)
        print()
        print("    extracting...")
        with tarfile.open(tmp_path, "r:bz2") as tar:
            tar.extractall(dest_root)
    finally:
        tmp_path.unlink(missing_ok=True)


def main():
    print(f"Models target: {APP_DIR}\n")
    failures = []
    for name, url, required in MODELS:
        tag = "required" if required else "optional (TTS)"
        dest = APP_DIR / name
        if dest.is_dir() and any(dest.iterdir()):
            print(f"[skip] {name}  (already present)")
            continue
        print(f"[get ] {name}  ({tag})")
        try:
            fetch_and_extract(url, APP_DIR)
            print(f"[ ok ] {name}\n")
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"[FAIL] {name}: {e}\n")
            if required:
                failures.append(name)
    if failures:
        print(f"Failed required models: {', '.join(failures)}")
        sys.exit(1)
    print("Done. Run:  python live_mic_gui3.py")


if __name__ == "__main__":
    main()
