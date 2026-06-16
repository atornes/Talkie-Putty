"""Download and extract the models / runtime Talkie-Putty needs.

CLI:   python setup_models.py            # fetch sherpa-onnx models into the repo
Import: ensure_models(dest_root, ...)    # used by the app on first run
        ensure_cuda(dest_root, ...)       # fetch CUDA DLLs for GPU Whisper backends

GPU Whisper *weights* are pulled from Hugging Face by faster-whisper on first use;
this module only provides the local sherpa-onnx model dirs and (optionally) the
CUDA runtime DLLs that ctranslate2 needs on Windows.

Pure standard library — no extra pip installs required.
"""
import io
import logging
import pathlib
import sys
import tarfile
import tempfile
import urllib.request
import zipfile

log = logging.getLogger("talkie.setup")
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

# CUDA 12 runtime wheels (win_amd64) — what ctranslate2 4.x / faster-whisper need.
CUDA_PACKAGES = ["nvidia-cublas-cu12", "nvidia-cudnn-cu12"]


def _noop(*_a, **_k):
    pass


def _download(url, sink, progress, name):
    """Stream url into file-like sink, reporting (name, 'download', done, total)."""
    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = resp.read(1 << 20)  # 1 MB
            if not chunk:
                break
            sink.write(chunk)
            done += len(chunk)
            progress(name, "download", done, total)


def fetch_and_extract(url, dest_root, name, progress):
    with tempfile.NamedTemporaryFile(suffix=".tar.bz2", delete=False) as tmp:
        tmp_path = pathlib.Path(tmp.name)
    try:
        with open(tmp_path, "wb") as f:
            _download(url, f, progress, name)
        progress(name, "extract", 0, 0)
        with tarfile.open(tmp_path, "r:bz2") as tar:
            tar.extractall(dest_root)
    finally:
        tmp_path.unlink(missing_ok=True)


def ensure_models(dest_root, include_optional=True, progress=None):
    """Ensure sherpa-onnx model dirs exist under dest_root.

    progress(name, stage, done, total): stage in download/extract/skip/ok/fail.
    Returns a list of names of *required* models that failed.
    """
    progress = progress or _noop
    dest_root = pathlib.Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    failures = []
    for name, url, required in MODELS:
        if not required and not include_optional:
            continue
        dest = dest_root / name
        if dest.is_dir() and any(dest.iterdir()):
            progress(name, "skip", 0, 0)
            continue
        try:
            log.info("downloading model %s from %s", name, url)
            fetch_and_extract(url, dest_root, name, progress)
            log.info("model ready: %s", name)
            progress(name, "ok", 0, 0)
        except Exception as e:  # noqa: BLE001
            log.exception("model download/extract failed: %s (%s)", name, url)
            progress(name, "fail", 0, 0)
            if required:
                failures.append(f"{name}: {e}")
    return failures


def _pypi_win_wheel_url(package):
    url = f"https://pypi.org/pypi/{package}/json"
    with urllib.request.urlopen(url) as resp:
        import json
        data = json.load(resp)
    # newest release with a win_amd64 wheel
    for version in sorted(data["releases"], reverse=True):
        for f in data["releases"][version]:
            if f["filename"].endswith("win_amd64.whl"):
                return f["url"]
    raise RuntimeError(f"no win_amd64 wheel found for {package}")


def cuda_present(dest_root):
    cuda_dir = pathlib.Path(dest_root) / "cuda"
    return cuda_dir.is_dir() and any(cuda_dir.glob("*.dll"))


def ensure_cuda(dest_root, progress=None):
    """Download CUDA 12 runtime DLLs into dest_root/cuda (best effort).

    Returns the cuda dir path on success, raises on failure.
    """
    progress = progress or _noop
    cuda_dir = pathlib.Path(dest_root) / "cuda"
    if cuda_present(cuda_dir.parent):
        return cuda_dir
    cuda_dir.mkdir(parents=True, exist_ok=True)
    for pkg in CUDA_PACKAGES:
        wheel_url = _pypi_win_wheel_url(pkg)
        log.info("downloading CUDA wheel %s from %s", pkg, wheel_url)
        buf = io.BytesIO()
        _download(wheel_url, buf, progress, pkg)
        progress(pkg, "extract", 0, 0)
        buf.seek(0)
        with zipfile.ZipFile(buf) as z:
            for member in z.namelist():
                if member.endswith(".dll"):
                    with z.open(member) as src:
                        (cuda_dir / pathlib.Path(member).name).write_bytes(src.read())
    return cuda_dir


def _cli_progress(name, stage, done, total):
    if stage == "skip":
        print(f"[skip] {name} (already present)")
    elif stage == "download":
        if total:
            pct = done / total * 100
            sys.stdout.write(f"\r[get ] {name}  {pct:5.1f}%  {done/1e6:6.1f} MB")
        else:
            sys.stdout.write(f"\r[get ] {name}  {done/1e6:6.1f} MB")
        sys.stdout.flush()
    elif stage == "extract":
        print(f"\n[ .. ] {name} extracting...")
    elif stage == "ok":
        print(f"[ ok ] {name}")
    elif stage == "fail":
        print(f"[FAIL] {name}")


def main():
    print(f"Models target: {APP_DIR}\n")
    failures = ensure_models(APP_DIR, include_optional=True, progress=_cli_progress)
    if failures:
        print("\nFailed required models:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    print("\nDone. Run:  python live_mic_gui3.py")


if __name__ == "__main__":
    main()
