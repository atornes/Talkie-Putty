# Talkie-Putty

Live speech-to-text dictation into terminals (Windows). Speak; words land in an
always-on-top window, then press **Insert** to paste them into the focused terminal.

## How it works

Two-pass recognition:

1. **Pass 1 — streaming zipformer** (CPU): instant raw words + endpoint detection.
2. **Pass 2 — refinement** re-decodes each utterance for accuracy, punctuation, and
   casing. Selectable backend (see [Models](#models)).

Pause handling: short pause → space (same line); pause > ~2.5 s → newline. Manually
typed newlines are respected. A spoken-command replacer maps phrases like *"clear chat"*
→ `/clear` (edit [`replacements.json`](replacements.json)).

The main app is **[`live_mic_gui3.py`](live_mic_gui3.py)**. Other `live_mic_*`,
`test_*`, and `bench_*` scripts are earlier iterations and experiments.

## Requirements

- Windows 10/11, Python 3.10+
- NVIDIA GPU + CUDA for the Whisper backends (the CPU parakeet backend needs no GPU)
- Microphone

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy sounddevice keyboard sherpa-onnx faster-whisper
```

GPU Whisper needs CUDA runtime DLLs. The app auto-adds `nvidia/*` packages from the
venv to `PATH`, so install the CUDA wheels into the venv (pulled in transitively by
`faster-whisper`/`ctranslate2`, or install `nvidia-cublas-cu12` / `nvidia-cudnn-cu12`
explicitly to match your CUDA version).

## Models

Large model files are **not** in the repo (gitignored). Run the bootstrap script to
download and extract them automatically:

```powershell
python setup_models.py
```

It fetches the sherpa-onnx archives into the project root (skipping any already present).
For reference, the directories and their sources:

| Directory | Purpose | Source |
|-----------|---------|--------|
| `sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/` | Pass 1 streaming (required) | [sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases) |
| `sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/` | Pass 2 CPU backend (default) | [sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases) |
| `vits-piper-en_US-libritts_r-medium/` | Piper TTS (test scripts) | [sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases) |

The GPU Whisper backends (`distil-large-v3.5`, `large-v3-turbo`, `medium.en`)
auto-download from Hugging Face on first use — no manual setup.

## Run

```powershell
.\.venv\Scripts\Activate.ps1
python live_mic_gui3.py
```

- Pick a model from the dropdown (default: `parakeet-tdt-0.6b-v2 (en, CPU)`).
- Speak. Press **Insert** while a terminal is focused to cut all text and paste it
  (Ctrl+V). Insert is suppressed system-wide while the app runs.
- Window size/position and chosen model persist in `gui_settings.json` (local, gitignored).

## Configuration

- **[`prompt.txt`](prompt.txt)** — vocabulary-bias terms, one per line. Fed to Whisper
  as an initial prompt (~224-token budget) to favour dev jargon.
- **[`replacements.json`](replacements.json)** — spoken-command replacements.
  `utterance` matches the whole sentence (normalized); `phrase` matches anywhere.
