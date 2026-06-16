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

## Install (Windows)

Grab the latest **`Talkie-Putty-Setup-*.exe`** from the
[Releases page](https://github.com/atornes/Talkie-Putty/releases) and run it. It adds a
Start Menu shortcut. On first launch the app downloads the speech models (~1.3 GB) into
`%LOCALAPPDATA%\Talkie-Putty`. Selecting a GPU Whisper backend downloads the CUDA runtime
once, on demand.

To run from source instead, see [Requirements](#requirements) below.

### Automatic updates

Installed builds check [GitHub Releases](https://github.com/atornes/Talkie-Putty/releases)
shortly after launch and once a day. When a newer version exists, the app asks whether to
update; if you accept, it downloads the installer and runs it with `/SILENT` — no wizard
buttons to click, but the progress window and any error dialogs stay visible — then
relaunches itself. (Source checkouts don't auto-update.)

## Requirements

- Windows 10/11, Python 3.10+
- NVIDIA GPU + CUDA for the Whisper backends (the CPU parakeet backend needs no GPU)
- Microphone

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
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
- Speak — words appear live in the always-on-top window.
- Window size/position and chosen model persist in `gui_settings.json` (local, gitignored).

## Hotkeys

The editing-cluster keys are captured globally while the app runs (the **dedicated**
keys only — their numpad twins with NumLock off are left alone).

| Key | Where | Action |
|-----|-------|--------|
| **Insert** | any other window focused | Cut all dictated text and paste it (Ctrl+V) into that window. Suppressed system-wide so it never types a literal Insert. |
| **Insert** | Talkie-Putty focused | Cut all text, refocus the last terminal/window you used, and paste there. |
| **Delete** | a terminal focused | Clear the dictation buffer. (Passes through to the OS everywhere else.) |
| **Home** (double-tap) | anywhere | Speak the clipboard aloud (Piper TTS). Double-tap again while it's talking to stop. |

## Configuration

- **[`prompt.txt`](prompt.txt)** — vocabulary-bias terms, one per line. Fed to Whisper
  as an initial prompt (~224-token budget) to favour dev jargon.
- **[`replacements.json`](replacements.json)** — spoken-command replacements.
  `utterance` matches the whole sentence (normalized); `phrase` matches anywhere.

When installed, these (and `gui_settings.json`) live in `%LOCALAPPDATA%\Talkie-Putty`.

## Building the installer

Pushing a tag like `v1.0.0` triggers
[`.github/workflows/build-installer.yml`](.github/workflows/build-installer.yml): it builds
the app with PyInstaller ([`Talkie-Putty.spec`](Talkie-Putty.spec)), packages it with
Inno Setup ([`installer.iss`](installer.iss)), and publishes the installer as a GitHub
release asset.

```powershell
git tag v1.0.0
git push origin v1.0.0
```

Local build:

```powershell
pip install pyinstaller
pyinstaller Talkie-Putty.spec
iscc /DAppVersion=1.0.0 installer.iss   # needs Inno Setup 6
```

The app icon is generated by [`assets/make_icon.py`](assets/make_icon.py) (regenerate only
when the design changes; `assets/icon.ico` is committed).

## License

[MIT](LICENSE) © 2026 Anders Tornes (atornes). Free to use, modify, and redistribute —
keep the copyright notice for attribution.
