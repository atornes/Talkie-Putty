# PyInstaller spec for Talkie-Putty.
# Build:  pyinstaller Talkie-Putty.spec   (produces dist/Talkie-Putty/)
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []
for pkg in ("sherpa_onnx", "ctranslate2", "faster_whisper", "sounddevice", "keyboard"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# editable default configs (seeded into %LOCALAPPDATA% on first run) + window icon
datas += [("prompt.txt", "."), ("replacements.json", "."),
          ("assets/icon.ico", "assets")]
hiddenimports += ["setup_models", "updater", "version"]

a = Analysis(
    ["live_mic_gui3.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Talkie-Putty",
    debug=False,
    strip=False,
    upx=False,
    console=False,          # windowed GUI app
    icon="assets/icon.ico",
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="Talkie-Putty",
)
