"""Live STT GUI v2 — two-pass:
  pass 1: streaming zipformer (instant raw words, endpoint detection)
  pass 2: NB-Whisper (Norwegian) on GPU re-decodes the utterance every ~1 s,
          correcting earlier words from context; final pass adds punctuation.
Gray italic = whisper partial (self-correcting). Black = finalized.
Status line shows instant raw zipformer words.
Text box is editable. Buttons: Cut all / Clear / Clear last sentence.
Close window to stop."""
import os
import pathlib
import queue
import threading
import time

VENV_NVIDIA = pathlib.Path(__file__).parent / ".venv" / "Lib" / "site-packages" / "nvidia"
_dll_dirs = [str(d) for d in VENV_NVIDIA.rglob("*")
             if d.is_dir() and any(f.suffix == ".dll" for f in d.iterdir() if f.is_file())]
os.environ["PATH"] = os.pathsep.join(_dll_dirs) + os.pathsep + os.environ["PATH"]

import numpy as np
import tkinter as tk
from tkinter.scrolledtext import ScrolledText

import sounddevice as sd
import sherpa_onnx
from faster_whisper import WhisperModel

MODEL_DIR = pathlib.Path(__file__).parent / "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"
WHISPER_MODEL = "distil-whisper/distil-large-v3.5-ct2"
SAMPLE_RATE = 16000
CHUNK = 1600          # 100 ms
PARTIAL_INTERVAL = 0.9  # s between whisper partial re-decodes
MAX_UTTERANCE = 28.0    # s, force-finalize beyond this (whisper window is 30 s)

msg_queue = queue.Queue()
stop_event = threading.Event()

# whisper job slots: finals must all run, partials only the latest
_job_lock = threading.Lock()
_pending_finals = []
_pending_partial = None
_job_available = threading.Event()
whisper_busy = threading.Event()

language = "en"  # distil-large-v3.5 is English-only


def submit_job(utt_id, audio, final):
    global _pending_partial
    with _job_lock:
        if final:
            _pending_finals.append((utt_id, audio))
        else:
            _pending_partial = (utt_id, audio)
    _job_available.set()


def whisper_worker():
    global _pending_partial
    model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
    # warm up
    model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), language="en", beam_size=1)
    msg_queue.put(("status", "Models ready — listening"))
    while not stop_event.is_set():
        if not _job_available.wait(timeout=0.2):
            continue
        with _job_lock:
            if _pending_finals:
                utt_id, audio = _pending_finals.pop(0)
                final = True
            elif _pending_partial is not None:
                utt_id, audio = _pending_partial
                _pending_partial = None
                final = False
            else:
                _job_available.clear()
                continue
        whisper_busy.set()
        try:
            segs, _ = model.transcribe(
                audio, language=language,
                beam_size=5 if final else 1,
                condition_on_previous_text=False,
            )
            text = " ".join(s.text.strip() for s in segs).strip()
            msg_queue.put(("final" if final else "partial", (utt_id, text)))
        except Exception as e:
            msg_queue.put(("status", f"whisper error: {e}"))
        finally:
            whisper_busy.clear()


def mic_worker():
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(MODEL_DIR / "tokens.txt"),
        encoder=str(MODEL_DIR / "encoder-epoch-99-avg-1.int8.onnx"),
        decoder=str(MODEL_DIR / "decoder-epoch-99-avg-1.int8.onnx"),
        joiner=str(MODEL_DIR / "joiner-epoch-99-avg-1.int8.onnx"),
        num_threads=2,
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        enable_endpoint_detection=True,
        rule2_min_trailing_silence=0.8,
    )
    stream = recognizer.create_stream()
    buffer = []
    utt_id = 0
    last_partial = 0.0
    had_speech = False

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=CHUNK) as mic:
        while not stop_event.is_set():
            chunk, _ = mic.read(CHUNK)
            mono = chunk[:, 0].copy()
            buffer.append(mono)
            msg_queue.put(("level", float(np.abs(mono).max())))

            stream.accept_waveform(SAMPLE_RATE, mono)
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            raw = recognizer.get_result(stream)
            if raw:
                had_speech = True
                msg_queue.put(("raw", raw))

            now = time.monotonic()
            dur = len(buffer) * CHUNK / SAMPLE_RATE
            if (had_speech and not whisper_busy.is_set()
                    and now - last_partial >= PARTIAL_INTERVAL):
                submit_job(utt_id, np.concatenate(buffer), final=False)
                last_partial = now

            if recognizer.is_endpoint(stream) or dur > MAX_UTTERANCE:
                if had_speech:
                    submit_job(utt_id, np.concatenate(buffer), final=True)
                    utt_id += 1
                recognizer.reset(stream)
                buffer = []
                had_speech = False
                msg_queue.put(("raw", ""))


# ---------------- GUI ----------------
root = tk.Tk()
root.title("Live STT v2 — NB-Whisper (GPU) + streaming zipformer")
root.geometry("800x500")

top = tk.Frame(root)
top.pack(fill="x", padx=8, pady=(8, 0))
status = tk.Label(top, text="Loading models (first run downloads ~1 GB)...", anchor="w")
status.pack(side="left")

tk.Label(top, text="distil-large-v3.5 (en)", fg="#999").pack(side="right")

level_bar = tk.Canvas(root, height=10, bg="#ddd", highlightthickness=0)
level_bar.pack(fill="x", padx=8, pady=(4, 0))
level_rect = level_bar.create_rectangle(0, 0, 0, 10, fill="#4caf50", width=0)

raw_line = tk.Label(root, text="", anchor="w", fg="#999",
                    font=("Consolas", 9))
raw_line.pack(fill="x", padx=8)

text_box = ScrolledText(root, font=("Segoe UI", 12), wrap="word", undo=True)
text_box.pack(fill="both", expand=True, padx=8, pady=(4, 0))
text_box.tag_configure("partial", foreground="#888", font=("Segoe UI", 12, "italic"))
text_box.mark_set("partial_start", "end-1c")
text_box.mark_gravity("partial_start", "left")

btns = tk.Frame(root)
btns.pack(fill="x", padx=8, pady=8)

def set_partial(s):
    text_box.delete("partial_start", "end")
    text_box.insert("end", s, "partial")
    text_box.see("end")

def add_final(s):
    text_box.delete("partial_start", "end")
    text_box.insert("end", s + "\n")
    text_box.mark_set("partial_start", "end-1c")
    text_box.see("end")

def cut_all():
    txt = text_box.get("1.0", "end-1c")
    if txt:
        root.clipboard_clear()
        root.clipboard_append(txt)
    clear_all()

def clear_all():
    text_box.delete("1.0", "end")
    text_box.mark_set("partial_start", "end-1c")

def clear_last():
    # partial region untouched; delete the last finalized line before it
    end_idx = text_box.index("partial_start")
    prev = text_box.index(f"{end_idx} -1 lines linestart")
    if text_box.compare(prev, "<", end_idx):
        text_box.delete(prev, end_idx)

tk.Button(btns, text="Cut all to clipboard", command=cut_all).pack(side="left")
tk.Button(btns, text="Clear", command=clear_all).pack(side="left", padx=8)
tk.Button(btns, text="Clear last sentence", command=clear_last).pack(side="left")

displayed_final_utt = -1

def poll():
    global displayed_final_utt
    try:
        while True:
            kind, payload = msg_queue.get_nowait()
            if kind == "status":
                status.config(text=payload)
            elif kind == "level":
                w = level_bar.winfo_width()
                level_bar.coords(level_rect, 0, 0, min(1.0, payload * 3) * w, 10)
            elif kind == "raw":
                raw_line.config(text=payload)
            elif kind == "partial":
                utt_id, text = payload
                if utt_id > displayed_final_utt:  # ignore stale partials
                    set_partial(text)
            elif kind == "final":
                utt_id, text = payload
                displayed_final_utt = utt_id
                if text:
                    add_final(text)
                else:
                    set_partial("")
    except queue.Empty:
        pass
    root.after(50, poll)

def on_close():
    stop_event.set()
    root.after(300, root.destroy)

root.protocol("WM_DELETE_WINDOW", on_close)
threading.Thread(target=whisper_worker, daemon=True).start()
threading.Thread(target=mic_worker, daemon=True).start()
root.after(50, poll)
root.mainloop()
