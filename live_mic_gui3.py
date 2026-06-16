"""Talkie-Putty — live STT dictation into terminals.

Two-pass recognition:
  pass 1: streaming zipformer (instant raw words + endpoint detection)
  pass 2: distil-large-v3.5 on GPU re-decodes the utterance every ~1 s
          (self-correcting partials), final pass adds punctuation/casing.

Pause behaviour: short pause = space (same line), pause > ~2.5 s = newline.
Manually typed newlines are respected — dictation continues after them.

Global hotkey: press Insert while a terminal is focused → cuts all text
from this window to the clipboard and pastes it into the terminal (Ctrl+V).
Insert is suppressed system-wide while this app runs.

Window is always-on-top; size/position persisted in gui_settings.json.
Close window to stop."""
import ctypes
import difflib
import json
import os
import pathlib
import queue
import re
import threading
import time
from ctypes import wintypes

import shutil
import sys

APP_DIR = pathlib.Path(__file__).parent
RES_DIR = pathlib.Path(getattr(sys, "_MEIPASS", APP_DIR))  # bundled read-only resources
FROZEN = getattr(sys, "frozen", False)
# When installed (frozen) models/config/settings live in a writable per-user dir;
# in a dev checkout they sit next to the script.
if FROZEN:
    DATA_DIR = pathlib.Path(os.environ.get("LOCALAPPDATA", str(APP_DIR))) / "Talkie-Putty"
else:
    DATA_DIR = APP_DIR
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Seed editable default configs into the data dir on first frozen run.
for _cfg in ("prompt.txt", "replacements.json"):
    _dst, _src = DATA_DIR / _cfg, RES_DIR / _cfg
    if FROZEN and not _dst.exists() and _src.exists():
        shutil.copyfile(_src, _dst)


def add_cuda_to_path():
    """Put CUDA runtime DLLs on the search path: the venv's nvidia packages in a
    dev checkout, or the downloaded DATA_DIR/cuda folder when frozen."""
    root_dir = (DATA_DIR / "cuda") if FROZEN else (APP_DIR / ".venv" / "Lib"
                / "site-packages" / "nvidia")
    if not root_dir.exists():
        return False
    dll_dirs = [str(d) for d in [root_dir, *root_dir.rglob("*")]
                if d.is_dir() and any(f.suffix == ".dll" for f in d.iterdir() if f.is_file())]
    for d in dll_dirs:
        try:
            os.add_dll_directory(d)
        except (OSError, AttributeError):
            pass
    if dll_dirs:
        os.environ["PATH"] = os.pathsep.join(dll_dirs) + os.pathsep + os.environ["PATH"]
    return bool(dll_dirs)


add_cuda_to_path()

import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

import keyboard
import sounddevice as sd
import sherpa_onnx
from faster_whisper import WhisperModel

import setup_models

MODEL_DIR = DATA_DIR / "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"
PARAKEET_DIR = DATA_DIR / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
PIPER_DIR = DATA_DIR / "vits-piper-en_US-libritts_r-medium"
PARAKEET_V3_DIR = DATA_DIR / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
DEFAULT_MODEL = "parakeet-tdt-0.6b-v2 (en, CPU)"
# vocabulary-bias terms: prompt.txt holds one term per line; the initial_prompt
# fed to whisper is built from them (~224 token budget)
VOCAB_FILE = DATA_DIR / "prompt.txt"


def load_vocab():
    try:
        raw = VOCAB_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return []
    if not raw:
        return []
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if len(lines) <= 2 and "," in raw:  # migrate old single-paragraph format
        body = raw.split("Vocabulary:", 1)[-1]
        return [t.strip().rstrip(".") for t in body.split(",") if t.strip()]
    return lines


def build_prompt(words):
    if not words:
        return None
    return "Developer dictation log. Vocabulary: " + ", ".join(words) + "."


VOCAB_WORDS = load_vocab()
INITIAL_PROMPT = build_prompt(VOCAB_WORDS)

# spoken-command replacer (edit replacements.json to extend)
try:
    _repl = json.loads((DATA_DIR / "replacements.json").read_text(encoding="utf-8"))
    UTTERANCE_REPLACEMENTS = _repl.get("utterance", {})
    PHRASE_REPLACEMENTS = _repl.get("phrase", {})
except (OSError, json.JSONDecodeError):
    UTTERANCE_REPLACEMENTS, PHRASE_REPLACEMENTS = {}, {}


def apply_replacements(text):
    norm = re.sub(r"[^\w\s]", "", text).lower()
    norm = re.sub(r"\s+", " ", norm).strip()
    if norm in UTTERANCE_REPLACEMENTS:
        return UTTERANCE_REPLACEMENTS[norm]
    # fuzzy: "clear shut" / "clear shot" etc. still hit "clear chat"
    close = difflib.get_close_matches(norm, UTTERANCE_REPLACEMENTS, n=1, cutoff=0.8)
    if close:
        return UTTERANCE_REPLACEMENTS[close[0]]
    out = text
    for k, v in PHRASE_REPLACEMENTS.items():
        out = re.sub(rf"\b{re.escape(k)}\b", v, out, flags=re.IGNORECASE)
    if out.lstrip().startswith("/"):
        out = out.strip().rstrip(".!?,")  # commands should not end with punctuation
    return out
SETTINGS_FILE = DATA_DIR / "gui_settings.json"
SAMPLE_RATE = 16000
CHUNK = 1600            # 100 ms
PARTIAL_INTERVAL = 0.9  # s between whisper partial re-decodes
MAX_UTTERANCE = 28.0    # s, force-finalize (whisper window is 30 s)
LONG_PAUSE = 1.7        # s of extra silence after endpoint (~2.5 s total) -> newline

TERMINAL_EXES = {"windowsterminal.exe", "wt.exe", "openconsole.exe",
                 "conhost.exe", "cmd.exe", "powershell.exe", "pwsh.exe",
                 "alacritty.exe", "wezterm-gui.exe"}

msg_queue = queue.Queue()
stop_event = threading.Event()

_job_lock = threading.Lock()
_pending_finals = []
_pending_partial = None
_job_available = threading.Event()
whisper_busy = threading.Event()


def submit_job(utt_id, audio, final):
    global _pending_partial
    with _job_lock:
        if final:
            _pending_finals.append((utt_id, audio))
        else:
            _pending_partial = (utt_id, audio)
    _job_available.set()


def make_whisper(repo, lang):
    model = WhisperModel(repo, device="cuda", compute_type="float16")
    model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), language=lang, beam_size=1)

    def decode(audio, final):
        segs, _ = model.transcribe(
            audio, language=lang,
            beam_size=5 if final else 1,
            condition_on_previous_text=False,
            initial_prompt=INITIAL_PROMPT if lang == "en" else None,
        )
        return " ".join(s.text.strip() for s in segs).strip()
    return decode


def make_parakeet(model_dir=None):
    d = model_dir or PARAKEET_DIR
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(d / "encoder.int8.onnx"),
        decoder=str(d / "decoder.int8.onnx"),
        joiner=str(d / "joiner.int8.onnx"),
        tokens=str(d / "tokens.txt"),
        num_threads=8,
        model_type="nemo_transducer",
    )

    def decode(audio, final):
        st = rec.create_stream()
        st.accept_waveform(SAMPLE_RATE, audio)
        rec.decode_stream(st)
        return st.result.text.strip()

    decode(np.zeros(SAMPLE_RATE, dtype=np.float32), False)  # warm up the session
    return decode


MODELS = {
    "distil-large-v3.5 (en, GPU)":
        lambda: make_whisper("distil-whisper/distil-large-v3.5-ct2", "en"),
    "large-v3-turbo (en, GPU)":
        lambda: make_whisper("deepdml/faster-whisper-large-v3-turbo-ct2", "en"),
    "whisper-medium.en (en, GPU)":
        lambda: make_whisper("medium.en", "en"),
    "parakeet-tdt-0.6b-v2 (en, CPU)":
        make_parakeet,
}

requested_model = DEFAULT_MODEL  # mutated by the UI combobox


def whisper_worker():
    global _pending_partial
    decode = None
    loaded = None
    while not stop_event.is_set():
        if loaded != requested_model:
            want = requested_model
            is_gpu = "GPU" in want  # GPU backends may download CUDA + weights
            msg_queue.put(("model_loading", want))
            if is_gpu:
                msg_queue.put(("dl_open", f"Loading {want}"))
            # GPU backends need CUDA DLLs; when frozen, fetch them on first use.
            if FROZEN and is_gpu and not setup_models.cuda_present(DATA_DIR):
                try:
                    setup_models.ensure_cuda(DATA_DIR, progress=lambda n, s, d, t: msg_queue.put(
                        ("dl_progress", (f"Downloading GPU runtime — {n}",
                                         (d / t * 100) if t else None))))
                    add_cuda_to_path()
                except Exception as e:
                    msg_queue.put(("dl_close", None))
                    msg_queue.put(("status", f"GPU setup failed: {e} — pick a CPU model"))
                    loaded = want
                    decode = None
                    continue
            try:
                if is_gpu:
                    msg_queue.put(("dl_progress", (
                        f"Loading {want}\n"
                        "(first run downloads model weights — this can take a few minutes)",
                        None)))
                decode = MODELS[want]()
                loaded = want
                msg_queue.put(("model_ready", want))
            except Exception as e:
                msg_queue.put(("status", f"load failed: {e}"))
                loaded = want  # don't retry in a tight loop
                decode = None
            finally:
                if is_gpu:
                    msg_queue.put(("dl_close", None))
            continue
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
        if decode is None:
            continue
        whisper_busy.set()
        try:
            text = apply_replacements(decode(audio, final))
            msg_queue.put(("final" if final else "partial", (utt_id, text)))
        except Exception as e:
            msg_queue.put(("status", f"decode error: {e}"))
        finally:
            whisper_busy.clear()


current_utt = 0      # utterance id the mic is currently collecting (or about to)
mic_speaking = False  # True while speech detected in the current utterance


def mic_worker():
    global current_utt, mic_speaking
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
    speech_announced = False
    last_final_end = None  # monotonic time of the endpoint of the last real utterance

    mic_announced = False
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
            if not mic_announced:
                mic_announced = True  # zipformer built + first chunk decoded
                msg_queue.put(("mic_ready", None))
            raw = recognizer.get_result(stream)
            now = time.monotonic()
            if raw:
                had_speech = True
                mic_speaking = True
                msg_queue.put(("raw", raw))
                if not speech_announced:
                    speech_announced = True
                    if last_final_end is not None:
                        gap = now - last_final_end
                        msg_queue.put(("sep", "\n" if gap >= LONG_PAUSE else " "))

            dur = len(buffer) * CHUNK / SAMPLE_RATE
            if (had_speech and not whisper_busy.is_set()
                    and now - last_partial >= PARTIAL_INTERVAL):
                submit_job(utt_id, np.concatenate(buffer), final=False)
                last_partial = now

            if recognizer.is_endpoint(stream) or dur > MAX_UTTERANCE:
                if had_speech:
                    submit_job(utt_id, np.concatenate(buffer), final=True)
                    utt_id += 1
                    last_final_end = now
                recognizer.reset(stream)
                buffer = []
                had_speech = False
                speech_announced = False
                mic_speaking = False
                current_utt = utt_id
                msg_queue.put(("raw", ""))


user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32


def foreground_window():
    hwnd = user32.GetForegroundWindow()
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return hwnd, pid.value


def exe_of_pid(pid):
    h = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
    if not h:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(260)
        size = wintypes.DWORD(260)
        if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return pathlib.Path(buf.value).name.lower()
        return ""
    finally:
        kernel32.CloseHandle(h)


def on_insert_hotkey():
    # runs on keyboard hook thread; defer all tk work to the main thread
    _, pid = foreground_window()
    if pid == os.getpid():
        msg_queue.put(("hotkey_paste_prev", None))
    else:
        msg_queue.put(("hotkey_paste", None))  # paste into whatever is focused


# ---------------- TTS (Piper, lazy-loaded) ----------------
tts_queue = queue.Queue()
tts_speaking = threading.Event()


tts_voice = 0    # speaker id, set from settings / UI
tts_speed = 1.0  # speaking rate, set from settings / UI
tts_stop = threading.Event()
SENTENCE_PAUSE = 0.40  # s of silence between sentences
PARAGRAPH_PAUSE = 0.90  # s of silence between lines/paragraphs


def tts_worker():
    tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(PIPER_DIR / "en_US-libritts_r-medium.onnx"),
                tokens=str(PIPER_DIR / "tokens.txt"),
                data_dir=str(PIPER_DIR / "espeak-ng-data")),
            num_threads=4),
        max_num_sentences=1))
    msg_queue.put(("tts_ready", tts.num_speakers))

    def speak(text):
        # sentence-by-sentence: natural pauses + playback starts immediately
        for para in (p.strip() for p in text.splitlines()):
            if tts_stop.is_set():
                return
            if not para:
                continue
            for sent in re.split(r"(?<=[.!?;:])\s+", para):
                if tts_stop.is_set():
                    return
                if not sent.strip():
                    continue
                audio = tts.generate(sent, sid=tts_voice, speed=tts_speed)
                if tts_stop.is_set():
                    return
                sd.play(audio.samples, audio.sample_rate)
                sd.wait()  # returns early on sd.stop()
                if tts_stop.wait(SENTENCE_PAUSE):
                    return
            if tts_stop.wait(PARAGRAPH_PAUSE - SENTENCE_PAUSE):
                return

    while not stop_event.is_set():
        try:
            text = tts_queue.get(timeout=0.3)
        except queue.Empty:
            continue
        tts_stop.clear()
        tts_speaking.set()
        try:
            speak(text)
        except Exception as e:
            msg_queue.put(("status", f"TTS error: {e}"))
        finally:
            tts_speaking.clear()


_last_home_press = [0.0]


def on_home_key(_event):
    now = time.monotonic()
    if now - _last_home_press[0] < 0.4:  # double tap
        msg_queue.put(("tts_clipboard", None))
        _last_home_press[0] = 0.0
    else:
        _last_home_press[0] = now


def on_delete_hotkey():
    # not suppressed (Delete is too common to swallow system-wide);
    # only acts when a terminal is focused — own window is handled by a tk binding
    _, pid = foreground_window()
    if pid != os.getpid() and exe_of_pid(pid) in TERMINAL_EXES:
        msg_queue.put(("clear", None))


# The editing-cluster keys (Insert/Delete/Home) and their numpad twins
# (numpad 0/. /7 with NumLock off) report the SAME name to the keyboard lib.
# Only the dedicated keys have is_keypad == False, so we filter on that and
# leave numpad input untouched. One blocking hook handles all three.
_NAV_KEYS = ("insert", "delete", "home")
_nav_held = set()


def _global_key_handler(e):
    # blocking hook: return True to pass the key through, False to suppress it.
    name = e.name
    if e.is_keypad or name not in _NAV_KEYS:
        return True
    if e.event_type == keyboard.KEY_UP:
        _nav_held.discard(name)
        return name != "insert"        # keep suppressing the dedicated Insert up
    if name in _nav_held:
        return name != "insert"        # auto-repeat: swallow Insert, ignore others
    _nav_held.add(name)
    if name == "insert":
        on_insert_hotkey()
        return False                   # suppress dedicated Insert system-wide
    if name == "delete":
        on_delete_hotkey()
    elif name == "home":
        on_home_key(e)
    return True                        # Delete/Home pass through to the OS


# ---------------- GUI ----------------
root = tk.Tk()
root.title("Talkie-Putty — dictate to terminal (Insert = paste)")
root.attributes("-topmost", True)


def geometry_on_a_monitor(geo):
    # saved position may be on a monitor that is now disconnected
    m = re.match(r"(\d+)x(\d+)([+-]\d+)([+-]\d+)$", geo or "")
    if not m:
        return False

    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    _, _, x, y = (int(g) for g in m.groups())
    MONITOR_DEFAULTTONULL = 0
    return bool(user32.MonitorFromPoint(POINT(x + 50, y + 50), MONITOR_DEFAULTTONULL))


saved = {}
try:
    saved = json.loads(SETTINGS_FILE.read_text())
except Exception:
    pass
if geometry_on_a_monitor(saved.get("geometry")):
    root.geometry(saved["geometry"])
else:
    root.geometry("800x500")  # fall back to primary monitor, default spot
if saved.get("model") in MODELS:
    requested_model = saved["model"]

top = tk.Frame(root)
top.pack(fill="x", padx=8, pady=(8, 0))
status = tk.Label(top, text="Loading models...", anchor="w")
status.pack(side="left")
model_var = tk.StringVar(value=requested_model)


def on_model_change(_event):
    global requested_model
    requested_model = model_var.get()
    text_box.focus_set()  # give focus back so hotkeys keep working
    save_settings()


model_box = ttk.Combobox(top, textvariable=model_var, values=list(MODELS),
                         state="readonly", width=30)
model_box.bind("<<ComboboxSelected>>", on_model_change)
model_box.pack(side="right")

level_bar = tk.Canvas(root, height=10, bg="#ddd", highlightthickness=0)
level_bar.pack(fill="x", padx=8, pady=(4, 0))
level_rect = level_bar.create_rectangle(0, 0, 0, 10, fill="#4caf50", width=0)

raw_line = tk.Label(root, text="", anchor="w", fg="#999", font=("Consolas", 9))
raw_line.pack(fill="x", padx=8)

btns = tk.Frame(root)
btns.pack(side="bottom", fill="x", padx=8, pady=8)

text_frame = tk.Frame(root)
text_frame.pack(fill="both", expand=True, padx=8, pady=(4, 0))
text_box = tk.Text(text_frame, font=("Segoe UI", 12), wrap="word", undo=True,
                   relief="flat")
text_scroll = ttk.Scrollbar(text_frame, orient="vertical", command=text_box.yview)
text_box.configure(yscrollcommand=text_scroll.set)
text_scroll.pack(side="right", fill="y")
text_box.pack(side="left", fill="both", expand=True)
text_box.tag_configure("partial", foreground="#888", font=("Segoe UI", 12, "italic"))
text_box.mark_set("partial_start", "end-1c")
text_box.mark_gravity("partial_start", "left")

finals_stack = []  # committed utterance strings, for "clear last sentence"


def set_partial(s):
    # mark has left gravity, so it stays put and the partial text lands after it
    text_box.delete("partial_start", "end-1c")
    text_box.insert("partial_start", s, "partial")
    text_box.see("end")


def commit_sep(sep):
    if text_box.compare("partial_start", "==", "1.0"):
        return  # box empty before insertion point
    prev = text_box.get("partial_start -1c", "partial_start")
    if prev == "\n":
        return  # manual or pause newline already there — continue from it
    if sep == " " and prev == " ":
        return
    text_box.mark_gravity("partial_start", "right")
    text_box.insert("partial_start", sep)
    text_box.mark_gravity("partial_start", "left")


def add_final(s):
    text_box.delete("partial_start", "end-1c")
    text_box.mark_gravity("partial_start", "right")
    text_box.insert("partial_start", s)
    text_box.mark_gravity("partial_start", "left")
    finals_stack.append(s)
    text_box.see("end")


def cut_all():
    txt = text_box.get("1.0", "end-1c").rstrip()
    if txt:
        root.clipboard_clear()
        root.clipboard_append(txt)
    clear_all()
    return txt


def clear_all():
    global ignore_before_utt, _pending_partial
    text_box.delete("1.0", "end")
    text_box.mark_set("partial_start", "end-1c")
    finals_stack.clear()
    # drop everything in-flight: the user has consumed/discarded the text,
    # so late whisper results for old (or currently spoken) audio must not land
    ignore_before_utt = current_utt + (1 if mic_speaking else 0)
    with _job_lock:
        _pending_finals.clear()
        _pending_partial = None


def clear_last():
    while finals_stack:
        s = finals_stack.pop()
        idx = text_box.search(s, "partial_start", backwards=True,
                              stopindex="1.0")
        if not idx:
            continue  # user edited it away; try the one before
        end = f"{idx} +{len(s)}c"
        # also remove a separator just before it
        start = idx
        if text_box.compare(idx, ">", "1.0"):
            prev = text_box.get(f"{idx} -1c", idx)
            if prev in (" ", "\n"):
                start = f"{idx} -1c"
        text_box.delete(start, end)
        break


def copy_all():
    txt = text_box.get("1.0", "end-1c").rstrip()
    if txt:
        root.clipboard_clear()
        root.clipboard_append(txt)


def has_selection():
    return bool(text_box.tag_ranges("sel"))


def on_delete_key(_event):
    clear_all()
    return "break"


def on_ctrl_x(_event):
    if has_selection():
        text_box.event_generate("<<Cut>>")  # normal cut of the selection
    else:
        cut_all()
    return "break"


def on_ctrl_c(_event):
    if has_selection():
        text_box.event_generate("<<Copy>>")  # normal copy of the selection
    else:
        copy_all()
    return "break"


text_box.bind("<Delete>", on_delete_key)
text_box.bind("<Control-x>", on_ctrl_x)
text_box.bind("<Control-c>", on_ctrl_c)
# window-wide fallbacks: fire when focus is on a button or elsewhere in the
# window; the text_box bindings above run first and "break" before these
root.bind("<Delete>", on_delete_key)
root.bind("<Control-x>", on_ctrl_x)
root.bind("<Control-c>", on_ctrl_c)
root.bind("<FocusIn>", lambda e: text_box.focus_set() if e.widget is root else None)

def open_vocab_editor():
    c = THEMES[dark_mode]
    dlg = tk.Toplevel(root)
    dlg.title("Technical vocabulary")
    dlg.transient(root)
    # place right next to the main window
    x = root.winfo_x() + root.winfo_width() + 8
    y = root.winfo_y()
    dlg.geometry(f"380x460+{x}+{y}")
    dlg.configure(bg=c["bg"])

    def themed_btn(parent, label, cmd):
        return tk.Button(parent, text=label, command=cmd,
                         bg=c["btn_bg"], fg=c["btn_fg"],
                         activebackground=c["btn_active"], activeforeground=c["fg"],
                         relief="flat" if dark_mode else "raised", bd=1)

    tk.Label(dlg, text="Terms biasing recognition (whisper models only).",
             bg=c["bg"], fg=c["dim"], anchor="w", justify="left",
             wraplength=360).pack(fill="x", padx=8, pady=(8, 0))

    entry_bar = tk.Frame(dlg, bg=c["bg"])
    entry_bar.pack(fill="x", padx=8, pady=(8, 0))
    entry = tk.Entry(entry_bar, bg=c["box_bg"], fg=c["box_fg"],
                     insertbackground=c["caret"], relief="flat")
    entry.pack(side="left", fill="x", expand=True, ipady=3)

    list_frame = tk.Frame(dlg, bg=c["bg"])
    list_frame.pack(fill="both", expand=True, padx=8, pady=8)
    lb = tk.Listbox(list_frame, bg=c["box_bg"], fg=c["box_fg"],
                    selectbackground=c["sel"], selectforeground=c["box_fg"],
                    relief="flat", font=("Segoe UI", 11), activestyle="none")
    sb = ttk.Scrollbar(list_frame, orient="vertical", command=lb.yview)
    lb.configure(yscrollcommand=sb.set)
    sb.pack(side="right", fill="y")
    lb.pack(side="left", fill="both", expand=True)
    for w in VOCAB_WORDS:
        lb.insert("end", w)

    edit_index = [None]  # index currently being edited, None = adding

    def add_or_update(_event=None):
        term = entry.get().strip()
        if not term:
            return
        if edit_index[0] is not None:
            lb.delete(edit_index[0])
            lb.insert(edit_index[0], term)
            edit_index[0] = None
        elif term not in lb.get(0, "end"):
            lb.insert("end", term)
        entry.delete(0, "end")

    def remove_selected():
        for i in reversed(lb.curselection()):
            lb.delete(i)
        edit_index[0] = None
        entry.delete(0, "end")

    def edit_selected(_event=None):
        sel = lb.curselection()
        if sel:
            edit_index[0] = sel[0]
            entry.delete(0, "end")
            entry.insert(0, lb.get(sel[0]))
            entry.focus_set()

    entry.bind("<Return>", add_or_update)
    lb.bind("<Double-Button-1>", edit_selected)
    lb.bind("<Delete>", lambda e: (remove_selected(), "break")[1])
    themed_btn(entry_bar, "Add / Update", add_or_update).pack(side="left", padx=(8, 0))

    bar = tk.Frame(dlg, bg=c["bg"])
    bar.pack(fill="x", padx=8, pady=(0, 8))

    def save():
        global VOCAB_WORDS, INITIAL_PROMPT
        VOCAB_WORDS = list(lb.get(0, "end"))
        VOCAB_FILE.write_text("\n".join(VOCAB_WORDS) + "\n", encoding="utf-8")
        INITIAL_PROMPT = build_prompt(VOCAB_WORDS)
        dlg.destroy()

    themed_btn(bar, "Cancel", dlg.destroy).pack(side="right", padx=(8, 0))
    themed_btn(bar, "Save", save).pack(side="right")
    themed_btn(bar, "Remove selected", remove_selected).pack(side="left")

    dlg.update_idletasks()
    try:
        val = ctypes.c_int(1 if dark_mode else 0)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            user32.GetParent(dlg.winfo_id()), 20, ctypes.byref(val), 4)
    except Exception:
        pass
    entry.focus_set()


btn_cut = tk.Button(btns, text="Cut all to clipboard", command=cut_all)
btn_cut.pack(side="left")
btn_clear = tk.Button(btns, text="Clear", command=clear_all)
btn_clear.pack(side="left", padx=8)
btn_last = tk.Button(btns, text="Clear last sentence", command=clear_last)
btn_last.pack(side="left")
btn_vocab = tk.Button(btns, text="Vocabulary…", command=open_vocab_editor)
btn_vocab.pack(side="left", padx=8)
hint_label = tk.Label(btns, text="Insert = cut & paste anywhere | Del = clear")
hint_label.pack(side="right")

voice_var = tk.StringVar(value=str(saved.get("tts_voice", 0)))
tts_voice = int(voice_var.get())


def on_voice_change(_event):
    global tts_voice
    tts_voice = int(voice_var.get())
    save_settings()
    if not tts_speaking.is_set():
        tts_queue.put(f"This is voice number {tts_voice}.")
    text_box.focus_set()


voice_box = ttk.Combobox(btns, textvariable=voice_var, state="readonly",
                         width=5, values=["0"])
voice_box.bind("<<ComboboxSelected>>", on_voice_change)
voice_box.pack(side="right", padx=(0, 8))
voice_label = tk.Label(btns, text="Voice:")
voice_label.pack(side="right", padx=(0, 2))

tts_speed = float(saved.get("tts_speed", 1.0))


def on_speed_change(_event):
    global tts_speed
    tts_speed = round(speed_scale.get(), 2)
    save_settings()
    if not tts_speaking.is_set():
        tts_queue.put(f"Speed {tts_speed:g}.")
    text_box.focus_set()


speed_scale = tk.Scale(btns, from_=0.6, to=1.6, resolution=0.05,
                       orient="horizontal", length=110, showvalue=True,
                       highlightthickness=0, bd=0)
speed_scale.set(tts_speed)
speed_scale.bind("<ButtonRelease-1>", on_speed_change)
speed_scale.pack(side="right", padx=(0, 12))
speed_label = tk.Label(btns, text="Speed:")
speed_label.pack(side="right", padx=(0, 2))
theme_btn = tk.Button(top, width=3, command=lambda: toggle_theme())
theme_btn.pack(side="right", padx=(0, 6))
theme_buttons = [btn_cut, btn_clear, btn_last, btn_vocab, theme_btn]

displayed_final_utt = -1
ignore_before_utt = 0      # whisper results for utterances below this are stale
last_external_hwnd = None  # most recent foreground window that is not ours
mic_pipeline_ready = False
ready_model_name = None    # set when the decode model is loaded and warm


def update_ready_status():
    if mic_pipeline_ready and ready_model_name:
        status.config(text=f"Ready — {ready_model_name}")
    elif ready_model_name:
        status.config(text="Starting microphone pipeline...")
    # else: a model_loading/status message is already showing

# ---------------- theme ----------------
THEMES = {
    True: dict(bg="#1e1e1e", fg="#dddddd", dim="#888888",
               box_bg="#252526", box_fg="#e8e8e8", caret="#ffffff",
               sel="#264f78", level_bg="#333333",
               btn_bg="#333333", btn_fg="#dddddd", btn_active="#454545"),
    False: dict(bg="SystemButtonFace", fg="#000000", dim="#999999",
                box_bg="#ffffff", box_fg="#111111", caret="#000000",
                sel="#cce5ff", level_bg="#dddddd",
                btn_bg="SystemButtonFace", btn_fg="#000000",
                btn_active="SystemButtonFace"),
}
dark_mode = bool(saved.get("dark", True))
_style = ttk.Style(root)
_style.theme_use("clam")


def set_titlebar_dark(dark):
    try:
        hwnd = user32.GetParent(root.winfo_id())
        val = ctypes.c_int(1 if dark else 0)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            hwnd, 20, ctypes.byref(val), 4)  # DWMWA_USE_IMMERSIVE_DARK_MODE
    except Exception:
        pass


def apply_theme():
    c = THEMES[dark_mode]
    root.configure(bg=c["bg"])
    for w in (top, btns):
        w.configure(bg=c["bg"])
    status.configure(bg=c["bg"], fg=c["fg"])
    raw_line.configure(bg=c["bg"], fg=c["dim"])
    hint_label.configure(bg=c["bg"], fg=c["dim"])
    voice_label.configure(bg=c["bg"], fg=c["dim"])
    speed_label.configure(bg=c["bg"], fg=c["dim"])
    speed_scale.configure(bg=c["bg"], fg=c["fg"], troughcolor=c["level_bg"],
                          activebackground=c["btn_active"])
    level_bar.configure(bg=c["level_bg"])
    text_box.configure(bg=c["box_bg"], fg=c["box_fg"],
                       insertbackground=c["caret"], selectbackground=c["sel"])
    text_box.tag_configure("partial", foreground=c["dim"])
    for b in theme_buttons:
        b.configure(bg=c["btn_bg"], fg=c["btn_fg"],
                    activebackground=c["btn_active"], activeforeground=c["fg"],
                    relief="flat" if dark_mode else "raised", bd=1)
    text_frame.configure(bg=c["bg"])
    _style.configure("Vertical.TScrollbar", background=c["btn_bg"],
                     troughcolor=c["bg"], bordercolor=c["bg"],
                     arrowcolor=c["fg"])
    _style.map("Vertical.TScrollbar",
               background=[("active", c["btn_active"])])
    _style.configure("TCombobox", fieldbackground=c["box_bg"],
                     background=c["btn_bg"], foreground=c["box_fg"],
                     arrowcolor=c["fg"])
    _style.map("TCombobox",
               fieldbackground=[("readonly", c["box_bg"])],
               foreground=[("readonly", c["box_fg"])],
               selectbackground=[("readonly", c["box_bg"])],
               selectforeground=[("readonly", c["box_fg"])])
    root.option_add("*TCombobox*Listbox.background", c["box_bg"])
    root.option_add("*TCombobox*Listbox.foreground", c["box_fg"])
    root.option_add("*TCombobox*Listbox.selectBackground", c["sel"])
    set_titlebar_dark(dark_mode)
    theme_btn.configure(text="☀" if dark_mode else "🌙")


def toggle_theme():
    global dark_mode
    dark_mode = not dark_mode
    apply_theme()
    save_settings()


root.update_idletasks()  # window must exist before the title bar attribute sticks
apply_theme()


def send_paste_if_focus_left():
    # only paste if focus actually moved away from us (SetForegroundWindow can fail)
    _, pid = foreground_window()
    if pid != os.getpid():
        keyboard.send("ctrl+v")


class ProgressDialog:
    """A download/loading window driven from the main thread via msg_queue.

    Determinate bar when a percentage is known (CUDA download), animated
    indeterminate bar otherwise (Whisper weights download / model build)."""

    def __init__(self, master):
        self.master = master
        self.win = None
        self.mode = None

    def open(self, title):
        if self.win is not None:
            self.close()
        self.win = tk.Toplevel(self.master)
        self.win.title(title)
        self.win.geometry("460x130")
        self.win.resizable(False, False)
        self.win.attributes("-topmost", True)
        self.win.protocol("WM_DELETE_WINDOW", lambda: None)  # no closing mid-download
        self.msg = tk.StringVar(value="Preparing...")
        tk.Label(self.win, textvariable=self.msg, wraplength=430,
                 justify="left").pack(padx=16, pady=(18, 10))
        self.bar = ttk.Progressbar(self.win, length=420, maximum=100)
        self.bar.pack(padx=16)
        self.mode = "determinate"

    def progress(self, text, pct):
        if self.win is None:
            return
        if text:
            self.msg.set(text)
        if pct is None:
            if self.mode != "indeterminate":
                self.bar.config(mode="indeterminate")
                self.bar.start(12)
                self.mode = "indeterminate"
        else:
            if self.mode != "determinate":
                self.bar.stop()
                self.bar.config(mode="determinate")
                self.mode = "determinate"
            self.bar["value"] = pct

    def close(self):
        if self.win is not None:
            try:
                self.bar.stop()
            except tk.TclError:
                pass
            self.win.destroy()
            self.win = None


progress_dialog = ProgressDialog(root)


def poll():
    global displayed_final_utt, last_external_hwnd
    global mic_pipeline_ready, ready_model_name
    hwnd, pid = foreground_window()
    if hwnd and pid != os.getpid():
        last_external_hwnd = hwnd
    try:
        while True:
            kind, payload = msg_queue.get_nowait()
            if kind == "status":
                status.config(text=payload)
            elif kind == "mic_ready":
                mic_pipeline_ready = True
                update_ready_status()
            elif kind == "model_loading":
                ready_model_name = None
                status.config(text=f"Loading {payload}...")
            elif kind == "model_ready":
                ready_model_name = payload
                update_ready_status()
            elif kind == "dl_open":
                progress_dialog.open(payload)
            elif kind == "dl_progress":
                progress_dialog.progress(*payload)
            elif kind == "dl_close":
                progress_dialog.close()
            elif kind == "tts_ready":
                voice_box.configure(values=[str(i) for i in range(payload)])
            elif kind == "level":
                w = level_bar.winfo_width()
                level_bar.coords(level_rect, 0, 0, min(1.0, payload * 3) * w, 10)
            elif kind == "raw":
                raw_line.config(text=payload)
            elif kind == "sep":
                commit_sep(payload)
            elif kind == "partial":
                utt_id, text = payload
                if utt_id >= ignore_before_utt and utt_id > displayed_final_utt:
                    set_partial(text)
            elif kind == "final":
                utt_id, text = payload
                displayed_final_utt = utt_id
                if utt_id < ignore_before_utt:
                    pass  # cleared before this result arrived — discard
                elif text:
                    add_final(text)
                else:
                    set_partial("")
            elif kind == "hotkey_paste":
                txt = cut_all()
                if txt:
                    root.after(120, lambda: keyboard.send("ctrl+v"))
            elif kind == "hotkey_paste_prev":
                txt = cut_all()
                if txt and last_external_hwnd:
                    user32.SetForegroundWindow(last_external_hwnd)
                    root.after(200, send_paste_if_focus_left)
            elif kind == "clear":
                clear_all()
            elif kind == "tts_clipboard":
                if tts_speaking.is_set():
                    tts_stop.set()  # double tap while speaking = stop
                    sd.stop()
                else:
                    try:
                        clip = root.clipboard_get()
                    except tk.TclError:
                        clip = ""
                    if clip.strip():
                        tts_queue.put(clip.strip()[:4000])
    except queue.Empty:
        pass
    root.after(50, poll)


_last_geometry = saved.get("geometry", "800x500")
_save_after_id = None


def save_settings():
    global _last_geometry
    if root.state() == "normal":  # don't store minimized/withdrawn coordinates
        _last_geometry = root.geometry()
    try:
        SETTINGS_FILE.write_text(json.dumps(
            {"geometry": _last_geometry, "model": requested_model,
             "dark": dark_mode, "tts_voice": tts_voice,
             "tts_speed": tts_speed}))
    except OSError:
        pass


def schedule_save(_event=None):
    # debounce: Configure fires continuously while dragging/resizing
    global _save_after_id
    if _save_after_id is not None:
        root.after_cancel(_save_after_id)
    _save_after_id = root.after(1000, save_settings)


root.bind("<Configure>", schedule_save)


def on_close():
    save_settings()
    stop_event.set()
    keyboard.unhook_all()
    root.after(300, root.destroy)


root.protocol("WM_DELETE_WINDOW", on_close)
keyboard.hook(_global_key_handler, suppress=True)  # dedicated Insert/Delete/Home only
def first_run_setup():
    """Download required speech models on first run (blocks with a small splash
    before the worker threads start). The streaming + default CPU models must be
    present for the app to function."""
    if MODEL_DIR.exists() and PARAKEET_DIR.exists():
        return
    win = tk.Toplevel(root)
    win.title("Talkie-Putty — first-run setup")
    win.geometry("440x130")
    win.attributes("-topmost", True)
    msg = tk.StringVar(value="Downloading speech models (first run only)...")
    tk.Label(win, textvariable=msg, wraplength=410, justify="left").pack(padx=16, pady=(18, 10))
    bar = ttk.Progressbar(win, length=400, mode="determinate", maximum=100)
    bar.pack(padx=16)
    st = {"text": "", "pct": 0}

    def progress(name, stage, done, total):
        if stage == "download":
            st["text"] = f"Downloading {name}"
            st["pct"] = (done / total * 100) if total else 0
        elif stage == "extract":
            st["text"] = f"Extracting {name}"
        elif stage in ("skip", "ok"):
            st["pct"] = 0

    holder = {}

    def run():
        holder["failures"] = setup_models.ensure_models(
            DATA_DIR, include_optional=False, progress=progress)
        holder["done"] = True

    threading.Thread(target=run, daemon=True).start()
    while not holder.get("done"):
        msg.set(st["text"] or "Preparing...")
        bar["value"] = st["pct"]
        root.update()
        time.sleep(0.05)
    win.destroy()
    if holder.get("failures"):
        from tkinter import messagebox
        messagebox.showerror(
            "Talkie-Putty",
            "Failed to download required models:\n\n" + "\n".join(holder["failures"])
            + "\n\nCheck your internet connection and restart.")
        root.destroy()
        sys.exit(1)


first_run_setup()
threading.Thread(target=whisper_worker, daemon=True).start()
threading.Thread(target=mic_worker, daemon=True).start()
threading.Thread(target=tts_worker, daemon=True).start()
root.after(50, poll)
root.mainloop()
