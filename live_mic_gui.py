"""Live STT in a tkinter window. Gray italic line = current partial,
black lines = finalized utterances. Close window to stop."""
import queue
import threading
from pathlib import Path

import tkinter as tk
from tkinter.scrolledtext import ScrolledText

import sounddevice as sd
import sherpa_onnx

MODEL_DIR = Path(__file__).parent / "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"
SAMPLE_RATE = 16000
CHUNK = 1600  # 100 ms

msg_queue = queue.Queue()
stop_event = threading.Event()


def mic_worker():
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(MODEL_DIR / "tokens.txt"),
        encoder=str(MODEL_DIR / "encoder-epoch-99-avg-1.int8.onnx"),
        decoder=str(MODEL_DIR / "decoder-epoch-99-avg-1.int8.onnx"),
        joiner=str(MODEL_DIR / "joiner-epoch-99-avg-1.int8.onnx"),
        num_threads=4,
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        enable_endpoint_detection=True,
        rule2_min_trailing_silence=0.8,
    )
    stream = recognizer.create_stream()
    last_text = ""
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=CHUNK) as mic:
        msg_queue.put(("status", "Listening..."))
        while not stop_event.is_set():
            chunk, _ = mic.read(CHUNK)
            level = float(abs(chunk[:, 0]).max())
            msg_queue.put(("level", level))
            stream.accept_waveform(SAMPLE_RATE, chunk[:, 0])
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)

            text = recognizer.get_result(stream)
            if text != last_text:
                msg_queue.put(("partial", text))
                last_text = text

            if recognizer.is_endpoint(stream):
                if last_text:
                    msg_queue.put(("final", last_text))
                recognizer.reset(stream)
                last_text = ""
                msg_queue.put(("partial", ""))


root = tk.Tk()
root.title("Live STT — sherpa-onnx streaming Zipformer")
root.geometry("700x400")

status = tk.Label(root, text="Loading model...", anchor="w")
status.pack(fill="x", padx=8, pady=(8, 0))

level_bar = tk.Canvas(root, height=10, bg="#ddd", highlightthickness=0)
level_bar.pack(fill="x", padx=8, pady=(4, 0))
level_rect = level_bar.create_rectangle(0, 0, 0, 10, fill="#4caf50", width=0)

text_box = ScrolledText(root, font=("Segoe UI", 12), wrap="word")
text_box.pack(fill="both", expand=True, padx=8, pady=8)
text_box.tag_configure("partial", foreground="#888", font=("Segoe UI", 12, "italic"))
text_box.tag_configure("final", foreground="#111")
text_box.configure(state="disabled")

partial_mark = None  # index where current partial line starts


def set_partial(s):
    text_box.configure(state="normal")
    text_box.delete("partial_start", "end")
    text_box.insert("end", s, "partial")
    text_box.see("end")
    text_box.configure(state="disabled")


def add_final(s):
    text_box.configure(state="normal")
    text_box.delete("partial_start", "end")
    text_box.insert("end", s + "\n", "final")
    text_box.mark_set("partial_start", "end-1c")
    text_box.see("end")
    text_box.configure(state="disabled")


text_box.configure(state="normal")
text_box.mark_set("partial_start", "end-1c")
text_box.mark_gravity("partial_start", "left")
text_box.configure(state="disabled")


def poll():
    try:
        while True:
            kind, payload = msg_queue.get_nowait()
            if kind == "status":
                status.config(text=payload)
            elif kind == "level":
                w = level_bar.winfo_width()
                level_bar.coords(level_rect, 0, 0, min(1.0, payload * 3) * w, 10)
            elif kind == "partial":
                set_partial(payload)
            elif kind == "final":
                add_final(payload)
    except queue.Empty:
        pass
    root.after(50, poll)


def on_close():
    stop_event.set()
    root.after(200, root.destroy)


root.protocol("WM_DELETE_WINDOW", on_close)
threading.Thread(target=mic_worker, daemon=True).start()
root.after(50, poll)
root.mainloop()
