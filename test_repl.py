import difflib
import json
import pathlib
import re

_repl = json.loads((pathlib.Path(__file__).parent / "replacements.json").read_text(encoding="utf-8"))
UTTERANCE_REPLACEMENTS = _repl.get("utterance", {})
PHRASE_REPLACEMENTS = _repl.get("phrase", {})


def apply_replacements(text):
    norm = re.sub(r"[^\w\s]", "", text).lower()
    norm = re.sub(r"\s+", " ", norm).strip()
    if norm in UTTERANCE_REPLACEMENTS:
        return UTTERANCE_REPLACEMENTS[norm]
    close = difflib.get_close_matches(norm, UTTERANCE_REPLACEMENTS, n=1, cutoff=0.8)
    if close:
        return UTTERANCE_REPLACEMENTS[close[0]]
    out = text
    for k, v in PHRASE_REPLACEMENTS.items():
        out = re.sub(rf"\b{re.escape(k)}\b", v, out, flags=re.IGNORECASE)
    if out.lstrip().startswith("/"):
        out = out.strip().rstrip(".!?,")
    return out


cases = [
    ("Clear chat.", "/clear"),
    ("clear chat", "/clear"),
    ("Exit chat!", "/exit"),
    # fuzzy hits
    ("Clear shut.", "/clear"),
    ("Clear shot.", "/clear"),
    ("Clear chad.", "/clear"),
    ("Clear the chat.", "/clear"),
    ("Exit shat.", "/exit"),
    ("Exit that.", "/exit"),
    # must NOT fire
    ("Slash compact.", "Slash compact."),
    ("CP.", "CP."),
    ("Please clear chat history.", "Please clear chat history."),
    ("Clean the cache.", "Clean the cache."),
    ("Fix the bug in the auth middleware.", "Fix the bug in the auth middleware."),
]
ok = True
for inp, want in cases:
    got = apply_replacements(inp)
    mark = "ok" if got == want else "FAIL"
    if got != want:
        ok = False
    print(f"{mark}: {inp!r} -> {got!r}" + ("" if got == want else f"  (want {want!r})"))
print("ALL OK" if ok else "FAILURES")
