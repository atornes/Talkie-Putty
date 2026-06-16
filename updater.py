"""Check GitHub Releases for a newer Talkie-Putty and download the installer.

Pure standard library. Network calls are blocking — the GUI runs them on a
worker thread and marshals results back to the main thread.
"""
import json
import pathlib
import re
import tempfile
import urllib.request

REPO = "atornes/Talkie-Putty"
API_LATEST = f"https://api.github.com/repos/{REPO}/releases/latest"
USER_AGENT = "Talkie-Putty-updater"


def parse_version(tag):
    """'v1.2.3' / '1.2.3' -> (1, 2, 3). Missing parts default to 0."""
    nums = re.findall(r"\d+", tag or "")
    return tuple(int(n) for n in nums[:3]) if nums else (0,)


def is_newer(latest_version, current_str):
    """latest_version: tuple from parse_version; current_str: e.g. '0.1.0'."""
    cur = parse_version(current_str)
    n = max(len(latest_version), len(cur))
    return latest_version + (0,) * (n - len(latest_version)) > cur + (0,) * (n - len(cur))


def _get_json(url):
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.load(resp)


def get_latest_release():
    """Return {tag, version, url, name, notes} for the latest release that has an
    .exe asset, or None if there isn't one."""
    data = _get_json(API_LATEST)
    tag = data.get("tag_name", "")
    asset = next((a for a in data.get("assets", [])
                  if a.get("name", "").lower().endswith(".exe")), None)
    if not asset:
        return None
    return {
        "tag": tag,
        "version": parse_version(tag),
        "url": asset["browser_download_url"],
        "name": asset["name"],
        "notes": data.get("body", "") or "",
    }


def download_installer(url, name, progress=None):
    """Download the installer to a temp file. progress(name, 'download', done, total)."""
    progress = progress or (lambda *a, **k: None)
    dest = pathlib.Path(tempfile.gettempdir()) / name
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                progress(name, "download", done, total)
    return dest
