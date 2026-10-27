#!/usr/bin/env python
"""
drive_list.py — recursively list a PUBLIC Google Drive folder, completely.

WHY THIS EXISTS: `gdown.download_folder(skip_download=True)` silently truncates
at ~50 entries per folder. It scrapes the folder's rendered HTML page, and Drive
only puts the first page of items in that HTML — the rest arrive by XHR on
scroll, which gdown never issues. gdown's `MAX_NUMBER_FILES` constant does NOT
control this: it only decides whether a warning is raised. Raising it changes
nothing, which is a very convincing way to "verify" a wrong answer.

This uses `embeddedfolderview` instead — the old public folder-view widget,
which renders the full listing as plain HTML and needs no API key or OAuth.

Retries with exponential backoff, because Drive rate-limits a recursive walk and
a throttled response looks exactly like an empty folder. An empty result is
retried rather than believed.

Usage:
    python scripts/drive_list.py <folder_url_or_id> --out listing.json
"""
import argparse
import json
import re
import sys
import time

import requests

VIEW = "https://drive.google.com/embeddedfolderview?id={fid}#list"
ENTRY_RE = re.compile(
    r'<div class="flip-entry" id="entry-([A-Za-z0-9_-]+)".*?'
    r'(?P<kind>aria-label="Folder"|aria-label="[^"]*")?.*?'
    r'<div class="flip-entry-title">([^<]*)</div>',
    re.S,
)


def folder_id(s):
    m = re.search(r"/folders/([A-Za-z0-9_-]+)", s)
    return m.group(1) if m else s


def fetch(fid, session, retries, pause):
    """GET one folder's listing HTML, retrying empties and errors alike."""
    delay = pause
    last = ""
    for attempt in range(retries):
        try:
            r = session.get(VIEW.format(fid=fid), timeout=90)
            if r.status_code == 200 and "flip-entry" in r.text:
                return r.text
            last = f"status {r.status_code}, {len(r.text)} bytes"
            # A folder that is genuinely empty also has no flip-entry, so accept
            # that only after we have retried enough to rule out throttling.
            if r.status_code == 200 and attempt >= 1:
                return r.text
        except requests.RequestException as e:
            last = repr(e)
        time.sleep(delay)
        delay = min(delay * 2, 60)
    print(f"    ! giving up on {fid}: {last}", file=sys.stderr)
    return ""


def parse(html):
    """-> [(id, name, is_folder)] for one folder's direct children."""
    out = []
    for chunk in html.split('<div class="flip-entry"')[1:]:
        m_id = re.match(r' id="entry-([A-Za-z0-9_-]+)"', chunk)
        m_nm = re.search(r'<div class="flip-entry-title">([^<]*)</div>', chunk)
        if not (m_id and m_nm):
            continue
        out.append((m_id.group(1), m_nm.group(1),
                    'aria-label="Folder"' in chunk.split("flip-entry-title")[0]))
    return out


def walk(fid, path, session, a, rows, seen):
    if fid in seen:                      # shortcuts can make the tree a graph
        return
    seen.add(fid)
    entries = parse(fetch(fid, session, a.retries, a.pause))
    files = [e for e in entries if not e[2]]
    subs = [e for e in entries if e[2]]
    print(f"  {path or '/':60s} {len(files):5d} files  {len(subs):3d} subfolders",
          flush=True)
    for cid, name, _ in files:
        rows.append({"id": cid, "path": f"{path}/{name}".lstrip("/")})
    for cid, name, _ in subs:
        time.sleep(a.pause)
        walk(cid, f"{path}/{name}".lstrip("/"), session, a, rows, seen)


def main(a):
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0"
    rows, seen = [], set()
    root = folder_id(a.folder)
    print(f"[drive] walking {root}")
    walk(root, "", session, a, rows, seen)
    print(f"\n[drive] TOTAL FILES: {len(rows):,}  in {len(seen)} folders")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(rows, fh, indent=1)
        print(f"[drive] wrote {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--out", default="")
    p.add_argument("--retries", type=int, default=6)
    p.add_argument("--pause", type=float, default=1.0,
                   help="seconds between requests, doubled on each retry")
    main(p.parse_args())
