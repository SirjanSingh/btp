#!/usr/bin/env python
"""
fetch_drive_folder.py — parallel Google Drive folder downloader.

WHY: `gdown --folder` downloads strictly one file at a time. Each file transfers
fine (~14 MB/s on this node) but every file costs a fresh request + confirm-token
round trip, and that latency dominates for folders of many medium files. AIRS is
~950 files x ~20 MB, where serial gdown projects to several hours against a few
minutes of actual transfer.

This enumerates once, then downloads with a thread pool. Already-complete files
are skipped, so it is safe to re-run after an interruption.

Usage:
    python fetch_drive_folder.py <folder_url_or_id> <dest_dir> [--workers 8]
"""
import argparse
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import gdown
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def enumerate_folder(url, dest):
    """Return [(file_id, local_path)] without downloading anything.

    Uses drive_list, NOT gdown.download_folder. gdown scrapes the folder's
    rendered HTML, which only ever contains the first ~50 entries; the rest load
    by XHR that gdown never issues. That silently truncated AIRS to 50 of 857
    images and made the folder look like a sample of the dataset rather than the
    dataset. `MAX_NUMBER_FILES` does not control this and raising it proves
    nothing -- it only gates a warning.
    """
    from drive_list import folder_id, walk       # same directory

    class _Args:
        retries, pause = 6, 1.0

    rows, seen = [], set()
    walk(folder_id(url), "", requests.Session(), _Args(), rows, seen)
    if not rows:
        sys.exit("could not enumerate folder (private? bad id?)")
    return [(r["id"], os.path.join(dest, r["path"])) for r in rows]


def fetch(file_id, path, retries=5):
    """Download one file unless it already exists with non-zero size.

    Drive rate-limits concurrent access and answers with FileURLRetrievalError
    ("Cannot retrieve the public link"). That is transient, so back off
    exponentially with jitter rather than giving up on the file.
    """
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path, "skip"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    for attempt in range(retries):
        try:
            gdown.download(id=file_id, output=path, quiet=True)
            if os.path.exists(path) and os.path.getsize(path) > 0:
                return path, "ok"
        except Exception as exc:                      # noqa: BLE001
            if attempt == retries - 1:
                return path, f"FAIL {exc!r:.60}"
        time.sleep((2 ** attempt) + random.uniform(0, 1.5))
    return path, "FAIL empty"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("dest")
    # 8 workers trips Drive's rate limiter hard; 3 sustains throughput without
    # burning most requests on retries.
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    print(f"[fetch] enumerating {args.url} ...", flush=True)
    files = enumerate_folder(args.url, args.dest)
    print(f"[fetch] {len(files)} files, {args.workers} workers", flush=True)

    done = skipped = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, fid, p): p for fid, p in files}
        for i, fut in enumerate(as_completed(futures), 1):
            path, status = fut.result()
            if status == "ok":
                done += 1
            elif status == "skip":
                skipped += 1
            else:
                failed += 1
                print(f"  [{status}] {path}", flush=True)
            if i % 50 == 0 or i == len(files):
                print(f"  {i}/{len(files)}  new={done} skip={skipped} fail={failed}",
                      flush=True)

    print(f"[fetch] complete: {done} downloaded, {skipped} already present, "
          f"{failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
