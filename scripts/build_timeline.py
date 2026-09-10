#!/usr/bin/env python
"""
build_timeline.py — chronological record of what was run, what wasn't, and why.

WHY a fourth document. The repo already has:
  experiments/README.md    index BY QUESTION  — "what did we learn about X?"
  experiments/RUN_LEDGER   every run's METRICS — "what did run Y score?"
  experiments/BACKLOG.md   the QUEUE          — "what is next?"
  docs/sessions/           per-day NARRATIVE  — "what happened that day?"

None of them answers "what was tried, in what order, and what came of it?" —
which is the question you have months later when writing a thesis and cannot
remember whether an idea was tested, abandoned, or never reached. That is what
this file is for, and it deliberately records **things that were NOT run**,
because a reader has no other way to distinguish "we tried it and it failed"
from "we never got to it".

GENERATED, not hand-written. Hand-maintained timelines drift the moment someone
forgets to update them, and a drifted timeline is worse than none — it looks
authoritative. Everything here is derived from git history, the run ledger, the
experiment write-ups and the backlog, so it cannot disagree with them.

Usage:
    python scripts/build_timeline.py        # rewrites experiments/TIMELINE.md
"""
import glob
import json
import os
import re
import subprocess
from collections import defaultdict
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sh(cmd):
    return subprocess.run(cmd, cwd=ROOT, shell=True, capture_output=True,
                          text=True).stdout.strip()


def commits():
    """Every commit, newest first, with author date."""
    out = []
    raw = sh("git log --pretty=format:'%ai|%h|%s'")
    for line in raw.splitlines():
        parts = line.strip().strip("'").split("|", 2)
        if len(parts) == 3:
            out.append({"when": parts[0][:16], "sha": parts[1], "subject": parts[2]})
    return out


def experiments():
    """Parse each experiment write-up for status, question and headline."""
    rows = []
    for d in sorted(glob.glob(os.path.join(ROOT, "experiments/2026-*"))):
        p = os.path.join(d, "README.md")
        if not os.path.isfile(p):
            continue
        txt = open(p, errors="ignore").read()
        title = txt.splitlines()[0].lstrip("# ").strip()
        m = re.search(r"\|\s*\*\*Status\*\*\s*\|([^|]*)\|", txt)
        status = m.group(1).strip() if m else "?"
        rows.append({
            "id": os.path.basename(d),
            "date": os.path.basename(d)[:10],
            "title": title,
            "status": status,
        })
    return rows


def index_headlines():
    """Headline results, from the experiments index table."""
    p = os.path.join(ROOT, "experiments/README.md")
    out = {}
    if not os.path.isfile(p):
        return out
    for line in open(p, errors="ignore"):
        m = re.match(r"\|\s*\[`([^`]+)`\]\([^)]*\)\s*\|[^|]*\|[^|]*\|([^|]*)\|", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def not_run():
    """Queued and blocked items — the 'never got to it' half of the story."""
    p = os.path.join(ROOT, "experiments/BACKLOG.md")
    if not os.path.isfile(p):
        return []
    items = []
    for line in open(p, errors="ignore"):
        m = re.match(r"###\s+(.+)", line.strip())
        if m:
            t = m.group(1).strip()
            blocked = "BLOCKED" in t or "Blocked" in t or "⛔" in t
            items.append({"title": t, "blocked": blocked})
    return items


def main():
    cs = commits()
    exps = experiments()
    heads = index_headlines()
    pending = not_run()

    by_day = defaultdict(list)
    for c in cs:
        by_day[c["when"][:10]].append(c)

    L = [
        "# Timeline — what was run, what wasn't, and in what order",
        "",
        f"*Generated {datetime.now():%Y-%m-%d %H:%M} by `scripts/build_timeline.py`. "
        "Do not edit by hand — rerun the script.*",
        "",
        "This answers the question the other documents do not: **what was tried, in what "
        "order, and what came of it?** Months later, when writing up, the hard question is "
        "usually not \"what did X score\" but \"did we ever actually test X, or did we just "
        "plan to?\" — so §3 records what was **never run**, and why, as deliberately as §2 "
        "records what was.",
        "",
        "| See also | For |",
        "|---|---|",
        "| [`README.md`](README.md) | results indexed **by question** |",
        "| [`RUN_LEDGER.md`](RUN_LEDGER.md) | every run's **metrics + per-epoch curves** |",
        "| [`BACKLOG.md`](BACKLOG.md) | the **queue** |",
        "| [`../docs/sessions/`](../docs/sessions/) | per-day **narrative** |",
        "",
        "---",
        "",
        "## 1. Experiments, oldest first",
        "",
        "| Date | Experiment | Status | Headline result |",
        "|---|---|---|---|",
    ]
    for e in exps:
        head = heads.get(e["id"], "—")
        L.append(f"| {e['date']} | [`{e['id']}`]({e['id']}/) | {e['status']} | {head} |")

    L += [
        "",
        f"**{len(exps)} experiments written up.** Status legend: ✅ done · ❌ negative result "
        "(kept deliberately) · ⚠️ confounded or unresolved · running.",
        "",
        "---",
        "",
        "## 2. Full commit history, newest first",
        "",
        f"{len(cs)} commits. Each is a unit of work — a run launched, a result recorded, a "
        "bug found, a document corrected.",
        "",
    ]
    for day in sorted(by_day, reverse=True):
        L.append(f"### {day}  ·  {len(by_day[day])} commits")
        L.append("")
        for c in by_day[day]:
            L.append(f"- `{c['when'][11:]}` **{c['sha']}** {c['subject']}")
        L.append("")

    L += [
        "---",
        "",
        "## 3. Queued and NOT run",
        "",
        "The half of the record that is normally lost. An idea absent from this repo was "
        "either never had, or was had and forgotten — and there is no way to tell later.",
        "",
    ]
    if pending:
        for it in pending:
            mark = "⛔ **blocked**" if it["blocked"] else "queued"
            L.append(f"- {mark} — {it['title']}")
    else:
        L.append("*(backlog empty)*")

    L += ["", "---", "",
          "## 4. How to regenerate", "",
          "```bash",
          "python scripts/build_run_ledger.py   # metrics first",
          "python scripts/build_timeline.py     # then the timeline",
          "```",
          ""]

    out = os.path.join(ROOT, "experiments/TIMELINE.md")
    with open(out, "w") as fh:
        fh.write("\n".join(L))
    print(f"[timeline] {len(exps)} experiments, {len(cs)} commits, "
          f"{len(pending)} not-run items")
    print(f"[timeline] wrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
