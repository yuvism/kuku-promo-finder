"""Keyword baseline. Same JSON contract as the LLM, so the whole pipeline runs offline.

Useful for: testing without an API key, and as a 'dumb baseline' to compare the LLM against.
"""
from __future__ import annotations

import re

LINE = re.compile(r"\[(E\d+-B\d+) (\d+):(\d+\.\d+)-(\d+):(\d+\.\d+)\] (.*)")
CLIP_LINE = re.compile(r"^\s*(\d+):(\d+\.\d+) (.*)$")
HOOK_WORDS = {
    "slap": 5, "slaps": 5, "strikes": 4, "punch": 4, "knife": 4, "sword": 4, "gun": 4, "blood": 3,
    "kiss": 4, "kneel": 3, "kneels": 3, "duke": 3, "reveal": 4, "pregnant": 5, "divorce": 4,
    "betray": 4, "liar": 3, "humiliat": 3, "scream": 3, "gasps": 3, "shock": 3, "guards": 2,
    "marry": 3, "proposal": 3, "throws": 3, "grabs": 3, "falls": 2, "dead": 4, "kill": 4,
}
CLIFF_WORDS = {"cliffhanger": 5, "ominous": 3, "freezes": 3, "who": 1, "doors burst": 4, "cut to black": 4, "sting": 3}


def _beats(text):
    out = []
    for m in LINE.finditer(text):
        bid, m1, s1, m2, s2, body = m.groups()
        out.append({"id": bid, "start": int(m1) * 60 + float(s1), "end": int(m2) * 60 + float(s2), "text": body})
    return out


def _hook_score(t: str) -> int:
    low = t.lower()
    s = sum(w for k, w in HOOK_WORDS.items() if k in low)
    s += 1 if "!" in t else 0
    return s


def _cliff_score(t: str) -> int:
    low = t.lower()
    return sum(w for k, w in CLIFF_WORDS.items() if k in low) + (2 if t.rstrip('"').endswith("?") else 0)


def _scan(user: str) -> dict:
    beats = _beats(user)
    if not beats:
        return {"episode_summary": "", "hook_moments": [], "cliffhanger_moments": [], "promos": []}
    hooks = sorted(beats, key=lambda b: -_hook_score(b["text"]))[:4]
    cliffs = sorted(beats, key=lambda b: -_cliff_score(b["text"]))[:2]
    promos = []
    for h in hooks[:2]:
        target = h["start"] + 38
        end = min((b for b in beats if b["end"] >= h["start"] + 22),
                  key=lambda b: abs(b["end"] - target) - 2 * _cliff_score(b["text"]), default=beats[-1])
        promos.append({"start_beat": h["id"], "hook_beat": h["id"], "end_beat": end["id"],
                       "hook": h["text"][:100], "cliffhanger": end["text"][:100],
                       "why": "keyword hook + nearest tense line", "context_needed": "unknown"})
    last = beats[-1]
    st = min((b for b in beats if b["start"] >= last["end"] - 40), key=lambda b: -_hook_score(b["text"]), default=beats[0])
    promos.append({"start_beat": st["id"], "hook_beat": st["id"], "end_beat": last["id"],
                   "hook": st["text"][:100], "cliffhanger": "episode ending",
                   "why": "episode endings are written as cliffhangers", "context_needed": "unknown"})
    return {"episode_summary": "(heuristic baseline: no summary)",
            "hook_moments": [{"beat_id": b["id"], "type": "other", "text": b["text"][:80],
                              "strength": min(5, 1 + _hook_score(b["text"]) // 2)} for b in hooks],
            "cliffhanger_moments": [{"beat_id": b["id"], "text": b["text"][:80],
                                     "strength": min(5, 1 + _cliff_score(b["text"]))} for b in cliffs],
            "promos": promos}


def _judge(user: str) -> dict:
    js = []
    for block in re.split(r"\n\s*\n(?=CLIP )", user):
        m = re.search(r"CLIP (\S+)", block)
        if not m:
            continue
        lines = [CLIP_LINE.match(l) for l in block.splitlines()]
        lines = [(int(a) * 60 + float(b), t) for x in lines if x for a, b, t in [x.groups()]]
        first = " ".join(t for s, t in lines if s <= 3.0)
        last = lines[-1][1] if lines else ""
        js.append({"promo_id": m.group(1), "hook": min(5, 1 + _hook_score(first)),
                   "cliffhanger": min(5, 1 + _cliff_score(last)), "standalone": 3, "confusing": [],
                   "what_happens": "(heuristic)", "overall": 5})
    return {"judgements": js}


def respond(system: str, user: str) -> dict:
    task = re.search(r"TASK: (\w+)", system).group(1)
    if task == "EPISODE_SCAN":
        return _scan(user)
    if task == "JUDGE":
        return _judge(user)
    return {"promos": []}  # REPAIR / CROSS_EPISODE: baseline doesn't attempt these
