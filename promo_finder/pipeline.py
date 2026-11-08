"""Promo finder pipeline.

  parse (code) -> scan each episode (LLM) -> validate + snap timestamps (code) -> repair once (LLM)
  -> stitch multi-segment promos across episodes (LLM) -> validate (code)
  -> cold-viewer judge, no show context (LLM) -> score, dedupe, rank (code) -> outputs
"""
from __future__ import annotations

import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from . import prompts as P
from .llm import LLM
from .parser import Beat, Episode, beat_index, fmt

MIN_DUR, MAX_DUR, HOOK_WINDOW = 20.0, 60.0, 3.0
TOL = 0.0  # duration is measured AFTER snapping, so it is the real clip length: 20-60s is a hard rule


# ---------------------------------------------------------------- validation (pure code)
def _snap(ep: Episode, start_b: Beat, end_b: Beat) -> tuple[float, float]:
    """Use beat boundaries, never cut a dialogue line in half."""
    start, end = start_b.start, end_b.end
    for b in ep.beats:
        if b.kind == "DIALOGUE":
            if b.start < start < b.end:
                start = b.start
            if b.start < end < b.end:
                end = b.end
    return start, end


def validate(promo: dict, idx: dict[str, Beat], eps: dict[int, Episode]) -> list[str]:
    """Fill promo['segments'] with real timestamps; return a list of rule violations."""
    errs, segs = [], []
    raw = promo.get("segments") or [{"start_beat": promo.get("start_beat"), "end_beat": promo.get("end_beat")}]
    for s in raw:
        sb, eb = idx.get(s.get("start_beat") or ""), idx.get(s.get("end_beat") or "")
        if not sb or not eb:
            errs.append(f"unknown beat id in segment {s}")
            continue
        if sb.episode != eb.episode:
            errs.append(f"segment {sb.id}->{eb.id} spans two episodes; split it into two segments")
            continue
        if eb.end <= sb.start:
            errs.append(f"segment {sb.id}->{eb.id} ends before it starts")
            continue
        st, en = _snap(eps[sb.episode], sb, eb)
        segs.append({"episode": sb.episode, "start_beat": sb.id, "end_beat": eb.id,
                     "start": round(st, 2), "end": round(en, 2),
                     "start_ts": fmt(st), "end_ts": fmt(en)})
    promo["segments"] = segs
    dur = round(sum(s["end"] - s["start"] for s in segs), 2)
    promo["duration"] = dur
    if not segs:
        return errs or ["no valid segments"]
    if dur < MIN_DUR - TOL or dur > MAX_DUR + TOL:
        errs.append(f"duration {dur:.1f}s is outside 20-60s")
    hb = idx.get(promo.get("hook_beat") or "") or idx.get(segs[0]["start_beat"])
    first = segs[0]
    if hb.episode != first["episode"] or not (first["start"] - 0.01 <= hb.start <= first["end"]):
        errs.append(f"hook_beat {hb.id} is not inside the first segment")
    elif hb.start - first["start"] > HOOK_WINDOW:
        errs.append(f"hook lands at {hb.start - first['start']:.1f}s, must be within first 3s")
    for a in range(len(segs)):
        for b in range(a + 1, len(segs)):
            x, y = segs[a], segs[b]
            if x["episode"] == y["episode"] and x["start"] < y["end"] and y["start"] < x["end"]:
                errs.append("two segments overlap")
    return errs


# ---------------------------------------------------------------- LLM stages
def scan_episode(llm: LLM, show: str, ep: Episode, prev: Episode | None, total: int,
                 idx, eps) -> dict:
    tr = ep.transcript()
    out = llm.json(P.SCAN_SYSTEM, P.SCAN_USER.format(
        show=show, num=ep.number, total=total, logline=ep.logline or "n/a",
        previously=(prev.logline if prev else "This is the first episode."), transcript=tr),
        tag=f"scan-e{ep.number:02d}")
    good, bad = [], []
    for p in out.get("promos", []):
        p["source"] = "single-episode"
        (bad if (e := validate(p, idx, eps)) else good).append((p, e))
    repaired, rejected = [], []
    if bad:
        errors = "\n".join(json.dumps({**{k: p.get(k) for k in ("start_beat", "hook_beat", "end_beat", "hook", "cliffhanger")},
                                       "errors": e}) for p, e in bad)
        fix = llm.json(P.REPAIR_SYSTEM, P.REPAIR_USER.format(errors=errors, transcript=tr),
                       tag=f"repair-e{ep.number:02d}")
        for p in fix.get("promos", []):
            p["source"], p["repaired"] = "single-episode", True
            if e := validate(p, idx, eps):
                rejected.append({**p, "errors": e, "stage": "after-repair"})
            else:
                repaired.append(p)
        for p, e in bad:
            rejected.append({**p, "errors": e, "stage": "first-pass"})
    return {"episode": ep.number, "summary": out.get("episode_summary", ""),
            "hook_moments": out.get("hook_moments", []),
            "cliffhanger_moments": out.get("cliffhanger_moments", []),
            "promos": [p for p, _ in good] + repaired, "rejected": rejected}


def _window(ep: Episode, beat: Beat, before=4.0, after=14.0) -> str:
    return "\n".join("    " + b.line() for b in ep.beats
                     if b.kind != "CAMERA" and beat.start - before <= b.start <= beat.start + after)


def cross_episode(llm: LLM, show: str, eps: list[Episode], scans: list[dict], idx, epmap,
                  max_moments: int = 45) -> tuple[list[dict], list[dict]]:
    moments = []
    for s in scans:
        for m in s["hook_moments"]:
            moments.append(("HOOK", m.get("strength", 0), s["episode"], m))
        for m in s["cliffhanger_moments"]:
            moments.append(("CLIFF", m.get("strength", 0), s["episode"], m))
    moments = [m for m in moments if m[3].get("beat_id") in idx]
    moments.sort(key=lambda m: -(m[1] or 0))
    moments = sorted(moments[:max_moments], key=lambda m: (m[2], m[3]["beat_id"]))
    mtxt = "\n".join(f"- {kind} (strength {st}) Ep{e} {m['beat_id']}: {m.get('text', '')}\n"
                     f"{_window(epmap[e], idx[m['beat_id']])}" for kind, st, e, m in moments)
    summ = "\n".join(f"Ep{s['episode']}: {s['summary']}" for s in scans)
    out = llm.json(P.CROSS_SYSTEM, P.CROSS_USER.format(show=show, summaries=summ, moments=mtxt), tag="cross")
    good, rejected = [], []
    for p in out.get("promos", []):
        p["source"] = "multi-segment"
        if e := validate(p, idx, epmap):
            rejected.append({**p, "errors": e, "stage": "cross-episode"})
        else:
            good.append(p)
    return good, rejected


def clip_transcript(p: dict, eps: dict[int, Episode]) -> str:
    lines, offset = [], 0.0
    for s in p["segments"]:
        for b in eps[s["episode"]].beats:
            if s["start"] - 0.01 <= b.start < s["end"]:
                t = offset + b.start - s["start"]
                body = f'{b.speaker}: "{b.text}"' if b.kind == "DIALOGUE" else f"<{b.kind}> {b.text}"
                lines.append(f"  {fmt(t)} {body}")
        offset += s["end"] - s["start"]
        lines.append("  --- cut ---")
    return "\n".join(lines[:-1])


def judge(llm: LLM, promos: list[dict], eps, batch: int = 5) -> None:
    for i in range(0, len(promos), batch):
        chunk = promos[i:i + batch]
        clips = "\n\n".join(f"CLIP {p['id']} ({p['duration']:.0f}s):\n{clip_transcript(p, eps)}" for p in chunk)
        out = llm.json(P.JUDGE_SYSTEM, P.JUDGE_USER.format(clips=clips), tag=f"judge-{i // batch:03d}")
        by_id = {}  # models sometimes echo the label ("CLIP P011") instead of the bare id ("P011")
        for j in out.get("judgements", []):
            m = re.search(r"P\d+", str(j.get("promo_id")))
            by_id[m.group() if m else str(j.get("promo_id"))] = j
        for p in chunk:
            p["judge"] = by_id.get(p["id"], {})


# ---------------------------------------------------------------- scoring (pure code)
def _n(x, lo=1, hi=5):
    try:
        return max(0.0, min(1.0, (float(x) - lo) / (hi - lo)))
    except (TypeError, ValueError):
        return 0.0


def score(p: dict, total_eps: int) -> float:
    j = p.get("judge") or {}
    dur = p["duration"]
    dur_fit = 1.0 if 25 <= dur <= 45 else 0.7
    first_ep = p["segments"][0]["episode"]
    early = 1 - (first_ep - 1) / max(total_eps - 1, 1)   # new viewers land on Ep 1; less spoiler risk
    penalty = min(0.10, 0.03 * len(j.get("confusing") or []))
    s = (0.30 * _n(j.get("hook")) + 0.25 * _n(j.get("cliffhanger")) + 0.20 * _n(j.get("standalone"))
         + 0.10 * _n(j.get("overall"), 1, 10) + 0.08 * dur_fit + 0.07 * early - penalty)
    p["score_parts"] = {"hook": j.get("hook"), "cliffhanger": j.get("cliffhanger"), "standalone": j.get("standalone"),
                        "overall": j.get("overall"), "duration_fit": dur_fit, "early_episode": round(early, 2),
                        "confusion_penalty": penalty}
    return round(100 * s, 1)


def _overlap(a: dict, b: dict) -> float:
    inter = 0.0
    for x in a["segments"]:
        for y in b["segments"]:
            if x["episode"] == y["episode"]:
                inter += max(0.0, min(x["end"], y["end"]) - max(x["start"], y["start"]))
    return inter / max(min(a["duration"], b["duration"]), 1e-6)


def rank(promos: list[dict], total_eps: int) -> tuple[list[dict], list[dict]]:
    for p in promos:
        p["score"] = score(p, total_eps)
    promos.sort(key=lambda p: -p["score"])
    kept, dupes = [], []
    for p in promos:
        twin = next((k for k in kept if _overlap(p, k) > 0.5), None)
        if twin:
            dupes.append({**p, "duplicate_of": twin["id"]})
        else:
            kept.append(p)
    for i, p in enumerate(kept, 1):
        p["rank"] = i
    return kept, dupes


# ---------------------------------------------------------------- orchestration
def run(episodes: list[Episode], show: str, llm: LLM, out_dir: str | Path,
        cross: bool = True, workers: int = 1, log: Callable[[str], None] = print,
        cross_llm: LLM | None = None) -> dict:
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    idx, epmap = beat_index(episodes), {e.number: e for e in episodes}
    total = len(episodes)
    log(f"Parsed {total} episodes, {len(idx)} beats. Model: {llm.provider}/{llm.model}")

    def _scan(i):
        ep = episodes[i]
        try:
            r = scan_episode(llm, show, ep, episodes[i - 1] if i else None, total, idx, epmap)
        except Exception as e:  # one bad episode should not kill a 50-episode run
            log(f"  Ep {ep.number:>2}: FAILED ({e})")
            return {"episode": ep.number, "summary": "", "hook_moments": [], "cliffhanger_moments": [],
                    "promos": [], "rejected": [], "error": str(e)}
        log(f"  Ep {ep.number:>2}: {len(r['promos'])} promos, {len(r['rejected'])} rejected")
        return r

    if workers <= 1:  # sequential keeps UI logging (Streamlit) on the main thread
        scans = [_scan(i) for i in range(total)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            scans = list(ex.map(_scan, range(total)))

    promos = [p for s in scans for p in s["promos"]]
    rejected = [r for s in scans for r in s["rejected"]]
    if cross:
        log("Stitching multi-segment promos across episodes...")
        try:
            cp, cr = cross_episode(cross_llm or llm, show, episodes, scans, idx, epmap)
        except Exception as e:
            log(f"  cross-episode step failed ({e}); continuing with single-episode promos")
            cp, cr = [], []
        promos += cp
        rejected += cr
        log(f"  {len(cp)} multi-segment promos, {len(cr)} rejected")

    for i, p in enumerate(promos, 1):
        p["id"] = f"P{i:03d}"
    log(f"Cold-viewer judging {len(promos)} promos...")
    judge(llm, promos, epmap)
    ranked, dupes = rank(promos, total)

    meta = {"show": show, "episodes": total, "provider": llm.provider, "model": llm.model,
            "proposed": len(promos) + len(rejected), "valid": len(promos), "rejected_by_rules": len(rejected),
            "repaired": sum(1 for p in promos if p.get("repaired")), "duplicates_removed": len(dupes),
            "final": len(ranked), "llm_usage": llm.usage,
            "cross_model": (cross_llm.model if cross_llm else llm.model),
            "cross_llm_usage": (cross_llm.usage if cross_llm else None),  # strong-model call is not in llm_usage
            "est_cost_usd": round(llm.cost_usd() + (cross_llm.cost_usd() if cross_llm else 0), 4),
            "wall_seconds": round(time.time() - t0, 1)}
    _write(out, ranked, rejected, dupes, scans, meta, epmap)
    log(f"Done: {len(ranked)} ranked promos -> {out}")
    return {"ranked": ranked, "rejected": rejected, "duplicates": dupes, "scans": scans, "meta": meta}


def _seg_str(p):
    return " + ".join(f"Ep{s['episode']} {s['start_ts']}-{s['end_ts']}" for s in p["segments"])


def _write(out: Path, ranked, rejected, dupes, scans, meta, epmap):
    for p in ranked:
        p["transcript"] = clip_transcript(p, epmap)
    (out / "promos_ranked.json").write_text(json.dumps(ranked, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "rejected_by_rules.json").write_text(json.dumps(rejected, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "duplicates_removed.json").write_text(json.dumps(dupes, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "episode_scans.json").write_text(json.dumps(scans, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with open(out / "promos_ranked.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "id", "score", "type", "duration_s", "segments", "hook", "cliffhanger", "why",
                    "judge_hook", "judge_cliff", "judge_standalone", "judge_overall", "confusing", "cold_viewer_read"])
        for p in ranked:
            j = p.get("judge") or {}
            w.writerow([p["rank"], p["id"], p["score"], p["source"], p["duration"], _seg_str(p), p.get("hook"),
                        p.get("cliffhanger"), p.get("why"), j.get("hook"), j.get("cliffhanger"), j.get("standalone"),
                        j.get("overall"), "; ".join(j.get("confusing") or []), j.get("what_happens")])
    (out / "report.md").write_text(report(ranked, rejected, dupes, meta), encoding="utf-8")


def report(ranked, rejected, dupes, meta) -> str:
    L = [f"# Promo report: {meta['show']}",
         f"Model `{meta['provider']}/{meta['model']}` | {meta['episodes']} episodes | "
         f"{meta['proposed']} proposed, {meta['rejected_by_rules']} rejected by rules, "
         f"{meta['repaired']} repaired, {meta['duplicates_removed']} duplicates removed, {meta['final']} final | "
         f"{meta['llm_usage']['calls'] + (meta.get('cross_llm_usage') or {}).get('calls', 0)} calls, "
         f"~${meta['est_cost_usd']} | {meta['wall_seconds']}s", "",
         "## Post these first (top 10)", ""]
    for p in ranked[:10]:
        j = p.get("judge") or {}
        L += [f"**#{p['rank']} {p['id']}** ({p['score']}) {p['source']}, {p['duration']:.0f}s: {_seg_str(p)}",
              f"- Hook: {p.get('hook')}", f"- Cliffhanger: {p.get('cliffhanger')}", f"- Why: {p.get('why')}",
              f"- Cold viewer: hook {j.get('hook')}/5, cliff {j.get('cliffhanger')}/5, "
              f"standalone {j.get('standalone')}/5. Read it as: {j.get('what_happens')}", ""]
    weak = [p for p in ranked if (p.get("judge") or {}).get("standalone", 5) <= 2 or (p.get("judge") or {}).get("hook", 5) <= 2]
    L += ["## Where it picks badly", "",
          f"- {len(rejected)} proposals broke hard rules (length / hook timing / bad beat IDs) and were dropped. See rejected_by_rules.json.",
          f"- {len(weak)} valid promos scored <=2 on hook or standalone by the cold viewer:"]
    for p in weak[:8]:
        j = p["judge"]
        L.append(f"  - {p['id']} ({_seg_str(p)}): confusing = {', '.join(j.get('confusing') or []) or 'n/a'}")
    L += ["", "## Full ranking", "", "| # | id | score | type | dur | segments | hook |", "|---|---|---|---|---|---|---|"]
    L += [f"| {p['rank']} | {p['id']} | {p['score']} | {p['source']} | {p['duration']:.0f}s | {_seg_str(p)} | "
          f"{(p.get('hook') or '').replace('|', '/')} |" for p in ranked]
    return "\n".join(L)
