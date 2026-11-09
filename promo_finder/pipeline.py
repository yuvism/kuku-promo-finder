"""Promo finder pipeline.

  parse (code) -> scan each episode (LLM) -> validate + snap timestamps (code) -> repair once (LLM)
  -> stitch multi-segment promos across episodes (LLM) -> validate (code)
  -> cold-viewer judge, no show context (LLM) -> score, dedupe, rank (code) -> outputs
"""
from __future__ import annotations

import csv
import io
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
SCENE_TAIL, SCENE_HEAD = 8.0, 60.0  # episode N's last 8s + episode N+1's first 60s = one continuing scene
RESOLUTION_PENALTY = 10.0           # score points lost when the judge says the clip shows the payoff


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


def _auto_fix_hook(p: dict, errs: list[str], idx: dict[str, Beat], eps) -> list[str] | None:
    """Code-side fix for the most common failure (the LLM can't do timing maths): if the ONLY problem is a late
    hook, start the clip on the hook beat, keep the end, and re-validate. None = not applicable."""
    hb = p.get("hook_beat")
    if len(errs) != 1 or not errs[0].startswith("hook lands at") or hb not in idx or not p.get("segments"):
        return None
    p["segments"][0]["start_beat"] = hb
    if "start_beat" in p:
        p["start_beat"] = hb
    return validate(p, idx, eps)


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
        e = validate(p, idx, eps)
        if e and (e2 := _auto_fix_hook(p, e, idx, eps)) is not None:
            p["auto_fix_from_errors"] = e  # keep what was wrong for the record
            p["auto_fixed"] = not e2
            e = e2                         # still failing -> LLM repair sees the moved start + remaining error
        (bad if e else good).append((p, e))
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


def judge(llm: LLM, promos: list[dict], eps, batch: int = 5, key: str = "judge") -> None:
    """Cold-viewer judge. key="judge2" stores a second-opinion model's answers next to the main ones."""
    for i in range(0, len(promos), batch):
        chunk = promos[i:i + batch]
        clips = "\n\n".join(f"CLIP {p['id']} ({p['duration']:.0f}s):\n{clip_transcript(p, eps)}" for p in chunk)
        out = llm.json(P.JUDGE_SYSTEM, P.JUDGE_USER.format(clips=clips), tag=f"{key}-{i // batch:03d}")
        by_id = {}  # models sometimes echo the label ("CLIP P011") instead of the bare id ("P011")
        for j in out.get("judgements", []):
            m = re.search(r"P\d+", str(j.get("promo_id")))
            by_id[m.group() if m else str(j.get("promo_id"))] = j
        for p in chunk:
            p[key] = by_id.get(p["id"], {})


# ---------------------------------------------------------------- scoring (pure code)
def _n(x, lo=1, hi=5):
    try:
        return max(0.0, min(1.0, (float(x) - lo) / (hi - lo)))
    except (TypeError, ValueError):
        return 0.0


def _num(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _truthy(v) -> bool:
    return v is True or str(v).strip().lower() in ("true", "yes")


def judges_of(p: dict) -> list[dict]:
    """The judgements a promo got: the main judge, plus the second-opinion judge if one ran."""
    return [j for j in (p.get("judge"), p.get("judge2")) if j]


def _resolution_share(js: list[dict]) -> float:
    return sum(_truthy(j.get("shows_resolution")) for j in js) / len(js) if js else 0.0


def score(p: dict, total_eps: int, judges: list[dict] | None = None) -> float:
    """Weighted score. With two judges every judge field is AVERAGED. Pass `judges` to score with a chosen
    subset (used to compare the judges); that leaves p untouched."""
    js = judges_of(p) if judges is None else judges

    def avg(k):
        vals = [v for v in (_num(j.get(k)) for j in js) if v is not None]
        return sum(vals) / len(vals) if vals else None

    dur = p["duration"]
    dur_fit = 1.0 if 25 <= dur <= 45 else 0.7
    first_ep = p["segments"][0]["episode"]
    early = 1 - (first_ep - 1) / max(total_eps - 1, 1)   # new viewers land on Ep 1; less spoiler risk
    n_confusing = sum(len(j.get("confusing") or []) for j in js) / len(js) if js else 0.0
    penalty = min(0.10, 0.03 * n_confusing)
    s = (0.30 * _n(avg("hook")) + 0.25 * _n(avg("cliffhanger")) + 0.20 * _n(avg("standalone"))
         + 0.10 * _n(avg("overall"), 1, 10) + 0.08 * dur_fit + 0.07 * early - penalty)
    res_penalty = RESOLUTION_PENALTY * _resolution_share(js)   # 10 if every judge says so, 5 if one of two
    if judges is None:
        p["score_parts"] = {"judges": len(js), "hook": avg("hook"), "cliffhanger": avg("cliffhanger"),
                            "standalone": avg("standalone"), "overall": avg("overall"), "duration_fit": dur_fit,
                            "early_episode": round(early, 2), "confusion_penalty": round(penalty, 3),
                            "resolution_penalty": res_penalty}
    return round(100 * s - res_penalty, 1)


def shows_resolution(p: dict) -> bool:
    """True when at least half of the judges say the clip shows the payoff."""
    js = judges_of(p)
    return bool(js) and _resolution_share(js) >= 0.5


def judge_disagreement(p: dict) -> list[str]:
    """Where the two judges disagree a lot: >=2 points on a 1-5 score, >=3 on overall (1-10), or opposite
    verdicts on shows_resolution. Any of these = 'needs human review'."""
    a, b = p.get("judge") or {}, p.get("judge2") or {}
    if not a or not b:
        return []
    out = []
    for k, gap in (("hook", 2), ("cliffhanger", 2), ("standalone", 2), ("overall", 3)):
        x, y = _num(a.get(k)), _num(b.get(k))
        if x is not None and y is not None and abs(x - y) >= gap:
            out.append(f"{k} {x:g} vs {y:g}")
    if _truthy(a.get("shows_resolution")) != _truthy(b.get("shows_resolution")):
        out.append("shows_resolution")
    return out


def _avg_ranks(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: -vals[i])
    ranks, i = [0.0] * len(vals), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        for k in range(i, j + 1):   # ties share the average rank
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    rx, ry = _avg_ranks(x), _avg_ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx, vy = sum((a - mx) ** 2 for a in rx), sum((b - my) ** 2 for b in ry)
    return round(cov / (vx * vy) ** 0.5, 3) if vx and vy else None


def compare_judges(ranked: list[dict], total_eps: int, labels: tuple[str, str]) -> tuple[dict, str]:
    """Score every final promo with each judge alone and compare: rank correlation, top-10 overlap,
    biggest disagreements. Returns (stats for run_meta, judge_comparison.csv text)."""
    both = [p for p in ranked if p.get("judge") and p.get("judge2")]
    s1 = [score(p, total_eps, [p["judge"]]) for p in both]
    s2 = [score(p, total_eps, [p["judge2"]]) for p in both]
    top = lambda s: {p["id"] for _, p in sorted(zip(s, both), key=lambda t: -t[0])[:10]}
    gaps = sorted(zip(both, s1, s2), key=lambda t: -abs(t[1] - t[2]))
    stats = {"judges": list(labels), "promos_compared": len(both), "spearman_rank_correlation": spearman(s1, s2),
             "top10_overlap": len(top(s1) & top(s2)) if both else 0,
             "needs_human_review": sum(1 for p in ranked if p.get("needs_human_review")),
             "biggest_disagreements": [{"id": p["id"], "rank": p["rank"], f"{labels[0]}_score": a,
                                        f"{labels[1]}_score": b, "where": judge_disagreement(p)}
                                       for p, a, b in gaps[:5]]}
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    l1, l2 = labels
    fields = ("hook", "cliffhanger", "standalone", "overall", "shows_resolution")
    w.writerow(["rank", "promo_id", "final_score", f"{l1}_score", f"{l2}_score"]
               + [f"{l}_{k}" for k in fields for l in labels] + ["needs_human_review", "disagreement"])
    for p, a, b in zip(both, s1, s2):
        row = [p["rank"], p["id"], p["score"], a, b]
        for k in fields:
            row += [p["judge"].get(k), p["judge2"].get(k)]
        w.writerow(row + [bool(p.get("needs_human_review")), "; ".join(judge_disagreement(p))])
    return stats, buf.getvalue()


def _overlap(a: dict, b: dict) -> float:
    inter = 0.0
    for x in a["segments"]:
        for y in b["segments"]:
            if x["episode"] == y["episode"]:
                inter += max(0.0, min(x["end"], y["end"]) - max(x["start"], y["start"]))
    return inter / max(min(a["duration"], b["duration"]), 1e-6)


def _continues_scene(a: dict, b: dict, epmap: dict[int, Episode]) -> bool:
    """One promo ends in the last 8s of episode N and the other starts in the first 60s of episode N+1."""
    def tail_head(x, y):
        last, first = x["segments"][-1], y["segments"][0]
        ep = epmap.get(last["episode"])
        return (ep is not None and first["episode"] == last["episode"] + 1
                and last["end"] >= ep.duration - SCENE_TAIL and first["start"] <= SCENE_HEAD)
    return tail_head(a, b) or tail_head(b, a)


def rank(promos: list[dict], total_eps: int, epmap: dict[int, Episode] | None = None) -> tuple[list[dict], list[dict]]:
    for p in promos:
        p["score"] = score(p, total_eps)
    promos.sort(key=lambda p: -p["score"])  # so the higher score is always the one kept
    kept, dupes = [], []
    for p in promos:
        twin, reason = next((k for k in kept if _overlap(p, k) > 0.5), None), "overlaps a higher-scored promo by >50%"
        if not twin and epmap:
            twin, reason = next((k for k in kept if _continues_scene(p, k, epmap)), None), "continues scene"
        if twin:
            dupes.append({**p, "duplicate_of": twin["id"], "reason": reason})
        else:
            kept.append(p)
    for i, p in enumerate(kept, 1):
        p["rank"] = i
    return kept, dupes


# ---------------------------------------------------------------- orchestration
def run(episodes: list[Episode], show: str, llm: LLM, out_dir: str | Path,
        cross: bool = True, workers: int = 1, log: Callable[[str], None] = print,
        cross_llm: LLM | None = None, judge2_llm: LLM | None = None) -> dict:
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
    if judge2_llm:  # second opinion from another model family; final score = average of both judges
        log(f"Second-opinion judge {judge2_llm.provider}/{judge2_llm.model} on {len(promos)} promos...")
        try:
            judge(judge2_llm, promos, epmap, key="judge2")
        except Exception as e:  # e.g. daily token cap: keep what was judged; the cache lets a re-run resume
            log(f"  second judge stopped early ({e}); promos without it keep the single-judge score")
        for p in promos:
            if dis := judge_disagreement(p):
                p["needs_human_review"], p["judge_disagreement"] = True, dis
    ranked, dupes = rank(promos, total, epmap)
    labels = (_family(llm), _family(judge2_llm)) if judge2_llm else None
    comparison, comparison_csv = compare_judges(ranked, total, labels) if judge2_llm else (None, None)

    meta = {"show": show, "episodes": total, "provider": llm.provider, "model": llm.model,
            "proposed": len(promos) + len(rejected), "valid": len(promos), "rejected_by_rules": len(rejected),
            "auto_fixed": sum(1 for p in promos if p.get("auto_fixed")),
            "repaired": sum(1 for p in promos if p.get("repaired")), "duplicates_removed": len(dupes),
            "duplicates_scene_continuation": sum(1 for d in dupes if d.get("reason") == "continues scene"),
            "final": len(ranked), "final_showing_resolution": sum(1 for p in ranked if shows_resolution(p)),
            "llm_usage": llm.usage,
            "cross_model": (cross_llm.model if cross_llm else llm.model),
            "cross_llm_usage": (cross_llm.usage if cross_llm else None),  # strong-model call is not in llm_usage
            "judge2_model": (f"{judge2_llm.provider}/{judge2_llm.model}" if judge2_llm else None),
            "judge2_usage": (judge2_llm.usage if judge2_llm else None),
            "judge_comparison": comparison,
            "est_cost_usd": round(llm.cost_usd() + (cross_llm.cost_usd() if cross_llm else 0)
                                  + (judge2_llm.cost_usd() if judge2_llm else 0), 4),
            "wall_seconds": round(time.time() - t0, 1)}
    _write(out, ranked, rejected, dupes, scans, meta, epmap)
    if comparison_csv:
        (out / "judge_comparison.csv").write_text(comparison_csv, encoding="utf-8", newline="")
    log(f"Done: {len(ranked)} ranked promos -> {out}")
    return {"ranked": ranked, "rejected": rejected, "duplicates": dupes, "scans": scans, "meta": meta}


def _family(llm: LLM) -> str:
    """Short label for a judge in column names: the model family when obvious, else the provider."""
    m = llm.model.lower()
    return next((f for f in ("llama", "gemini", "gpt", "claude", "qwen", "mistral") if f in m), llm.provider)


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
                    "judge_hook", "judge_cliff", "judge_standalone", "judge_overall", "confusing", "cold_viewer_read",
                    "shows_resolution", "resolution_reason", "fixed_by"])
        for p in ranked:
            j = p.get("judge") or {}
            w.writerow([p["rank"], p["id"], p["score"], p["source"], p["duration"], _seg_str(p), p.get("hook"),
                        p.get("cliffhanger"), p.get("why"), j.get("hook"), j.get("cliffhanger"), j.get("standalone"),
                        j.get("overall"), "; ".join(j.get("confusing") or []), j.get("what_happens"),
                        shows_resolution(p), j.get("resolution_reason"), fixed_by(p)])
    (out / "cut_list.csv").write_text(cut_list_csv(ranked), encoding="utf-8", newline="")
    (out / "cut_commands.sh").write_text(cut_commands_sh(ranked), encoding="utf-8", newline="\n")
    (out / "report.md").write_text(report(ranked, rejected, dupes, meta), encoding="utf-8")


# ---------------------------------------------------------------- editor hand-off (pure code)
def cut_list_csv(ranked: list[dict]) -> str:
    """One row per segment: what an editor needs to cut every ranked promo."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["rank", "promo_id", "segment", "episode", "start_ts", "end_ts", "duration"])
    for p in ranked:
        for i, s in enumerate(p["segments"], 1):
            w.writerow([p["rank"], p["id"], i, s["episode"], s["start_ts"], s["end_ts"], round(s["end"] - s["start"], 2)])
    return buf.getvalue()


def cut_commands_sh(ranked: list[dict], top: int = 20) -> str:
    """ffmpeg script that cuts the top promos from per-episode videos named ep01.mp4, ep02.mp4, ...
    Re-encodes each segment (frame-accurate cuts), then joins multi-segment promos with the concat demuxer."""
    L = ["#!/usr/bin/env bash",
         f"# Cuts the top {top} promos with ffmpeg. Input: one video per episode named ep01.mp4, ep02.mp4, ...",
         "# (placeholder names: rename your files or point VIDEO_DIR at them). Times assume each video starts",
         "# at the script's 00:00. Usage:  VIDEO_DIR=/path/to/episodes OUT_DIR=promos bash cut_commands.sh",
         "set -euo pipefail",
         'VIDEO_DIR="${VIDEO_DIR:-.}"', 'OUT_DIR="${OUT_DIR:-promos}"', 'mkdir -p "$OUT_DIR"',
         "ENC=(-c:v libx264 -preset veryfast -crf 18 -c:a aac -b:a 192k)", ""]
    ff = "ffmpeg -hide_banner -loglevel error -y"
    for p in ranked[:top]:
        name = f"promo_{p['rank']:02d}_{p['id']}"
        L.append(f"# #{p['rank']} {p['id']} ({p['duration']:.1f}s): {_seg_str(p)}")
        segs = p["segments"]
        parts = [f"{name}.mp4"] if len(segs) == 1 else [f"{name}_seg{i}.mp4" for i in range(1, len(segs) + 1)]
        for s, part in zip(segs, parts):
            L.append(f'{ff} -ss {s["start"]:.3f} -i "$VIDEO_DIR/ep{s["episode"]:02d}.mp4" '
                     f'-t {s["end"] - s["start"]:.3f} "${{ENC[@]}}" "$OUT_DIR/{part}"')
        if len(segs) > 1:
            L += [f"printf \"file '%s'\\n\" {' '.join(parts)} > \"$OUT_DIR/{name}.txt\"",
                  f'{ff} -f concat -safe 0 -i "$OUT_DIR/{name}.txt" -c copy "$OUT_DIR/{name}.mp4"',
                  f'rm "$OUT_DIR/{name}.txt" ' + " ".join(f'"$OUT_DIR/{x}"' for x in parts)]
        L.append("")
    L.append('echo "Done: promos are in $OUT_DIR"')
    return "\n".join(L) + "\n"


def fixed_by(p: dict) -> str:
    return "code (hook moved)" if p.get("auto_fixed") else "LLM repair" if p.get("repaired") else ""


def load_results(files: dict[str, bytes]) -> dict:
    """Rebuild a run result from saved JSON outputs (promos_ranked.json required; run_meta.json and
    rejected_by_rules.json optional), so results can be browsed without re-running or an API key."""
    ranked, meta, rejected = None, {}, []
    for name, data in files.items():
        obj = json.loads(data.decode("utf-8"))
        if isinstance(obj, dict) and "final" in obj:
            meta = obj
        elif isinstance(obj, list) and obj and isinstance(obj[0], dict) and "rank" in obj[0]:
            ranked = obj
        elif isinstance(obj, list) and obj and isinstance(obj[0], dict) and "errors" in obj[0]:
            rejected = obj
    if ranked is None:
        raise ValueError("no promos_ranked.json found (a JSON list of promos with a 'rank' field)")
    meta = {"proposed": "n/a", "rejected_by_rules": len(rejected), "duplicates_removed": "n/a",
            "est_cost_usd": "n/a", "episodes": "n/a", **meta, "final": len(ranked)}
    return {"ranked": ranked, "rejected": rejected, "duplicates": [], "scans": [], "meta": meta}


def report(ranked, rejected, dupes, meta) -> str:
    L = [f"# Promo report: {meta['show']}",
         f"Model `{meta['provider']}/{meta['model']}` | {meta['episodes']} episodes | "
         f"{meta['proposed']} proposed, {meta['rejected_by_rules']} rejected by rules, "
         f"{meta.get('auto_fixed', 0)} auto-fixed (hook moved by code), {meta['repaired']} repaired by LLM, "
         f"{meta['duplicates_removed']} duplicates removed ({meta.get('duplicates_scene_continuation', 0)} continue a "
         f"scene across episodes), {meta['final']} final | "
         f"{meta['llm_usage']['calls'] + (meta.get('cross_llm_usage') or {}).get('calls', 0)} calls, "
         f"~${meta['est_cost_usd']} | {meta['wall_seconds']}s", "",
         "## Post these first (top 10)", ""]
    for p in ranked[:10]:
        j = p.get("judge") or {}
        fix = f" ({fixed_by(p)})" if fixed_by(p) else ""
        L += [f"**#{p['rank']} {p['id']}** ({p['score']}) {p['source']}{fix}, {p['duration']:.0f}s: {_seg_str(p)}",
              f"- Hook: {p.get('hook')}", f"- Cliffhanger: {p.get('cliffhanger')}", f"- Why: {p.get('why')}",
              f"- Cold viewer: hook {j.get('hook')}/5, cliff {j.get('cliffhanger')}/5, "
              f"standalone {j.get('standalone')}/5. Read it as: {j.get('what_happens')}",
              f"- Shows resolution: {'YES (-10 pts)' if shows_resolution(p) else 'no'}. {j.get('resolution_reason') or ''}"]
        if p.get("judge2"):
            j2 = p["judge2"]
            L.append(f"- Second judge: hook {j2.get('hook')}/5, cliff {j2.get('cliffhanger')}/5, standalone "
                     f"{j2.get('standalone')}/5, overall {j2.get('overall')}/10, resolution {j2.get('shows_resolution')}"
                     + (f". **Needs human review** ({', '.join(p['judge_disagreement'])})" if p.get("needs_human_review") else ""))
        L.append("")
    if c := meta.get("judge_comparison"):
        l1, l2 = c["judges"]
        L += ["## Second-opinion judge", "",
              f"`{meta['provider']}/{meta['model']}` vs `{meta['judge2_model']}` on {c['promos_compared']} final promos. "
              f"Final score = average of both judges. Spearman rank correlation of their scores: "
              f"**{c['spearman_rank_correlation']}**. Top-10 overlap: **{c['top10_overlap']}/10**. "
              f"{c['needs_human_review']} final promos need human review (big disagreement). See judge_comparison.csv.",
              "", "Biggest disagreements:"]
        L += [f"- #{d['rank']} {d['id']}: {l1} {d[f'{l1}_score']} vs {l2} {d[f'{l2}_score']} "
              f"({', '.join(d['where']) or 'many small gaps'})" for d in c["biggest_disagreements"]]
        L.append("")
    weak = [p for p in ranked if (p.get("judge") or {}).get("standalone", 5) <= 2 or (p.get("judge") or {}).get("hook", 5) <= 2]
    resolved = [p["id"] for p in ranked if shows_resolution(p)]
    L += ["## Where it picks badly", "",
          f"- {len(rejected)} proposals broke hard rules (length / hook timing / bad beat IDs) and were dropped. See rejected_by_rules.json.",
          f"- {len(resolved)} final promos show the resolution per the cold viewer (-10 pts each): {', '.join(resolved) or 'none'}",
          f"- {len(weak)} valid promos scored <=2 on hook or standalone by the cold viewer:"]
    for p in weak[:8]:
        j = p["judge"]
        L.append(f"  - {p['id']} ({_seg_str(p)}): confusing = {', '.join(j.get('confusing') or []) or 'n/a'}")
    L += ["", "## Full ranking", "", "| # | id | score | type | dur | segments | hook |", "|---|---|---|---|---|---|---|"]
    L += [f"| {p['rank']} | {p['id']} | {p['score']} | {p['source']} | {p['duration']:.0f}s | {_seg_str(p)} | "
          f"{(p.get('hook') or '').replace('|', '/')} |" for p in ranked]
    return "\n".join(L)
