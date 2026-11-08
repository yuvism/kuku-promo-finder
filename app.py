"""Streamlit UI.  Run:  streamlit run app.py"""
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from promo_finder.llm import DEFAULT_MODELS, LLM
from promo_finder.parser import load_show
from promo_finder.pipeline import (cut_commands_sh, cut_list_csv, fixed_by, load_results, run,
                                   shows_resolution)

st.set_page_config(page_title="Promo Finder", page_icon="🎬", layout="wide")
st.title("🎬 Promo Finder")
st.caption("Reads microdrama screenplays and returns ranked 20 to 60s promo clips: "
           "hook in the first 3s, cliffhanger cut, timestamps you can hand to an editor.")


def secret(name):
    try:
        return st.secrets.get(name) or os.getenv(name)
    except Exception:
        return os.getenv(name)


FIND, BROWSE = "Find promos in scripts", "Browse existing results"
mode = st.radio("Mode", [FIND, BROWSE], horizontal=True,
                help="Browse loads promos_ranked.json (plus run_meta.json / rejected_by_rules.json if you have them) "
                     "from an earlier run. No API key and no API calls.")

with st.sidebar:
    st.header("Model")
    provider = st.selectbox("Provider", [p for p in DEFAULT_MODELS], index=0,
                            help="'heuristic' is an offline keyword baseline, no key needed")
    model = st.text_input("Model name", DEFAULT_MODELS[provider])
    env_name = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY", "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY"}.get(provider)
    key = None
    if env_name:
        key = st.text_input(f"API key ({env_name})", type="password",
                            help="Used only for this session, never stored") or secret(env_name)
    base_url = None
    if provider == "openai":
        base_url = st.text_input("OpenAI-compatible base URL (optional)", "",
                                 help="Any OpenAI-compatible endpoint, e.g. https://openrouter.ai/api/v1 or a company "
                                      "gateway. Blank = api.openai.com") or None
    st.header("Run")
    cross = st.checkbox("Also build multi-segment promos across episodes", True)
    limit = st.number_input("Only first N episodes (0 = all)", 0, 500, 0)
    gap = st.number_input("Seconds between API calls", 0.0, 30.0, 0.0 if provider == "heuristic" else 7.0,
                          help="Free tiers limit requests per minute")

res = None
if mode == FIND:
    files = st.file_uploader("Upload a .zip of episode scripts or several .txt files",
                             type=["zip", "txt"], accept_multiple_files=True)

    if files:
        eps = load_show(files={f.name: f.getvalue() for f in files})
        if limit:
            eps = eps[: int(limit)]
        default_show = Path(files[0].name).stem.split("-scripts")[0].replace("_", " ").strip()
        show = st.text_input("Show title", default_show)
        st.write(f"Parsed **{len(eps)} episodes**, {sum(len(e.beats) for e in eps)} timestamped beats, "
                 f"{sum(e.duration for e in eps) / 60:.0f} min of story.")

        if st.button("Find promos", type="primary", disabled=(env_name is not None and not key)):
            tmp = Path(tempfile.mkdtemp())
            llm = LLM(provider, model, api_key=key, cache_dir=tmp / ".cache", min_interval=gap, base_url=base_url)
            with st.status("Running...", expanded=True) as status:
                res = run(eps, show, llm, tmp / "out", cross=cross, log=st.write)
                status.update(label="Done", state="complete")
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                for f in (tmp / "out").iterdir():
                    z.write(f, f.name)
            st.session_state["res"], st.session_state["zip"] = res, buf.getvalue()
        elif env_name and not key:
            st.info(f"Add your {env_name} in the sidebar, or pick 'heuristic' to try the offline baseline.")
    res = st.session_state.get("res")
else:
    saved = st.file_uploader("Upload promos_ranked.json from a run (add run_meta.json and rejected_by_rules.json "
                             "for the full picture)", type=["json"], accept_multiple_files=True, key="browse_files")
    if saved:
        try:
            res = load_results({f.name: f.getvalue() for f in saved})
        except (ValueError, UnicodeDecodeError) as e:  # JSONDecodeError is a ValueError
            st.error(f"Couldn't read those files: {e}")

if res:
    m = res["meta"]
    errs = [f"Ep {s['episode']}: {s['error']}" for s in res["scans"] if s.get("error")]
    if errs:  # surface failed episodes instead of silently showing 0 promos
        st.error(f"{len(errs)} of {m['episodes']} episodes failed. First error: {errs[0]}")
    c = st.columns(5)
    c[0].metric("Proposed", m["proposed"])
    c[1].metric("Failed rules", m["rejected_by_rules"])
    c[2].metric("Duplicates", m["duplicates_removed"])
    c[3].metric("Final promos", m["final"])
    cost = m.get("est_cost_usd")
    c[4].metric("Est. cost", f"${cost}" if isinstance(cost, (int, float)) else "n/a")
    if mode == FIND and st.session_state.get("zip"):
        st.download_button("Download all outputs (.zip)", st.session_state["zip"], "promo_outputs.zip")
    if res["ranked"]:  # editor hand-off, rebuilt from the ranked list so it works in browse mode too
        d1, d2, _ = st.columns([1, 1, 2])
        d1.download_button("Cut list (.csv)", cut_list_csv(res["ranked"]), "cut_list.csv", "text/csv")
        d2.download_button("ffmpeg commands, top 20 (.sh)", cut_commands_sh(res["ranked"]), "cut_commands.sh",
                           "text/x-shellscript", help="Expects one video per episode named ep01.mp4, ep02.mp4, ...")

        st.subheader("Where the promos sit in the season")
        segs = pd.DataFrame([{"episode": s["episode"], "start": s["start"], "end": s["end"], "rank": p["rank"],
                              "id": p["id"], "tier": "Top 10" if p["rank"] <= 10 else "Rest",
                              "cut": f"{s['start_ts']}-{s['end_ts']}", "hook": p.get("hook")}
                             for p in res["ranked"] for s in p["segments"]])
        n_eps = max(int(segs["episode"].max()), m["episodes"] if isinstance(m.get("episodes"), int) else 0)
        st.altair_chart(alt.Chart(segs).mark_bar(opacity=0.75, size=10).encode(
            x=alt.X("episode:O", title="Episode", scale=alt.Scale(domain=list(range(1, n_eps + 1)))),
            y=alt.Y("start:Q", title="Seconds into the episode"), y2="end:Q",
            color=alt.Color("tier:N", title="Rank", scale=alt.Scale(domain=["Top 10", "Rest"],
                                                                     range=["#e4572e", "#76b7d6"])),
            order=alt.Order("rank:Q", sort="descending"),  # draw the top-ranked bars last, on top
            tooltip=["rank", "id", "episode", "cut", "hook"]).properties(height=320))

    rows = [{"rank": p["rank"], "score": p["score"], "type": p["source"], "sec": round(p["duration"]),
             "resolves?": ("yes" if shows_resolution(p) else
                           "no" if "shows_resolution" in (p.get("judge") or {}) else ""),
             "fixed by": fixed_by(p),
             "segments": " + ".join(f"Ep{s['episode']} {s['start_ts']}-{s['end_ts']}" for s in p["segments"]),
             "hook": p.get("hook"), "cliffhanger": p.get("cliffhanger"), "why": p.get("why")}
            for p in res["ranked"]]
    st.subheader("Ranked promos")
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    if res["ranked"]:  # selectbox returns None on an empty list
        st.subheader("Inspect a promo")
        by_label = {f"#{p['rank']} {p['id']} ({p['score']})": p for p in res["ranked"]}
        p = by_label[st.selectbox("Promo", list(by_label))]
        j = p.get("judge") or {}
        a, b = st.columns([3, 2])
        a.code(p.get("transcript", ""), language=None)
        b.markdown(f"**Hook:** {p.get('hook')}  \n**Cliffhanger:** {p.get('cliffhanger')}  \n**Why:** {p.get('why')}  \n"
                   f"**Context a new viewer lacks:** {p.get('context_needed')}")
        b.markdown(f"**Cold-viewer judge** (sees only this clip): hook {j.get('hook')}/5 · cliffhanger "
                   f"{j.get('cliffhanger')}/5 · standalone {j.get('standalone')}/5 · overall {j.get('overall')}/10  \n"
                   f"*Read it as:* {j.get('what_happens')}  \n*Confused by:* {', '.join(j.get('confusing') or []) or 'nothing'}  \n"
                   f"*Shows the resolution:* {'yes (-10 pts)' if shows_resolution(p) else 'no'}"
                   f"{' - ' + j['resolution_reason'] if j.get('resolution_reason') else ''}")
        b.json(p["segments"], expanded=False)

    with st.expander(f"Rejected by rule checks ({len(res['rejected'])})"):
        st.json(res["rejected"], expanded=False)
