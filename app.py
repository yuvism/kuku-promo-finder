"""Streamlit UI.  Run:  streamlit run app.py"""
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from promo_finder.llm import DEFAULT_MODELS, LLM
from promo_finder.parser import load_show
from promo_finder.pipeline import run

st.set_page_config(page_title="Promo Finder", page_icon="🎬", layout="wide")
st.title("🎬 Promo Finder")
st.caption("Reads microdrama screenplays and returns ranked 20 to 60s promo clips: "
           "hook in the first 3s, cliffhanger cut, timestamps you can hand to an editor.")


def secret(name):
    try:
        return st.secrets.get(name) or os.getenv(name)
    except Exception:
        return os.getenv(name)


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
    st.header("Run")
    cross = st.checkbox("Also build multi-segment promos across episodes", True)
    limit = st.number_input("Only first N episodes (0 = all)", 0, 500, 0)
    gap = st.number_input("Seconds between API calls", 0.0, 30.0, 0.0 if provider == "heuristic" else 7.0,
                          help="Free tiers limit requests per minute")

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
        llm = LLM(provider, model, api_key=key, cache_dir=tmp / ".cache", min_interval=gap)
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

if res := st.session_state.get("res"):
    m = res["meta"]
    errs = [f"Ep {s['episode']}: {s['error']}" for s in res["scans"] if s.get("error")]
    if errs:  # surface failed episodes instead of silently showing 0 promos
        st.error(f"{len(errs)} of {m['episodes']} episodes failed. First error: {errs[0]}")
    c = st.columns(5)
    c[0].metric("Proposed", m["proposed"])
    c[1].metric("Failed rules", m["rejected_by_rules"])
    c[2].metric("Duplicates", m["duplicates_removed"])
    c[3].metric("Final promos", m["final"])
    c[4].metric("Est. cost", f"${m['est_cost_usd']}")
    st.download_button("Download all outputs (.zip)", st.session_state["zip"], "promo_outputs.zip")

    rows = [{"rank": p["rank"], "score": p["score"], "type": p["source"], "sec": round(p["duration"]),
             "segments": " + ".join(f"Ep{s['episode']} {s['start_ts']}-{s['end_ts']}" for s in p["segments"]),
             "hook": p.get("hook"), "cliffhanger": p.get("cliffhanger"), "why": p.get("why")}
            for p in res["ranked"]]
    st.subheader("Ranked promos")
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    if res["ranked"]:  # selectbox returns None on an empty list, which crashed the line below
        st.subheader("Inspect a promo")
        pick = st.selectbox("Promo", [f"#{p['rank']} {p['id']} ({p['score']})" for p in res["ranked"]])
        p = res["ranked"][int(pick.split()[0][1:]) - 1]
        j = p.get("judge") or {}
        a, b = st.columns([3, 2])
        a.code(p.get("transcript", ""), language=None)
        b.markdown(f"**Hook:** {p.get('hook')}  \n**Cliffhanger:** {p.get('cliffhanger')}  \n**Why:** {p.get('why')}  \n"
                   f"**Context a new viewer lacks:** {p.get('context_needed')}")
        b.markdown(f"**Cold-viewer judge** (sees only this clip): hook {j.get('hook')}/5 · cliffhanger "
                   f"{j.get('cliffhanger')}/5 · standalone {j.get('standalone')}/5 · overall {j.get('overall')}/10  \n"
                   f"*Read it as:* {j.get('what_happens')}  \n*Confused by:* {', '.join(j.get('confusing') or []) or 'nothing'}")
        b.json(p["segments"], expanded=False)

    with st.expander(f"Rejected by rule checks ({len(res['rejected'])})"):
        st.json(res["rejected"], expanded=False)
