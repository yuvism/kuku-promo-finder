"""CLI: python run.py <folder-or-zip> [--provider gemini --model gemini-3.8-flash]"""
import argparse
import os
from pathlib import Path

from promo_finder.llm import LLM, DEFAULT_MODELS
from promo_finder.parser import load_show
from promo_finder.pipeline import run

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ap = argparse.ArgumentParser(description="Find ranked promo clips in microdrama screenplays")
ap.add_argument("input", help="folder of episode .txt files, or a .zip")
ap.add_argument("--show", default=None, help="show title (default: folder/zip name)")
ap.add_argument("--provider", default=os.getenv("PROMO_PROVIDER", "gemini"), choices=list(DEFAULT_MODELS))
ap.add_argument("--model", default=os.getenv("PROMO_MODEL"))
ap.add_argument("--out", default="outputs")
ap.add_argument("--strong-model", default=os.getenv("PROMO_STRONG_MODEL"),
                help="optional stronger model for the single cross-episode call, e.g. gemini-3.8-flash")
ap.add_argument("--no-cross", action="store_true", help="skip multi-segment cross-episode promos")
ap.add_argument("--episodes", type=int, default=0, help="only first N episodes (quick test)")
ap.add_argument("--workers", type=int, default=1, help="parallel episode scans (paid APIs)")
ap.add_argument("--min-interval", type=float, default=float(os.getenv("PROMO_MIN_INTERVAL", 7)),
                help="seconds between API calls (free tiers have per-minute limits)")
a = ap.parse_args()

eps = load_show(a.input)
if a.episodes:
    eps = eps[: a.episodes]
show = a.show or Path(a.input).stem.split("-scripts")[0].replace("_", " ").strip()
llm = LLM(a.provider, a.model, min_interval=0 if a.provider == "heuristic" else a.min_interval,
          cache_dir=Path(a.out) / ".cache")
slug = "".join(c if c.isalnum() else "_" for c in show)[:40] + f"__{llm.provider}"
strong = LLM(a.provider, a.strong_model, min_interval=a.min_interval,
             cache_dir=Path(a.out) / ".cache") if a.strong_model else None
run(eps, show, llm, Path(a.out) / slug, cross=not a.no_cross, workers=a.workers, cross_llm=strong)
