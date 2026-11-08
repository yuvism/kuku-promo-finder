# Promo Finder (Kuku AI Generalist assignment, Problem 1)

Reads microdrama screenplays and returns **ranked 20 to 60 s promo clips** with exact segment timestamps,
the hook, the cliffhanger and why each works. Single clips and multi-segment clips stitched across episodes.

## How it works
```
parse scripts (code) ──> scan each episode (LLM) ──> validate + snap timestamps (code)
      ──> late hook? start on the hook beat (code) ──> repair the rest once (LLM)
      └─> stitch multi-segment promos across episodes (LLM) ──> validate (code)
      └─> "cold viewer" judge that sees ONLY the clip (LLM) ──> score, dedupe, rank (code) ──> cut list + ffmpeg script
```
* **The LLM never writes a timestamp.** It points at beat IDs (`E03-B017`) the parser extracted; code turns them into
  times, snaps cuts to whole dialogue lines, and rejects anything outside 20 to 60 s or with the hook later than 3 s.
  If the only problem is a late hook, code moves the start to the hook beat (`auto_fixed`); only the rest go to the LLM.
* **Independent quality check.** A second prompt plays a TikTok user who has never seen the show and scores hook,
  cliffhanger and standalone clarity from the clip transcript alone. It also flags clips that show the resolution
  (`shows_resolution`, -10 points), since a promo should cut before the payoff.
* **Dedupe** drops promos that overlap a better one by >50%, and promos that continue the same scene across an
  episode break (A ends in episode N's last 8 s, B starts in episode N+1's first 60 s).
* **Model is swappable**: `--provider gemini|groq|openai|anthropic|heuristic --model <name>`. Any OpenAI-compatible
  endpoint works via `OPENAI_BASE_URL`. `heuristic` is an offline keyword baseline (no key) for testing and comparison.
* All LLM answers are cached in `outputs/.cache`, so a re-run is free and a crashed run resumes where it stopped.

## Run it
```bash
pip install -r requirements.txt
cp .env.example .env          # paste your free Gemini key (aistudio.google.com -> API keys)

# quick test on 3 episodes (writes to outputs/<show>__<provider>__first3ep/, never over a full run)
python run.py path/to/scripts.zip --episodes 3

# full show: ~106 API calls on a cold cache (measured on 50 episodes; fits the Gemini Flash-Lite free tier of
# 500/day). --strong-model is used for 1 call only (3.8 Flash free tier is 20/day). Re-runs hit the cache.
python run.py path/to/scripts.zip --strong-model gemini-3.8-flash

# other models
python run.py path/to/scripts.zip --provider groq
python run.py path/to/scripts.zip --provider heuristic     # offline baseline
```
Outputs land in `outputs/<show>__<provider>/`:
`report.md` (top 10 + failures), `promos_ranked.csv/json`, `rejected_by_rules.json`,
`duplicates_removed.json`, `episode_scans.json`, `run_meta.json` (calls, tokens, est. cost at paid-tier list price, time),
`cut_list.csv` (one row per segment) and `cut_commands.sh`.

**Cutting the videos.** `cut_commands.sh` cuts the top 20 promos with ffmpeg and joins multi-segment ones. It expects
one video per episode named **`ep01.mp4`, `ep02.mp4`, ...** (placeholder names: rename your files to match) whose
timeline starts at the script's 00:00. Needs bash + ffmpeg (on Windows: Git Bash).
```bash
VIDEO_DIR=/path/to/episodes OUT_DIR=promos bash cut_commands.sh   # -> promos/promo_01_P004.mp4, ...
```

UI: `streamlit run app.py`, then upload the zip, paste a key and click **Find promos**. The app also shows a season
timeline of where promos sit, offers the cut list and ffmpeg script as downloads, takes an optional
OpenAI-compatible base URL for the `openai` provider, and has a **Browse existing results** mode that opens a saved
`promos_ranked.json` (+ `run_meta.json`) with no API key.

## Files
`promo_finder/parser.py` screenplay parser · `prompts.py` all prompts · `pipeline.py` stages, rules, scoring ·
`llm.py` provider wrapper + cache + retries · `heuristic.py` offline baseline · `run.py` CLI · `app.py` Streamlit UI
