# Promo Finder (Kuku AI Generalist assignment, Problem 1)

Reads microdrama screenplays and returns **ranked 20 to 60 s promo clips** with exact segment timestamps,
the hook, the cliffhanger and why each works. Single clips and multi-segment clips stitched across episodes.

## How it works
```
parse scripts (code) ──> scan each episode (LLM) ──> validate + snap timestamps (code) ──> repair once (LLM)
      └─> stitch multi-segment promos across episodes (LLM) ──> validate (code)
      └─> "cold viewer" judge that sees ONLY the clip (LLM) ──> score, dedupe, rank (code)
```
* **The LLM never writes a timestamp.** It points at beat IDs (`E03-B017`) the parser extracted; code turns them into
  times, snaps cuts to whole dialogue lines, and rejects anything outside 20 to 60 s or with the hook later than 3 s.
* **Independent quality check.** A second prompt plays a TikTok user who has never seen the show and scores hook,
  cliffhanger and standalone clarity from the clip transcript alone.
* **Model is swappable**: `--provider gemini|groq|openai|anthropic|heuristic --model <name>`. Any OpenAI-compatible
  endpoint works via `OPENAI_BASE_URL`. `heuristic` is an offline keyword baseline (no key) for testing and comparison.
* All LLM answers are cached in `outputs/.cache`, so a re-run is free and a crashed run resumes where it stopped.

## Run it
```bash
pip install -r requirements.txt
cp .env.example .env          # paste your free Gemini key (aistudio.google.com -> API keys)

# quick test on 3 episodes
python run.py path/to/scripts.zip --episodes 3

# full show (~90 calls: fits Gemini Flash-Lite free tier of 500/day; 3.8 Flash free tier is only 20/day)
python run.py path/to/scripts.zip --strong-model gemini-3.8-flash

# other models
python run.py path/to/scripts.zip --provider groq
python run.py path/to/scripts.zip --provider heuristic     # offline baseline
```
Outputs land in `outputs/<show>__<provider>/`:
`report.md` (top 10 + failures), `promos_ranked.csv/json`, `rejected_by_rules.json`,
`duplicates_removed.json`, `episode_scans.json`, `run_meta.json` (calls, tokens, cost, time).

UI: `streamlit run app.py`, then upload the zip, paste a key and click **Find promos**.

## Files
`promo_finder/parser.py` screenplay parser · `prompts.py` all prompts · `pipeline.py` stages, rules, scoring ·
`llm.py` provider wrapper + cache + retries · `heuristic.py` offline baseline · `run.py` CLI · `app.py` Streamlit UI
