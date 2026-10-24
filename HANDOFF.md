# HANDOFF.md — Nepal Multi-Hazard Early Warning System (Himalayan-MHEWS)

> **Who this is for:** Any AI coding agent (Gemini, Claude, GPT, etc.) or human engineer picking up this project from where it was left off.
> **Last updated:** 2026-10-23 ~04:15 local (UTC-7)
> **Read these first:** `AGENTS.md` (architecture, domain equations, milestone log), then this file (actual current state, known problems, next actions).
> **Important:** `AGENTS.md` describes what was *intended and built*. **Section 4 of this file ("Known Issues & Honest Caveats") describes where the real system falls short of those claims. Read it before trusting any metric or dashboard value.**

---

## 0. TL;DR for the Next Agent

1. Code for Milestones 1–11 exists and the main pipelines run. **25/25 pytest tests passed** after the Milestone 11 work. After the later ground-creep change, only 11 of them (`test_audit` and `test_physics`) were re-run, and those passed.
2. **The latest work is NOT committed to git.** The sandbox could not create `.git/index.lock` (read-only filesystem). See Section 6, step 1.
3. The perpetual 30-min orchestrator **is running on the host** (status file still updating), **but it runs the old code**. It was started about 13.6 h before the 03:12 edit to `perpetual_orchestrator.py`, so it must be restarted to pick up changes.
4. Several of the "22 live ingestion points" are **placeholders or derived values, not real sensor feeds**. That includes the "Sentinel-1 InSAR creep" value now shown on the dashboard. See Section 4.
5. The weekly AI audit **appended 5 demo/seed records into the real training catalog**. Its deterministic vision classifier also **labels every candidate, including true scars, as `SNOWMELT_RETREAT`**. Fix both before a 6-month unattended run (Section 6).
6. The user's stated goal: **let the system run, collect data, and self-calibrate for ~6 months** (through pre-monsoon and monsoon).

---

## 1. Project Mission (short)

The system provides early warning and susceptibility mapping for cascading hazards in Nepal: rock-ice avalanches, GLOFs, monsoon landslides and landslide-dam breaches, and co-seismic landslides. It combines:
- **Tier A:** static 30 m susceptibility (physics + ML).
- **Tier B:** dynamic precursors (rain, melt, seismic, creep, lake growth).
- **Multi-satellite physics variable synthesis**, used where no borehole data exist.

Pilot basins are Langtang/Trishuli (Rasuwa) and Budhi Gandaki (Gorkha). Coverage has since expanded to 10 corridors and **491 sites** (the user sometimes says "479"; the catalog actually holds 491 = 468 rainfall landslides + 13 cryosphere sites + 10 radar stations).

Full domain equations, thresholds and alert matrix: **`AGENTS.md` §3 and §6**.

---

## 2. Environment & Operational Rules (MUST follow)

| Rule | Detail |
|---|---|
| **Thread caps (shared 256-core host)** | Always set `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4` before any Python/ML run. Unbounded XGBoost/BLAS spawns ~136 threads and stalls. |
| **Python** | `backend/.venv/bin/python` (Python 3.12.3). Run with `PYTHONPATH=backend`. |
| **Slow imports** | `import torch` takes ~11–18 s and `import sklearn` ~9 s on this host. The full test suite takes ~10 min, and the FastAPI warm-up (susceptibility fit + runoff autoencoder) takes ~2–4 min. Commands that look hung are usually just slow. Wait before killing them. |
| **Sandbox has no network/DNS** | Agent sandbox commands cannot reach the internet (`Name or service not known`). Live API calls (Open-Meteo, USGS, RSS, Copernicus S3, Gemini/Claude) only work on the host or with sandbox bypass. `dem_processor.py` now does a DNS pre-flight so it falls back instantly instead of hanging in GDAL. |
| **Sandbox cannot see host processes** | `ps`, `tmux ls` and `curl localhost:8000` from the sandbox return nothing or are denied. Check host state through files (`backend/data/perpetual_status.json` mtime) or ask the user. |
| **`.git` is read-only from the sandbox** | `git add`/`commit` fail. The user must commit on the host, or the agent must request sandbox bypass. |
| **Credentials** | Never print API keys. Check presence with `grep -sq "^GEMINI_API_KEY=" ~/.env`. Keys are read from env: `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OLLAMA_URL`, `OLLAMA_MODEL`. |
| **Ollama** | Runs on the user's Windows PC and is reached through `ssh -R 11434:127.0.0.1:11434 <user>@<server>`. Use local models only (never `*-cloud`). Currently **not reachable** (the tunnel is down). |
| **LLMs never set alert levels** | The physics engine is the source of truth. LLMs only review, explain, extract or propose. |
| **Rule 8.4 Zero-Catastrophic-Forgetting gate** | A new equation or model is promoted only if Recall ≥ 91.36%, Cryosphere-anchor accuracy = 100%, Live-GT accuracy ≥ 95%, and mean FS deviation ≤ 5.0%. (`AGENTS.md` M9 text says ≤ 1.50%; the **code uses 5.0%**. Reconcile this.) |

---

## 3. What Exists (inventory of the codebase)

### 3.1 Key files

| Path | Purpose |
|---|---|
| `AGENTS.md` | Architecture, equations, milestone log (M1–M11). |
| `docker-compose.yml` | PostGIS + TimescaleDB + Redis + API + frontend. |
| `backend/sql/init-timescaledb.sql` | DDL: hypertables `telemetry_snapshots` and `physics_calculations`; spatial tables `sites`, `ground_truth_events`, `hazard_alerts`; **M11 queue tables** `unprocessed_bulletins`, `satellite_visual_candidates`, `ai_audit_runs`. |
| `backend/app/core/database.py` | `BaseTimeSeriesStore` → `TimescalePostGISStore` / `SQLiteTimeSeriesStore` (auto-fallback when port 5432 is closed, which is the current state). Includes SQLite auto-migration (`PRAGMA table_info` + `ALTER TABLE ADD COLUMN`) and queue methods. |
| `backend/app/engine/sites_catalog.py` | Catalog of 491 sites. |
| `backend/app/engine/perpetual_orchestrator.py` | 30-min loop: fetch weather → synthesize physics → FS → triggers → alert → store. Stages bulletins and visual candidates every 12 cycles (zero LLM tokens). |
| `backend/scripts/start_perpetual_system.sh` | Launches the orchestrator in tmux session `nepal-mhews-perpetual`. |
| `backend/scripts/run_with_ollama.sh` | Launches the FastAPI server on `0.0.0.0:8000`. |
| `backend/app/ingestion/expanded_weather.py` | Batched Open-Meteo + USGS FDSN telemetry (the genuinely live inputs). |
| `backend/app/ingestion/unstructured_collector.py` | **M11:** RSS (Google News EN/NE, Kathmandu Post) + `SEED_DISASTER_BULLETINS` (5 hardcoded demo bulletins) + Pillow-generated **synthetic** 512×512 thumbnails. 77-district matching in English, plus a Devanagari map covering ~39 districts. |
| `backend/scripts/run_weekly_ai_audit.py` | **M11:** 4-pillar audit (NLP bulletins → GT catalog; vision screening; residual-driven equation proposal + Rule 8.4 gate; markdown report). Engines: `deterministic`, `ollama`, `gemini`, `claude`, `auto`. Flags: `--dry-run`, `--limit-bulletins`, `--limit-visual`. |
| `backend/app/ml/continuous_equation_optimizer.py` | Evolutionary tuning of 7-factor equation weights. sklearn metrics were replaced with NumPy equivalents to remove a ~9 s import. |
| `backend/app/ml/susceptibility.py` | Hybrid real + physics model, `DifferentialPhysicsCorrector`, neighbor-block granulation, `synthesize_point_physics_from_terrain` (7-factor formula). |
| `backend/app/ml/pinn_inversion.py`, `moisture_downscaler.py`, `runoff_anomaly.py`, `ollama_evaluator.py` | AI accelerators, SMAP downscaler, Tier B autoencoder, Ollama second opinion. |
| `backend/app/ingestion/dem_processor.py` | Copernicus DEM COG reader + derivatives. **Now has `_can_reach_s3_host()` DNS pre-flight.** |
| `backend/app/ingestion/ground_truth_feedback.py` | `live_ground_truth_feedback.csv` append/load. |
| `backend/app/api/routes.py`, `schemas.py` | All endpoints (see 3.2). |
| `backend/app/static/dashboard.html` | Leaflet + Google tiles dashboard, 30 s auto-refresh, 22-point telemetry table, Ground-Truth Feedback tab. |
| `backend/tests/` | `test_api`, `test_audit` (new), `test_ingestion`, `test_ml`, `test_physics`: 25 tests. |

### 3.2 Data files (`backend/data/`)

| File | Notes |
|---|---|
| `perpetual_telemetry.db` | SQLite store (~2.3 MB as of now). Expect ~1–2 GB after 6 months at 491 sites × 48 cycles/day. |
| `perpetual_status.json` | Heartbeat. As of 03:49: cycle 28, uptime ~13.6 h, 447 GREEN / 41 YELLOW / 3 ORANGE / 0 RED. |
| `active_equation_parameters.json` / `equation_optimizer_log.json` | Current champion weights and optimizer history. |
| `processed/nepal_training_points.csv` | 949 real points (NASA GLC COOLR + DEM + GEM faults). |
| `processed/himalayan_cryosphere_avalanches_glofs.csv` | 24 cryosphere anchors. |
| `processed/live_ground_truth_feedback.csv` | 19 data rows. **Contains demo and test rows; see Issue #4.** |
| `audit_reports/weekly_ai_audit_*.md` | 8+ reports from test and dev runs. |
| `thumbnails/` | Synthetic before/after PNGs. |

### 3.3 API endpoints (FastAPI, prefix `/api/v1`)

- Core: `/basins`, `/physics/fs`, `/physics/synthesize`, `/alerts/evaluate`, `/alerts/evaluate-ollama`, `/alerts/replay/{id}`, `/ollama/status`, `/ml/differential-blocks`, `/accelerators/dem-and-pinn/{basin_id}`, `/database/metadata`.
- Sites: `GET /sites/{site_id}` (full risk report + 22 ingestion points), `POST /sites/{site_id}/recalculate`, `/engine/status`.
- Feedback: `GET/POST /feedback/ground-truth`.
- **M11 audit (new):** `GET /audit/staging-queues`, `POST /audit/trigger-weekly-audit` (body: `engine`, `limit_bulletins`, `limit_visual`, `dry_run`), `GET /audit/runs`, `GET /audit/reports`, `GET /audit/reports/{filename}` (path-traversal guarded).

### 3.4 Reported model metrics (from `AGENTS.md` and optimizer logs)

| Metric | Value |
|---|---|
| Recall | ~91.4–91.65% |
| ROC-AUC | ~0.91 |
| Cryosphere-anchor accuracy | 100% |
| Live-GT accuracy | 100% |
| FS deviation | ~1.98% |
| Spatial-holdout accuracy | ~85.4% |

> **Caveat:** "Live-GT 100%" is computed over a 19-row catalog that is partly demo data, so it carries little evidential weight right now.

---

## 4. Known Issues & Honest Caveats (READ BEFORE TRUSTING OUTPUTS)

These are real gaps found during this session. Some were described too optimistically in earlier chat replies and in `AGENTS.md`.

| # | Issue | Severity | Where | Fix |
|---|---|---|---|---|
| 1 | **Latest work uncommitted** (read-only `.git` in sandbox). | 🔴 High | repo | Commit on the host (Section 6, step 1). |
| 2 | **"Sentinel-1 InSAR LOS Creep Velocity" is not InSAR data.** It is a physics-derived estimate. In `routes.py` it is computed from `dyn_fs`, slope, cryo flag and saturation; in `perpetual_orchestrator.py` it uses a slope/rain heuristic that then feeds cohesion softening (somewhat circular). The dashboard still labels it Sentinel-1 with an "Every 6–12 Days" cadence, which is misleading. The two formulas also differ, so the API and the orchestrator disagree. Previously it was a hardcoded `2.5` everywhere (user-reported bug; fixed to be site-varying, not real). | 🔴 High (trust) | `routes.py` ~L669–700, `perpetual_orchestrator.py` ~L88–110, `dashboard.html` | Relabel as **"Modelled creep (physics proxy)"** until real InSAR (EGMS / COMET-LiCSAR / ASF HyP3) is ingested. Unify into one function in `app/physics/`. |
| 3 | **Other "live" points are placeholders or derived:** NDVI is hardcoded `0.32`; lake expansion is hardcoded `0.0%`; SWE = `max(20, wind_surcharge*100)`; 30-day API = `max(12, rain_72h*1.5)` (not a real decay-weighted API); SMAP is taken from the synthesizer, not NASA SMAP. **Genuinely live:** rain 72 h / 1 h peak, temperature, pressure, pressure deficit, cloud, wind, humidity, PDD, USGS seismic. **Static:** DEM, slope, curvature, fault distance. | 🟠 Medium | `routes.py` ingestion_points, orchestrator | Label each point as LIVE / DERIVED / PLACEHOLDER in the UI; wire real sources over time. |
| 4 | **Training catalog contaminated.** The live (non-dry) audit run at 02:33 appended 5 rows from the **hardcoded `SEED_DISASTER_BULLETINS`** (`source=AI_AUDIT_DETERMINISTIC`). Earlier dashboard and test submissions also added demo rows (e.g. "Langtang Lirung (NDRRMA)", "Site #427 …"). Seed bulletins are re-staged on every collector run. | 🔴 High (for 6-month learning) | `live_ground_truth_feedback.csv`, `unstructured_collector.py` | Back up the CSV, remove demo/test rows, stop staging seed bulletins in production (env flag `MHEWS_SEED_DEMO=0`), and add a dedup check (lat/lon/date) in `append_ground_truth_event`. |
| 5 | **Deterministic vision classifier is broken:** all 4 candidates, including the "true scar" ones (`vis_cand_jhyaple_flow`, `vis_cand_tatopani_scarp`), came out `SNOWMELT_RETREAT`. The earlier summary "false artifacts rejected" was misleading because true scars were rejected too. Also, thumbnails are synthetic, not real imagery. | 🟠 Medium | `run_weekly_ai_audit.py` `_deterministic_vision_screen` (~L430) | Recalibrate thresholds against the generator's actual colours (compute ΔR−ΔG on the synthetic pairs); add a unit test asserting scar → `TRUE_LANDSLIDE_SCAR`, shadow → `CLOUD_SHADOW_ARTIFACT`. Later, pull real Sentinel-2 chips. |
| 6 | **Equation "promotion" on equal fitness** ("0.9305 > 0.9305"). The comparison or rounding lets a no-op candidate be promoted and saved. | 🟡 Low | `run_weekly_ai_audit.py` Pillar 3 (~L540) | Require improvement > ε (e.g. 1e-4). |
| 7 | **Deterministic engine reports "token usage"** (~2,965 tokens) although no LLM is called. This inflates the sense that AI was involved. | 🟡 Low | `TokenAccountant` calls in deterministic paths | Record 0 tokens for the deterministic engine. |
| 8 | **Gemini / Claude / Ollama adapters never exercised live** (no network in sandbox; Ollama tunnel down). The model id `gemini-3.8-flash` and the Interactions API payload shape are unverified at runtime. | 🟠 Medium | `run_weekly_ai_audit.py` `_call_gemini_json`, `_call_claude_json`, `_call_ollama_json` | Run once on the host: `--engine gemini --dry-run --limit-bulletins 2 --limit-visual 1` and check the report. |
| 9 | **Perpetual orchestrator runs stale code** (started before the 03:12 edit). | 🟠 Medium | host tmux | Restart (Section 6, step 2). |
| 10 | **API server was started as an agent background task** (`run_with_ollama.sh`). It may die when the agent session ends, and it is not under systemd/tmux. | 🟠 Medium | host | Run under tmux or systemd (Section 6, step 3). |
| 11 | **No auto-restart on host reboot** (tmux only). **Weekly cron not installed.** | 🟠 Medium | host | Section 6, steps 3–4. |
| 12 | **Full test suite not re-run after the creep change** (only `test_audit` + `test_physics`, 11 passed). | 🟡 Low | tests | Run the full suite (~10 min). |
| 13 | **TimescaleDB not actually running.** Port 5432 is closed and all data goes to SQLite. The cutover code and migration CLI exist but have not been exercised against a live Postgres. | 🟡 Low (fine for prototype) | `docker-compose.yml` | `docker compose up -d db`, then run `migrate_sqlite_to_timescaledb.py`. |
| 14 | **Threshold doc inconsistency:** FS-deviation gate is ≤ 1.50% in `AGENTS.md` M9 but ≤ 5.0% in code. | 🟡 Low | `AGENTS.md` | Pick one and document it. |
| 15 | **AGENTS.md M11 overstates** ("Verified 100% test pass rate", vision "distinguishes" scars, "Gemini 3.8 Flash" support). | 🟡 Low | `AGENTS.md` | Amend after fixing Issues #5 and #8. |
| 16 | FastAPI uses deprecated `@app.on_event("startup")`; Starlette warns about `httpx`. | ⚪ Cosmetic | `main.py` | Migrate to a lifespan handler. |

---

## 5. Changes Made This Session (chronological, all UNCOMMITTED)

1. **M11 staging DDL and store methods.** Added `init-timescaledb.sql` tables 6–8. In `database.py`: queue CRUD, `get_telemetry_outliers_for_audit`, SQLite column auto-migration for `telemetry_snapshots` (fixes silently failing inserts of `pressure_deficit_hpa`, `relative_humidity_pct`, `cloudburst_flag`).
2. **`unstructured_collector.py`** (new). RSS + seed bulletins + synthetic thumbnails. Added `NEPAL_DISTRICTS_DEVANAGARI` so Nepali text maps to districts.
3. **`perpetual_orchestrator.py`.** Staging every 12 cycles. Later added the slope/rain creep heuristic, `SAR_SLOPE_CREEP_ACCELERATION` trigger at ≥ 10 mm/mo, and fed creep into `insar_los_velocity_mm_month`.
4. **`run_weekly_ai_audit.py`** (new):
   - The bash/python polyglot shebang was replaced with plain `#!/usr/bin/env python3` plus `os.environ.setdefault` thread caps.
   - Fixed invalid `\$` escape.
   - Report table now shows `candidate_id`.
5. **`dem_processor.py`.** Added DNS pre-flight to avoid GDAL `/vsicurl/` hangs offline.
6. **`continuous_equation_optimizer.py`.** Replaced sklearn metrics with NumPy (accuracy, recall, F1, rank-based AUC). Import went from ~9 s to ~0.9 s.
7. **`routes.py` + `schemas.py`:**
   - Added the 5 `/audit/*` endpoints and `WeeklyAuditTriggerRequest`.
   - Added the dynamic creep computation, `ground_creep_mm_month` in `risk_evaluation`, and the creep trigger.
8. **`dashboard.html`:**
   - Fixed the floating site card (`stCardCreep` and `stCardRain` were never updated, so every site showed "2.5 mm/m").
   - The unit is now "mm/mo".
   - Added a JS slope-based preview while the report loads, and updated `INITIAL_REPORT`.
9. **`tests/test_audit.py`** (new). 5 tests: district matching, staging, token accountant, deterministic dry-run audit, audit endpoints.
10. **`AGENTS.md`.** Added the Milestone 11 entry.
11. **Data side effects.** Audit reports were written; `active_equation_parameters.json` was rewritten by a live audit "promotion" (no-op weights); 5 seed rows were appended to `live_ground_truth_feedback.csv`.

---

## 6. Next Action Items (prioritised)

### Phase A — Before the 6-month unattended run (do these first)

1. **Commit work (host shell):**
   ```bash
   cd /data-mqs/users/djena/avalanche_detector/nepal-mhews
   cp backend/data/processed/live_ground_truth_feedback.csv /tmp/gt_backup_$(date +%s).csv
   git add AGENTS.md HANDOFF.md backend/app backend/scripts/run_weekly_ai_audit.py backend/sql backend/tests/test_audit.py
   git commit -m "feat(m11): token-frugal staging queues, weekly AI audit, audit API; fix site-card creep display"
   ```
   Decide separately whether to commit the `backend/data/*` artefacts. Clean them first (Issue #4).
2. **Clean the training catalog** (Issue #4). Remove rows with `source=AI_AUDIT_DETERMINISTIC` that came from seed bulletins, plus manual/test demo rows. Gate seed bulletins behind an env flag. Add dedup.
3. **Fix the vision classifier and promotion epsilon** (Issues #5 and #6), add tests, then run the full test suite:
   ```bash
   env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 PYTHONPATH=backend backend/.venv/bin/pytest backend/tests/ -v
   ```
4. **Relabel non-live telemetry** (Issues #2 and #3) in `routes.py` and `dashboard.html`. Unify the two creep formulas.
5. **Restart services on the host** under tmux (or systemd, preferred):
   ```bash
   tmux kill-session -t nepal-mhews-perpetual; bash backend/scripts/start_perpetual_system.sh
   tmux new -d -s nepal-mhews-api 'bash backend/scripts/run_with_ollama.sh'
   ```
   Then create user systemd units (`nepal-mhews-perpetual.service`, `nepal-mhews-api.service`) with `Restart=always`, and run `loginctl enable-linger $USER` so they survive logout and reboot.
6. **Install the weekly audit cron** (Sunday 02:00):
   ```cron
   0 2 * * 0 cd /data-mqs/users/djena/avalanche_detector/nepal-mhews && OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 PYTHONPATH=backend backend/.venv/bin/python backend/scripts/run_weekly_ai_audit.py --engine auto --limit-bulletins 50 --limit-visual 20 >> backend/logs/weekly_audit.log 2>&1
   ```
   Smoke-test a real LLM engine once with `--dry-run` (Issue #8).
7. **Add monitoring:** a heartbeat check (alert if `perpetual_status.json` mtime is older than 45 min), log rotation for `backend/logs/`, a monthly SQLite `VACUUM`, and a weekly backup of `perpetual_telemetry.db` and `live_ground_truth_feedback.csv`.

### Phase B — During the 6-month run (improves learning quality)

8. **Real ground truth.** Scrape the BIPAD portal (`bipadportal.gov.np`) incident tables and NASA GLC updates. Keep a human-review step (dashboard "approve/reject") before rows enter the training catalog. With no clean labels, 6 months of telemetry cannot improve accuracy.
9. **Real InSAR creep.** Ingest from EGMS-like products, COMET-LiCSAR (covers Nepal), or ASF HyP3 on-demand Sentinel-1 interferograms. Replace the physics proxy.
10. **Real optical change detection.** Pull Sentinel-2 L2A chips via Microsoft Planetary Computer STAC (already listed in `AGENTS.md` §8.2) for ΔNDVI/NDWI. Feed real chips into the vision pillar instead of synthetic thumbnails.
11. **Real decay-weighted 30-day API** (k = 0.88) computed from stored hourly rain history in the DB.
12. **Alerting:** Telegram/email/webhook on ORANGE/RED transitions with the bilingual broadcast, the settlement ETA and a map link. Add rate-limiting so the same site does not re-alert repeatedly.
13. **Adaptive cadence:** switch to a 15-min cycle for corridors where IMERG 72 h > 150 mm.
14. **TimescaleDB cutover** (`docker compose up -d`, run the migration CLI, enable compression) if multi-node scaling or longer retention is needed.

### Phase C — After ~6 months (evaluation)

15. Re-run the spatial-block holdout and Rule 8.4 evaluation on the **cleaned, enlarged** catalog. Compare against the M7 baseline (86.33% accuracy, 91.36% recall, 0.9100 AUC).
16. Run a monsoon backtest: compare the alert timeline against BIPAD-reported events (lead time, hit rate, false-alarm rate per corridor).
17. Only then consider retraining the Tier A XGBoost model and the Tier B autoencoder on the accumulated real telemetry, with chronological splits and no leakage.

---

## 7. How to Run / Verify (quick reference)

```bash
cd /data-mqs/users/djena/avalanche_detector/nepal-mhews
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 PYTHONPATH=backend

# API server (dashboard at http://<host>:8000/)
bash backend/scripts/run_with_ollama.sh

# Perpetual engine (tmux session nepal-mhews-perpetual)
bash backend/scripts/start_perpetual_system.sh

# Weekly audit: dry run, offline
backend/.venv/bin/python backend/scripts/run_weekly_ai_audit.py --engine deterministic --dry-run --limit-bulletins 5 --limit-visual 4

# Tests (~10 min; test_api is the slow one because of model warm-up)
backend/.venv/bin/pytest backend/tests/ -v

# Quick site check without a running server
backend/.venv/bin/python -c "from fastapi.testclient import TestClient; from app.main import app; c=TestClient(app); r=c.get('/api/v1/sites/glc_361').json(); print(r['risk_evaluation'])"
```

Sample observed values after the creep fix (TestClient):

| Site | Dynamic FS | Creep proxy |
|---|---|---|
| `langtang_lirung` | 2.047 | 2.1 mm/mo |
| `glc_361` | 0.98 | 12.1 mm/mo (trips creep trigger) |
| `glc_102` | 1.28 | 3.7 mm/mo |

---

## 8. Conversation Decisions & User Preferences (context)

- The user wants **token-frugal AI**: zero LLM calls in 30-min cycles, with intelligence batched weekly. Local Ollama is preferred where possible; Gemini/Claude are optional through API keys.
- The user wants **plain-language explanations** in reports and dashboard (Ground-Truth tab already has a glossary and a big "is there a landslide/avalanche risk" indicator plus a single recalibration button).
- **Dashboard cosmetic changes** come from the user separately; do not redesign without asking.
- The user asked for **real dynamic values**, not constants. They noticed the constant 2.5 creep, so expect scrutiny of any placeholder. Be explicit about what is live versus modelled.
- Long-term goal: **run ~6 months, accumulate data, train, and keep searching for better equations** under the Rule 8.4 gate.
- Online Nepal landslide sources discussed earlier: BIPAD Portal (NDRRMA), NASA Global Landslide Catalog / COOLR, ICIMOD regional database, DesInventar Nepal, Nepal Police disaster updates, Kathmandu Post / Google News feeds.

