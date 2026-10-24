# SIX_MONTH_RUN_CHECKLIST.md — Unattended 6-Month Operation Readiness

> **Goal:** leave Himalayan-MHEWS running unattended for ~6 months (through pre-monsoon
> and monsoon), collecting telemetry and self-calibrating, with weekly dual-model AI audits.
> **Companion docs:** `AGENTS.md` (architecture), `HANDOFF.md` (state + known caveats).
> **Status legend:** `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked/needs user action

---

## Phase 0 — Blockers found during readiness review (gate everything below)

- [x] **P0.1 — `NameError: np` crash in the perpetual orchestrator.**
  `perpetual_orchestrator.py` used `np.clip` at the creep calculation without importing numpy. Fixed by importing numpy and refactoring to `compute_viscoplastic_creep_velocity`.
- [x] **P0.2 — Deterministic vision classifier labels every candidate `SNOWMELT_RETREAT`**,
  Fixed chromaticity/cloud screening in `run_weekly_ai_audit.py` to identify true scars vs cloud shadow artifacts; verified by `test_generated_scar_is_classified_as_a_true_scar`.
- [x] **P0.3 — Equation promoted on equal fitness** (`0.9311 > 0.9311` logged as "Promoted!").
  Enforced strict improvement threshold (`candidate_fitness > baseline_fitness + 1e-4`); verified by unit tests.
- [x] **P0.4 — Training catalog contaminated** with hardcoded demo/seed rows.
  Cleaned via `clean_training_catalog.py --apply`: 12 contaminated/test rows safely pruned into timestamped backup; 7 clean historical rows retained. Seed staging gated behind `MHEWS_SEED_DEMO=0`.
- [x] **P0.5 — Deterministic engine reports fake token usage** (~2,965 tokens with no LLM call).
  Zeroed out artificial token reporting for deterministic paths; verified by `test_deterministic_paths_report_zero_api_tokens`.

---

## A — Weekly AI audit cron job (dual Gemini **and** Claude)

- [x] **A.1** Add `DUAL_CONSENSUS` engine: Gemini and Claude analyse each item **independently**,
  then cross-validate (`AUDIT_REQUIRE_DUAL_CONSENSUS=1`). Disagreements quarantined.
- [x] **A.2** Migrate Claude path to official `anthropic` SDK, model `claude-opus-5`, adaptive thinking, structured JSON.
- [x] **A.3** Keep Gemini on raw HTTP (different provider), model `gemini-3.8-flash`.
- [x] **A.4** Real token/cost accounting per engine; deterministic engine reports 0 tokens.
- [x] **A.5** Fixed deterministic vision thresholds + verified unit tests.
- [x] **A.6** Promotion epsilon > 1e-4 verified.
- [x] **A.7** `install_mhews_cron.sh` created — Sunday 02:00 UTC, thread caps, log redirection.
- [~] **A.8** Keys added to `backend/.env` and confirmed loading correctly (dotenv strips the
  surrounding quotes). First LIVE dual run on 2026-10-23 exercised both providers for real:
  - **Gemini — WORKING.** `gemini-3.8-flash` returns persistent `503 UNAVAILABLE` ("high demand").
    Added automatic failover to `gemini-3.6-flash` / `gemini-3.5-flash-lite`, covering both the
    503 and read-timeout forms of overload. Verified live: returns a valid parsed vote.
  - [!] **Claude — BLOCKED, needs user action.** Every call fails
    `400 invalid_request_error: "This API key is not scoped to a workspace"`. The key is
    ORGANIZATION-scoped. Fix either way:
      (a) create a **workspace-scoped** key in the Anthropic Console and swap it in, or
      (b) set `ANTHROPIC_WORKSPACE_ID=<id>` in `backend/.env` (support added; sent as the
          `anthropic-workspace-id` header).
  - Until Claude answers, `AUDIT_REQUIRE_DUAL_CONSENSUS=1` means a lone Gemini vote is LOW
    confidence and **admits nothing** to the training catalog. That is correct, conservative
    behaviour — but it means the audit cannot yet learn. Verified in the live run:
    3 bulletins scanned -> 0 ingested, 3 quarantined, 0 tokens billed, nothing fabricated.

## B — Machine reboot resilience (systemd)

- [x] **B.1** `nepal-mhews-perpetual.service` — orchestrator, `Restart=always`.
- [x] **B.2** `nepal-mhews-api.service` — FastAPI dashboard, `Restart=always`.
- [x] **B.3** `backend/scripts/install_systemd_services.sh` created with linger enabling instructions.
- [ ] **B.4** User executes `bash backend/scripts/install_systemd_services.sh` on host SSH.

## C — Database growth, storage & monitoring (6-month projection)

Projection: 491 sites × 48 cycles/day × 180 days ≈ **4.24M records**, ~1.2–2.0 GB SQLite
(~250–350 MB compressed in TimescaleDB).

- [x] **C.1** Monthly `VACUUM` + `ANALYZE` maintenance script (`backend/scripts/db_maintenance.py`).
- [x] **C.2** Weekly backup script (`backend/scripts/backup_mhews_data.sh`).
- [x] **C.3** Heartbeat monitor (`backend/scripts/heartbeat_monitor.py`).
- [x] **C.4** Log rotation config (`backend/deploy/logrotate/nepal-mhews`).

## D — Automated emergency alerts (Telegram / webhook / email)

- [x] **D.1** `app/alerting/` created with multi-channel dispatch on ORANGE/RED **transitions only**.
- [x] **D.2** Payload carries the bilingual Nepali/English CAP broadcast, settlement arrival ETA, FS, active triggers, and map link.
- [x] **D.3** Per-site+level cooldown (default 180 min) preventing alert spam during sustained monsoons.
- [x] **D.4** Wire into the orchestrator; physics sets the level, never an LLM.

## E — Direct InSAR interferogram ingestion (CDSE)

- [x] **E.1** `app/ingestion/insar_cdse.py` created with live OData catalogue integration.
- [~] **E.2** Honest provenance labelling is DONE (`MEASURED_INSAR` / `MODELLED_PROXY` /
  `UNAVAILABLE`, with `register_measured_velocity_provider()` as the extension point and tests
  asserting nothing is ever upgraded to MEASURED without a real backing product).
  **Real measured velocities are still NOT implemented** — do not read this item as finished.
  Source survey (verified 2026-10-23):
  - *EGMS* — reachable but EEA39/Europe only. Unusable for Nepal.
  - *COMET-LiCSAR* — frame metadata serves 200, but `interferograms/<pair>/` and `epochs/<date>/`
    are listed in the index yet **404 on access**; Nepal frames are also 9–27 months stale
    (085A latest pair 2025-01-10, 019D latest 2024-07-10). Dead end.
  - *NASA CMR* — **open, no auth**, current (latest granule over Langtang 2026-06-25). Good for
    real acquisition/pair-formability metadata today.
  - *ASF HyP3* — the only viable route to a real LOS velocity. Needs a free NASA Earthdata
    login; free tier 8000 credits/month; products ~1h after submission, so it belongs on the
    WEEKLY cadence, never in the 30-min loop. Config + `.env` keys are already staged
    (`HYP3_ENABLED`, `EARTHDATA_TOKEN`, budget/cadence settings).
  Implementation did not happen: the build workflow hit a session limit. This is the open item.
- [x] **E.3** Unified creep formula `compute_viscoplastic_creep_velocity` across all components.

## F — NDRRMA BIPAD portal harvesting

- [x] **F.1** `app/ingestion/bipad_scraper.py` created to harvest official incident records.
- [x] **F.2** Feeds into the staging queue, never directly into the training catalog.
- [x] **F.3** Deduplication against existing catalog rows by (lat, lon, date).
- [x] **F.4** Degrades cleanly when BIPAD portal is unreachable (`BIPAD_ENABLED=0`).

## G — Adaptive monsoon throttle

- [x] **G.1** Switched perpetual loop from 30 min → 15 min when any monitored site exceeds 72h rainfall of 150 mm.
- [x] **G.2** Surfaced active cadence in `perpetual_status.json` under `"cadence"`.

---

## Deferred (explicitly out of scope for this pass)

- TimescaleDB cutover (port 5432 closed; SQLite fallback is fine at this scale).
- Real Sentinel-2 optical chips replacing synthetic thumbnails.
- Retraining Tier A XGBoost / Tier B autoencoder on accumulated telemetry — correctly belongs
  at the **end** of the 6-month window (`HANDOFF.md` Phase C), on a cleaned catalog.
