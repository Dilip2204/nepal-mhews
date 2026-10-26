# INSAR_NOTES.md — measured ground deformation: what was built, what it means, what is wrong

> Written 2026-10-26 to be picked up cold later. Everything here was measured against live
> APIs and real products, not assumed. Where something is unverified it says so.
> Companion docs: `AGENTS.md`, `HANDOFF.md`, `SIX_MONTH_RUN_CHECKLIST.md`.

---

## 1. Why this exists

The dashboard used to display a **physics heuristic** labelled "Sentinel-1 InSAR LOS Creep
Velocity". It was never InSAR. The goal of this work was to make a genuinely *measured*
deformation velocity possible, and to make the system refuse to claim one when it cannot
measure it.

---

## 2. The result, in one line

For the ORANGE site `glc_105` (29.5023, 82.2504, 2706 m):

```
-0.3 +/- 6.7 mm/yr LOS
39 pairs over 17 epochs, 204 days, residual RMS 2.7 mm, 11 epochs dropped (disconnected)
```

**Read this as: no detectable movement.** −0.3 against ±6.7 is statistically
indistinguishable from zero.

**It does NOT mean the slope is stable.** The ±6.7 mm/yr uncertainty means we can only rule
out motion faster than roughly **13 mm/yr (2 sigma)**. Creep slower than that is invisible at
this precision. "Not detected" ≠ "not moving".

Useful context: the physics engine rates `glc_105` ORANGE (FS 0.96) from rainfall and
terrain. InSAR independently says there is no measurable deformation there. That disagreement
is the point of having a second, physical observation.

---

## 3. How it works (the four steps)

1. **Find bursts** — NASA CMR (`short_name=SENTINEL-1_BURSTS`), open, no credentials.
   Group into stacks sharing the same relative burst id + swath + polarization.
2. **Plan a network** — connect each acquisition to its next 3 neighbours (small-baseline),
   giving redundancy ~2.7 over a ~34-epoch annual stack.
3. **Submit to ASF HyP3** — `INSAR_ISCE_BURST`, 1 credit per pair, ~1 h each.
4. **Collect and invert** — double-difference each finished interferogram, then solve the
   network by least squares and fit a velocity.

### 3.1 Two things that make this correct rather than merely plausible

**Double differencing is mandatory.** HyP3's own product README says the unwrapping
"reference point was set internally by SNAPHU in a manner that minimizes the statistical cost
of the resulting unwrapped solution" — i.e. **every interferogram has its own arbitrary,
undocumented phase offset**. Absolute phase is not comparable across pairs. Stacking raw
values would sum unrelated constants into a confident, meaningless number. Every observation
is therefore `target − reference_annulus` *within the same interferogram* (annulus 600–2500 m),
which cancels the unknown datum and also removes atmosphere common to both locations.

**The network is inverted, not averaged.** For acquisitions t₀..t_N and M interferograms,
each pair (a→b) observes `d_b − d_a`. That is `G·d = obs`, solved by least squares with
`d₀ = 0` as datum. The residual RMS of that fit is the honesty metric — a redundant,
self-consistent network gives small residuals; unwrapping errors and atmospheric outliers
inflate them.

---

## 4. Hard numbers measured from real data

### 4.1 Costs (live `GET /costs`)

| Job type | Credits |
|---|---|
| **INSAR_ISCE_BURST** | **1** ← what we use |
| INSAR_GAMMA | 10 (20x4 looks) / 15 (10x2) |
| INSAR_ISCE_MULTI_BURST | 1–10 |
| AUTORIFT | 25 |
| RTC_GAMMA | 5–60 |

Free tier: **8000 credits/month**. A full network is ~90 credits/site → ~88 sites/month.
**Budget is not the constraint. Coherence is.**

### 4.2 Coherence reality (real product, job `dcf84b0a`, Langtang, 12-day VV pair)

Only **27.3%** of the scene was usable (coherence ≥ 0.3 **and** successfully unwrapped):

| Elevation | Usable |
|---|---|
| 0–1500 m | **3.5%** — dense vegetation/agriculture decorrelates in 12 days |
| 1500–3000 m | 26.9% |
| 3000–4500 m | **43.6%** — best band, sparse cover and bare rock |
| 4500 m+ | 19.4% — snow and ice decorrelate |

**This is the dominant practical limitation.** InSAR will corroborate *some* sites, never all,
and it is weakest above 4500 m — exactly where the rock-ice avalanche sites are.
Both `langtang_lirung` and a test point at 28.22/85.52 returned coherence **0.000** with
`conncomp 0`: genuinely unmeasurable, correctly reported as `None`.

### 4.3 The glc_105 network outcome

- 90 pairs submitted, **36 came back unusable** (decorrelated / no usable reference annulus)
- The surviving 54 **fragmented the network**: rank 25 of 27 unknowns → 2 epochs unconstrained
- Largest connected component solved: 39 pairs / 17 epochs / 204 days, **11 epochs dropped**

---

## 5. Bugs found and fixed (all were the "confidently wrong" class)

| # | Bug | Why it mattered |
|---|---|---|
| 1 | **Sign inverted.** Code applied the textbook `d = −(λ/4π)·φ` to a raster HyP3 has *already* sign-corrected. README: *"Positive values indicate movement away from the sensor."* | Every velocity pointed the wrong way — downslope creep reported as uplift, with a perfectly plausible magnitude. |
| 2 | **Raster resolution treated as metres unconditionally.** | Real products are UTM so it worked; a geographic-CRS product computes a 3.1-million-pixel reference radius and silently returns `None` forever. |
| 3 | **Per-run job cap truncated a network.** `hyp3_max_jobs_per_run=20` (a per-*site* screening cap) applied to a 90-pair network → redundancy 0.61, below the 1.3 gate. | 20 credits spent on a network the inversion would then refuse. Networks now budget against the monthly allowance only and **refuse rather than part-build**. |
| 4 | **Planner could plan an unusable network.** 40 pairs / 34 epochs = redundancy 1.21 < 1.3 gate. | Same waste. Default raised to 90 + a guard that logs when a plan falls short. |
| 5 | **Exact-zero phase read as "no motion".** HyP3 writes `0.0` where unwrapping *failed*. | Would report a stable slope for an unmeasurable one. Now masked, plus the authoritative `conncomp == 0` flag. |
| 6 | **Rank check too blunt.** A disconnected network was refused outright. | Threw away a valid answer. Now solves the largest connected component and reports `epochs_dropped_disconnected`. |

Also corrected: the single-pair provider was registered as `MEASURED_INSAR` while the
docstring said a single pair is *not* a validated rate. Inconsistent. **Only an inverted
network now earns `MEASURED_INSAR`**; single-pair stays available for screening.

---

## 6. Why a single pair is not a velocity

One interferogram's phase = deformation + **atmospheric delay** + orbital/DEM residual + noise.
In Himalayan relief the atmospheric term is routinely centimetres — the same order as a year
of real creep.

Demonstrated on synthetic data with a known +20.0 mm/yr truth and 8 mm per-epoch atmosphere:

| Estimate | Result | Error |
|---|---|---|
| Single 12-day pair | **+92.4 mm/yr** | 72 |
| SBAS time series (84 pairs) | **+14.4 ± 4.2 mm/yr** | 5.6 |

Same data. One pair is wrong by 4.6×. This is pinned by a test.

---

## 7. Quality gates (a velocity is refused unless all pass)

| Gate | Value | Rationale |
|---|---|---|
| `MIN_PAIRS` | 8 | |
| `MIN_EPOCHS` | 5 | |
| `MIN_REDUNDANCY` | 1.3 | 1.0 is a bare chain — nothing averages out |
| `MAX_RESIDUAL_RMS_MM` | 12.0 | network disagreeing with itself = unwrap errors / severe atmosphere |
| rank check | full rank | else `lstsq` invents minimum-norm values for unconstrained epochs |
| coherence | ≥ 0.3 | |
| `conncomp` | > 0 | ISCE's authoritative "unwrapping failed" flag |

**Refusing is a valid, correct outcome.** It is always preferable to a fabricated number.

---

## 8. What is still NOT claimed

- **Relative, not absolute.** Measured against the local reference annulus. That annulus is
  now *checked* for drift rather than assumed stable, but the rate remains relative to it.
- **DEM error and long-wavelength orbital ramps are unmodelled.**
- **Atmospheric correction is elevation-detrending only**, not a weather-model correction.
  GACOS was unreachable; ERA5 needs a separate registration (section 12).
- Ascending/descending are not combined, so no vertical/horizontal decomposition.
- **Only one site has an inverted network** (`glc_105`). The other 490 run on the modelled
  proxy, and the system now says so on every surface.

---

## 11. What the measurement is wired into (2026-10-26)

Until this date the measured velocity was computed, validated, and then **ignored** — the
physics loop still ran on the heuristic. That gap is closed.

### 11.1 The circularity that was removed

The 30-minute loop did this:

```
est_fs  = f(rain, slope)
creep   = compute_viscoplastic_creep_velocity(est_fs, slope, elevation, rain)
raw.insar_los_velocity_mm_month = creep        # <- labelled InSAR, was never InSAR
dynamic_fs = compute_infinite_slope_fs(synthesize(raw))
```

The "InSAR LOS velocity" fed into the factor of safety was a deterministic function of the
slope and rainfall that FS is *already* computed from. It carried no independent
information, and the `SAR_SLOPE_CREEP_ACCELERATION` trigger fired on it.

### 11.2 The substitution policy

`insar_timeseries.resolve_creep_input(heuristic_mm_month, measured)` decides, in order:

| Case | Condition | Value used | Provenance |
|---|---|---|---|
| Motion detected | `abs(v) >= 2σ` | `abs(v)/12` | `MEASURED_INSAR` |
| Precise null | no motion, and `2σ/12 < heuristic` | `2σ/12` | `MEASURED_INSAR_NULL` |
| Imprecise null | no motion, but `2σ/12 >= heuristic` | heuristic kept | `MODELLED_PROXY` |
| Bad reference | annulus drifting | heuristic kept | `MODELLED_PROXY` |
| No network | — | heuristic kept | `MODELLED_PROXY` |

The third row is the one that matters most. **A wide null must never read as "the slope is
still."** If the measurement is too imprecise to contradict the heuristic, it is
uninformative and the heuristic stands. Only a null tight enough to actually *exclude* the
heuristic's rate is allowed to lower it.

The magnitude `abs(v)` is used, not the signed value: the sign of a LOS rate says which way
the satellite was looking, not whether the slope is in trouble. Where the viewing geometry
supports it (`|look · downslope| >= 0.15`) the slope-parallel projection is preferred over
raw LOS, since downslope motion is what the creep term means.

### 11.3 Live effect on the one measured site

```
glc_105  modelled 10.40 mm/month  ->  used 1.119 mm/month   MEASURED_INSAR_NULL
         "no motion detected (-0.3 +/- 6.7 mm/yr LOS); rules out the modelled
          10.4 mm/month, so the 2-sigma limit 1.12 mm/month is used"
```

The heuristic sat **above** the 10.0 mm/month threshold for
`SAR_SLOPE_CREEP_ACCELERATION`. Under monsoon rain (≥ 80 mm/72 h) that trigger would have
fired on a number no instrument ever measured. It no longer does.

### 11.4 Honest trigger naming

`SAR_SLOPE_CREEP_ACCELERATION` is now emitted **only** when the creep figure came from a
real interferometric measurement. The heuristic path emits `MODELLED_CREEP_ACCELERATION`.
Alert levels are unaffected — the matrix counts triggers, it does not match their names —
so this changes the label, not the alerting.

### 11.5 Hot-path safety

The loop runs ~491 sites every 30 minutes and must never depend on a remote host.
`load_measured_velocities()` reads the state file once per cycle, cached against its mtime,
and `timeseries_velocity_provider(..., allow_network=False)` forbids the one remote read in
the provider (the Copernicus DEM fetch for slope aspect). Aspect is resolved once during
weekly collection and persisted. A test asserts the hot path cannot reach the network.

### 11.6 Where provenance now surfaces

| Surface | Field |
|---|---|
| `GET /api/v1/sites` | `creep_provenance`, `creep_basis` |
| `GET /api/v1/sites/{id}` | `ground_creep_provenance`, `ground_creep_basis`, `insar_measured_mm_year`, `insar_uncertainty_mm_year`, `insar_detection_limit_mm_year` |
| Ingestion point label | "SBAS Measured" vs "Viscoplastic Proxy" |
| Dashboard map tooltip + site card | "Measured InSAR" / "no motion detected" / "Modelled proxy, not InSAR" |
| `physics_calculations` table | `creep_mm_month`, `creep_provenance`, `insar_velocity_mm_year`, `insar_uncertainty_mm_year` |
| Each cycle's stdout | `Creep input provenance: MEASURED_INSAR_NULL=1 \| MODELLED_PROXY=490` |

The DB columns are added idempotently (`PHYSICS_CALC_COLUMNS`, mirroring the existing
`TELEMETRY_COLUMNS` pattern) on both the TimescaleDB and SQLite paths, so an existing
database picks them up at init with no manual migration. Verified against the live SQLite
store without data loss.

---

## 12. Research settled while building this

| Question | Answer |
|---|---|
| NISAR via HyP3? | **No.** Not an available job type. L-band (24 cm) would penetrate the snow and vegetation that decorrelate C-band, so this is worth re-checking. |
| ARIA / OPERA products? | ARIA 60 credits, OPERA 10,001 — both far above the 1-credit `INSAR_ISCE_BURST`. |
| openEO? | Real, works, but a **separate** 10,000-credit allowance, not the HyP3 pool. |
| GACOS atmospheric correction? | Host unreachable from here. |
| ERA5 atmospheric correction? | Needs its own CDS registration. Not wired. |
| Look-vector units? | Radians, confirmed from the HyP3 product README (not degrees). |

### 12.1 Atmospheric correction: a design that did not survive testing

The first attempt modelled atmosphere as a **network-level joint unknown**, solved alongside
the displacements using each pair's `elevation_diff_m`. Testing showed the design matrix at
rank 29 of 30. That was not a test bug: the target and the reference annulus are *the same
two locations in every pair*, so their elevation difference is structurally constant and
cannot represent a per-date atmospheric state. It was reverted.

The shipped version is **per-interferogram elevation detrending**: within each product, fit
phase against elevation across the scene and remove the fitted ramp before taking the
target-minus-reference median. Skipped when the window is too flat (< 150 m range) or too
small (< 60 pixels) for the slope to mean anything. The fitted coefficient is reported per
pair and averaged into the solution, so the correction is auditable rather than silent.

---

## 9. Files

| Path | Role |
|---|---|
| `backend/app/ingestion/insar_hyp3.py` | CMR burst discovery, HyP3 submit/collect, single-pair measurement |
| `backend/app/ingestion/insar_timeseries.py` | Network planning, double differencing, SBAS inversion, the registered provider |
| `backend/app/ingestion/insar_cdse.py` | Scene/pair-formability metadata + the `register_measured_velocity_provider` hook |
| `backend/scripts/refresh_insar_baselines.py` | CLI: `--network SITE_ID`, `--collect-networks`, `--collect`, `--submit`, `--status`, `--dry-run` |
| `backend/tests/test_insar_hyp3.py` | 30 tests |
| `backend/tests/test_insar_timeseries.py` | 23 tests, incl. ground-truth velocity recovery |
| `backend/data/hyp3_insar_state.json` | Job/network/solution state (survives restarts) |
| `backend/data/hyp3_products/` | Downloaded interferograms |

Weekly cron (Sun 03:40 UTC): `--collect-networks --collect --submit`.

---

## 10. Open questions / next steps

1. **Run networks on the three RED sites** (`glc_427` 1112 m, `glc_361` 1650 m, `glc_453`
   2951 m). A *detected* velocity at a RED site would be far more actionable than a null
   result. Note `glc_427` at 1112 m sits in the 3.5%-usable band — expect it to fail.
2. **Tighten the uncertainty.** ±6.7 mm/yr is wide, and it is why glc_105 lands in the
   "precise null" case rather than detecting anything. `--extend-networks` now adds each
   week's new acquisitions to existing networks, lengthening the span — this is the main
   lever and it compounds over the six-month run. ERA5 correction would help further.
3. ~~Use the look vectors to project LOS onto the slope direction.~~ **Done** — see 11.2.
4. **Decide the fallback policy.** Currently a site with no inverted network reports
   `UNAVAILABLE` for measured velocity. Alternative: serve the single-pair value explicitly
   labelled as screening-only. Deliberately not done — it risks re-introducing the exact
   overclaiming this work removed.
5. **Watch the credit burn** if networks are run broadly: 90/site against 8000/month.
