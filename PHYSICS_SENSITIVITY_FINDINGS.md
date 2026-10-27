# Physics sensitivity findings — the factor of safety does not respond to monsoon rain

> Measured 2026-10-26 against the real 491-site catalog and the real synthesizer.
> **Nothing has been changed.** This is an investigation record, written because the
> finding is large enough that fixing it casually would be worse than not fixing it.
> Companion docs: `AGENTS.md`, `INSAR_NOTES.md`, `HANDOFF.md`.

---

## 1. The finding, in one table

Alert distribution across all 491 sites, varying only 72-hour rainfall:

| rain (72 h) | GREEN | YELLOW | ORANGE | RED |
|---|---|---|---|---|
| 0 mm | 432 | 50 | 3 | 6 |
| **120 mm** (the AGENTS.md hydromet alert threshold) | **432** | **50** | **3** | **6** |
| 250 mm | 393 | 88 | 4 | 6 |
| 300 mm | 348 | 120 | 14 | 9 |

**At the documented alert threshold, the physics engine produces an alert distribution
identical to zero rainfall. Not one of 491 sites changes level.**

Supporting numbers, measured per site:

- **415 of 491 sites (84.5%)** have *exactly* the same factor of safety at 200 mm as at 0 mm.
- **491 of 491 sites** need more than 120 mm before FS moves by even 0.1%.
- Median rainfall at which FS first moves: **270 mm**. Range 140–410 mm.
- 287 of 491 need more than 250 mm.

Monsoon rain currently reaches an alert level **only** through trigger *counting*
(`CLOUD_BURST_SATURATION` at >= 120 mm), never through the factor of safety. The physics
is inert in the operational range.

## 2. Why

`backend/app/physics/feature_synthesizer.py`:

```python
smap_baseline_zw = clip((raw.smap_rootzone_moisture_vwc / porosity) * z_m * 0.6, 0.0, z_m)
z_w_m            = clip(max(infiltration_zw, smap_baseline_zw), 0.0, z_m)
```

`smap_rootzone_moisture_vwc` is **never set by the production loop**, so it takes its
dataclass default of 0.31. `dry_bulk_density_g_cm3` is likewise never set, giving
porosity 0.415. So `smap_baseline_zw = 0.448 * z` at **every site, always**, and the
`max()` keeps the rainfall branch below it until roughly
`rain_72h + 0.35 * API > 319 mm`.

Measured at a glc_105-like site: `saturation_ratio` is **0.448 at 0 mm and 0.448 at
120 mm**, moving only at 300 mm (0.632).

## 3. The wider problem: 11 of 22 model inputs are constants

`RawEarthObservationInputs` has 22 fields. `perpetual_orchestrator.py` sets **11**. The
other 11 take dataclass defaults — *identical at all 491 sites*. `gamma`,
`matric_suction_kpa` and `saturation_ratio` are therefore constants everywhere, for the
same reason.

One-at-a-time substitution of real measured values on glc_105 terrain
(production baseline FS 1.0780, P(fail) 0.584):

| input (default → measured) | FS | ΔFS |
|---|---|---|
| `diurnal_lst_range_c` 16.0 → 6.03 | 1.330 | **+0.252** |
| `sentinel2_ndvi` 0.32 → 0.647 | 1.287 | **+0.209** |
| `dry_bulk_density_g_cm3` 1.55 → 1.14 | 1.195 | +0.117 |
| `clay_fraction` 0.16 → 0.247 | 1.184 | +0.106 |
| `smap_rootzone_moisture_vwc` 0.31 → 0.43 | 0.982 | −0.096 |
| `upslope_contributing_area_m2` 4500 → 45000 | 1.003 | −0.075 |
| `imerg_rain_72h_mm` 40 → 150 | 1.078 | **0.000** |

All substituted together: **FS 1.078 → 1.764, P(fail) 0.584 → 0.012, YELLOW → GREEN.**

## 4. Why this must NOT be "just fixed"

glc_105 is a **NASA-GLC catalogued landslide point**. Substituting real values moves it
toward GREEN — i.e. toward a false negative at a site where a landslide is recorded.

That is the crux: **the 85–86% accuracy and the 1.50% FS-deviation envelope were tuned
with these constants in place.** They are not neutral placeholders; they are acting as
*fitted parameters*. Replacing them with real measurements without re-running the
Rule 8.4 gate would silently invalidate the accuracy claim the whole system rests on,
and could push real landslide sites to GREEN.

Any change here has to be:

1. Made behind a flag, with the modelled path kept intact for comparison.
2. Evaluated offline against the full ground-truth catalog, not on single sites.
3. Put through the Rule 8.4 gate, with the *constants themselves* reconsidered as part of
   the recalibration rather than simply overwritten.
4. Compared on recall at known landslide points specifically — overall accuracy can rise
   while recall at the sites that matter falls.

## 5. What a fix would need (not done)

Real sources exist and are reachable, but each needs wiring and validating:

| input | candidate real source | notes |
|---|---|---|
| `smap_rootzone_moisture_vwc` | SMAP L4 | not wired anywhere today; the single highest-leverage term for the rainfall problem |
| `diurnal_lst_range_c` | MODIS LST day/night | largest single ΔFS measured |
| `sentinel2_ndvi` | Sentinel-2 via Planetary Computer | the STAC + SAS path already works in `sentinel2_scar.py` |
| `dry_bulk_density_g_cm3`, `clay_fraction`, `sand_fraction` | SoilGrids | static per site, so a one-off backfill rather than a live feed |
| `upslope_contributing_area_m2` | D8 flow accumulation on the Copernicus DEM | DEM already integrated |

The narrowest possible first change is the saturation floor itself: the
`max(infiltration_zw, smap_baseline_zw)` term is what pins saturation at 0.448 and makes
rainfall inert. Whether that is a bug or a deliberate conservative floor is not
recorded anywhere, and should be established before touching it.
