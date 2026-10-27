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

---

## 6. Which of these factors are modelled, frozen, or absent

Prompted by the question "should soil type, cohesion, binding, topography, vegetation
cover and mountain cuts be factored in?" — checked against the code rather than assumed.

**The equations are good. The inputs are starved.** `c_prime` is built as
`c_mineral + c_root + c_ice`, scaled by a fault-damage factor:

```python
c_mineral_kpa = 3.5 + 14.0 * clay_fraction + 4.0 * coarse_rock_fragment_fraction
c_root_kpa    = 12.0 * (sentinel2_ndvi ** 1.4) * clip(gedi_canopy_height_m / 15.0, 0.1, 1.5)
lidar_root_floor_m = 0.35 * gedi_canopy_height_m ** 0.55     # root-anchoring depth floor
```

So soil type, mineral cohesion, root binding from vegetation and root-anchoring depth are
all genuinely modelled. What they are not is *fed*:

| factor | in the equations? | fed real data? |
|---|---|---|
| soil type (`sand_fraction`, `clay_fraction`, `coarse_rock_fragment_fraction`) | yes | **no — frozen default at all 491 sites** |
| soil cohesion (`c_mineral`) | yes, derived | no — inherits the frozen fractions |
| vegetation binding (`c_root` from NDVI + GEDI canopy) | yes | **no — both frozen** |
| dry bulk density / porosity | yes | **no — frozen** |
| topography: elevation, slope, profile & plan curvature | yes | **yes** |
| topography: aspect | yes | **no — frozen at 195 deg** |
| **anthropogenic slope cuts (roads, terracing, excavation)** | **no** | **absent entirely** |

A terraced farmland slope and a bare rock face therefore receive identical clay fraction,
identical root cohesion and identical porosity. The sophistication of the cohesion model
is currently unreachable.

### 6.1 Mountain cuts are a real, unmodelled driver — and the data is free

Nothing anywhere in `backend/app` models anthropogenic undercutting. The only road
references are in `geocoding.py`, parsing place names out of news text. `runout_routing.py`
knows about settlements, but as *downstream exposure*, not as a cause of instability.

This matters disproportionately in Nepal, where rural road construction cutting into
hillslopes is among the best-documented landslide drivers: a cut removes toe support and
locally oversteepens the slope, which in infinite-slope terms raises the effective
`beta_deg` and strips lateral/basal restraint — exactly the terms the FS is most sensitive
to.

Measured live (OpenStreetMap Overpass, free, no key, HTTP 200): the six lowest-elevation
sites in the catalog sit **18–209 m from a mapped road**:

```
glc_166    76 m elev    18 m from a residential road
glc_357   118 m          30 m  unclassified
glc_043   153 m          75 m  trunk
glc_354   178 m          18 m  residential
```

These are NASA-GLC catalogued landslide points sitting essentially on top of road cuts.
Road proximity is **static**, so it is a one-off backfill into the site catalog, not a
live feed — which matters, because one bulk Overpass query covering six sites returned
843,425 ways in 148 s and would be far too slow for the 30-minute loop.

Candidate derived inputs, none of them built:
- distance to nearest mapped road, and the road class (a trunk road implies a larger cut
  than a footpath)
- whether the road is *below* the site (undercutting, destabilising) or above it
- a cut-slope indicator from high-resolution DEM residuals against a smoothed surface

The same caveat as section 4 applies with full force: adding an anthropogenic term would
change FS at many sites at once, and must go through the Rule 8.4 gate judged on recall
at known landslide points — not be dropped straight into the live engine.
