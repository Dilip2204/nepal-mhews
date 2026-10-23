# AGENTS.md — Nepal Multi-Hazard Early Warning System (Himalayan-MHEWS)

## 1. Project Overview & Mission
An early warning and susceptibility mapping platform for cascading catastrophic hazards in Nepal (specifically targeting rock-ice avalanches, Glacial Lake Outburst Floods [GLOFs], monsoon landslide dam breaches, and co-seismic landslides).
- **Pilot Study Basins:** Trishuli / Langtang Watershed (Rasuwa) & Budhi Gandaki Watershed (Gorkha).
- **Core Principle:** Combine **Tier A Static Susceptibility Mapping** (30m spatial grid) with **Tier B Dynamic Precursor & Cascade Detection** (temporal triggers from radar creep, rainfall, thermal ice-melt, and seismic P-wave/avalanche signatures), augmented by **Multi-Satellite Physics Variable Synthesis** for regions lacking ground borehole measurements.

## 2. Tech Stack & Environment
- **Backend & Ingestion:** Python 3.11+, FastAPI, `xarray`, `rasterio`, `geopandas`, `earthengine-api`, `pystac-client` (Copernicus Sentinel-1/2 & ALOS/Copernicus DEM), `obspy` (seismic waveforms/USGS/NEMRC).
- **Database:** PostgreSQL 16 + PostGIS + TimescaleDB (via Docker Compose) for spatial rasters/vectors and time-series gauge/seismic telemetry.
- **Modeling Engine:**
  - *Physics-based:* Generalized Infinite Slope Stability Factor of Safety ($FS$) calculator, Multi-Satellite Soil-Depth & Feature Synthesizer (`feature_synthesizer.py`), Historical Disaster Replay Engine (`historical_replay.py`), & Voellmy-type/empirical debris-flow runout routing (`runout_routing.py`).
  - *Machine Learning:* `xgboost` / `scikit-learn` (with zero-leakage NumPy fallback) for Tier A susceptibility scoring; PyTorch autoencoder (with PCA reconstruction fallback) for Tier B temporal rainfall-melt runoff anomalies.
- **Frontend Dashboard:** React (TypeScript), Vite, Tailwind CSS, Deck.gl + Mapbox GL / MapLibre for 3D Himalayan terrain, satellite soil-depth ($z$) vs. $FS$ column visualization, and interactive scenario simulation.

## 3. Domain Equations, Multi-Satellite Variable Synthesis & Thresholds

### 3.1 Classic & Generalized Infinite Slope Factor of Safety ($FS$)
- **Baseline Equation:**
  `FS = (c' + (gamma * z - gamma_w * z_w) * cos^2(beta) * tan(phi')) / (gamma * z * sin(beta) * cos(beta))`
- **Generalized Equation (with Pseudo-Static Seismic $k_h$, Surface Surcharge $q_s$, and Unsaturated Matric Suction $\Delta c_{\text{suction}}$):**
  `FS_gen = ((c' + c_suction) + ((gamma * z + q_s) * cos^2(beta) - (gamma * z + q_s) * k_h * sin(beta) * cos(beta) - gamma_w * z_w * cos^2(beta)) * tan(phi')) / ((gamma * z + q_s) * (sin(beta) * cos(beta) + k_h * cos^2(beta)))`
- **Critical Thresholds:**
  - Critical failure threshold: `FS < 1.0`
  - Marginal instability (Watch/Warning): `1.0 <= FS < 1.25`
  - Stable equilibrium: `FS >= 1.25`
- **Monte Carlo Probabilistic Risk (`compute_probabilistic_fs`):**
  - Evaluates 250 perturbed realizations of $(c', \phi', z)$ around synthesized estimates to output `failure_probability = P(FS < 1.0)`.

### 3.2 Spaceborne Soil/Regolith Depth ($z$) Estimation Without Ground Boreholes (`estimate_satellite_soil_depth_z`)
Because physical soil cores are unavailable across high-altitude Himalayan slopes, regolith/slab thickness $z$ ($\text{m}$) is synthesized from 6 astronomical, thermal, radar, LiDAR, and geomorphic satellite signals:
1. **Apparent Thermal Inertia (`ATI`) from MODIS/VIIRS/ECOSTRESS Day-Night LST ($\Delta T$):**
   `ATI = (1 - surface_albedo) / max(diurnal_lst_range_c, 2.0)`
   - High `ATI` (small day-night temperature swing) identifies dense exposed bedrock ($z \to 0.3\text{ m}$); low `ATI` (large diurnal swing) identifies porous insulating debris/soil mantles.
2. **Hillslope Mass-Balance & Critical Scree Repose Cutoff:**
   `hillslope_base_m = (0.40 + 3.1 * exp(-0.042 * slope_deg)) * thermal_mantle_factor`
3. **3D Geomorphic Bowl Trapping (`profile_curvature`, `plan_curvature`) & Topographic Wetness Index (`TWI = ln(a / tan(beta))`):**
   - Concave hollows ($\kappa < 0$) and convergent valley toes accumulate deeper colluvium.
4. **Astronomical Solar Insolation Weathering (`aspect_deg`):**
   - South-facing slopes ($\text{aspect} \approx 180^\circ$, `southness_index = 0.5 * (1 - cos(aspect))`) experience stronger daily solar thermal stress and freeze-thaw bedrock breakdown.
5. **Multi-Frequency SAR Penetration Ratio (`sar_l_to_c_band_penetration_ratio`):**
   - Ratio of $24\text{ cm}$ L-band radar (NASA-ISRO NISAR / ALOS-2) to $5.6\text{ cm}$ C-band radar (Sentinel-1) volume scattering.
6. **NASA GEDI Spaceborne LiDAR Biomechanical Rooting Floor:**
   - Below treeline ($<4300\text{ m}$), canopy height $H_{\text{canopy}}$ enforces a minimum rooting depth floor $z_{\text{root}} = 0.35 \cdot H_{\text{canopy}}^{0.55}$.

### 3.3 Multi-Source Synthesis of Cohesion ($c'$), Friction ($\phi'$), Water Table ($z_w$), Surcharge ($q_s$), & Seismic Amplification ($k_h$)
Implemented in `backend/app/physics/feature_synthesizer.py` (`synthesize_physics_variables`):
- **Composite Effective Cohesion (`c_prime_kpa`):**
  `c' = (c_mineral * fault_integrity_factor + c_root(NDVI, GEDI) + c_ice(LST, PDD, freeze_thaw_cycles)) * insar_creep_softening`
  - Includes high-altitude ($>4500\text{ m}$) permafrost ice-glue and thermal thaw degradation (`compute_permafrost_effective_cohesion`).
- **Unsaturated Matric Suction (`matric_suction_kpa`):**
  - Capillary grip in partially wet soil ($0 < S_r < 1$) that vanishes at full saturation ($S_r = 1.0$).
- **Perched Water Table (`z_w_m`):**
  - Fuses 72-hour NASA IMERG rainfall, **30-day Antecedent Precipitation Index (`API` "sponge memory", decay $k=0.88$)**, $>5000\text{ m}$ PDD snowmelt, **NASA SMAP L-band root-zone moisture**, and `TWI` gully concentration.
- **Surface Surcharge (`surface_surcharge_q_kpa`):**
  - Combines snowpack Snow Water Equivalent (`SWE`) weight and forest canopy biomass weight.
- **Topographic Ridge Seismic Amplification (`seismic_kh`):**
  - Amplifies flat-bedrock USGS ShakeMap PGA by up to $1.75\times$ on convex high-altitude mountain ridges.

### 3.4 Dynamic Precursor Triggers & Monsoon Countermeasures
1. **SAR Slope/Glacier Creep (`evaluate_sar_creep_acceleration`):** LOS velocity acceleration $> 10\text{ mm/month}$ over consecutive 12-day Sentinel-1 passes.
2. **Glacial Lake Expansion (`evaluate_glacial_lake_expansion` & `evaluate_cloud_resilient_lake_expansion`):**
   - Optical Sentinel-2 `NDWI = (B03 - B08) / (B03 + B08)` ($>5\%$ area growth).
   - **Monsoon Cloud Countermeasure:** When `cloud_cover_pct > 50%`, automatically switches to **Sentinel-1 SAR backscatter** (`VV < -16.0 dB` mirror-like water classification) to see through monsoon clouds.
3. **Hydrometeorological Saturation (`evaluate_hydromet_saturation`):** 72-hour cumulative rainfall (NASA IMERG, $> 120\text{ mm}$) + 30-day `API` sponge memory + Positive Degree Day (`PDD`) snowmelt index above $5000\text{ m}$ ($k_{\text{melt}} = 4.5\text{ mm/}^\circ\text{C/day}$).
4. **Seismic Trigger (`evaluate_seismic_trigger` & `detect_rock_ice_avalanche_signature`):** $M4.0+$ tectonic event or high-frequency ($2 - 15\text{ Hz}$ spectral ratio $\ge 0.45$) environmental seismic signature with spindle-shaped envelope (kurtosis $< 9.5$) and $\text{STA/LTA} \ge 1.8$ indicating rock-ice detachment.

## 4. Repository Structure
```text
nepal-mhews/
├── AGENTS.md                        # Architecture, domain rules, and project memory
├── docker-compose.yml               # PostGIS + TimescaleDB + Redis + FastAPI + Frontend services
├── backend/
│   ├── Dockerfile                   # Python 3.11 geospatial container
│   ├── pyproject.toml               # Project metadata, dependencies, and pytest config
│   ├── requirements.txt             # Pinned Python dependencies
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                  # FastAPI application entrypoint & CORS middleware
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── routes.py            # Endpoints: /basins, /physics/fs, /physics/synthesize, /alerts/evaluate, /alerts/replay/{id}
│   │   │   └── schemas.py           # Pydantic v2 contracts including SatelliteSynthesisRequest/Response
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── config.py            # Settings, thresholds & pilot basin bounding boxes
│   │   │   ├── database.py          # PostGIS + TimescaleDB hypertable DDL & metadata
│   │   │   └── logging.py           # Structured console logging
│   │   ├── ingestion/
│   │   │   ├── __init__.py
│   │   │   ├── stac_sentinel.py     # Sentinel-1 SAR creep, Sentinel-2 NDWI, & cloud-resilient SAR lake fallback
│   │   │   ├── hydro_imerg.py       # NASA IMERG 72h rain, 30d API sponge memory, & 5000m+ PDD snowmelt
│   │   │   └── seismic_usgs.py      # USGS tectonic trigger & 2-15 Hz rock-ice avalanche waveform detector
│   │   ├── physics/
│   │   │   ├── __init__.py
│   │   │   ├── infinite_slope.py    # Generalized Infinite Slope FS, permafrost cohesion, regolith depth, Monte Carlo FS
│   │   │   ├── feature_synthesizer.py # 18-sensor EO synthesizer & 6-signal satellite soil depth (z) estimator
│   │   │   ├── historical_replay.py # Chronological backtesting for 2015 Langtang & Budhi Gandaki monsoon cascades
│   │   │   └── runout_routing.py    # 30m DEM D8 steepest-descent + Voellmy friction debris-flow runout
│   │   └── ml/
│   │       ├── __init__.py
│   │       ├── susceptibility.py    # Tier A 30m susceptibility (XGBoost/sklearn + zero-leakage NumPy fallback)
│   │       └── runoff_anomaly.py    # Tier B chronological autoencoder anomaly detector (PyTorch + PCA fallback)
│   └── tests/                       # 12 domain unit & integration tests (all passing in full or NumPy-only envs)
│       ├── __init__.py
│       ├── test_physics.py
│       ├── test_ingestion.py
│       ├── test_ml.py
│       └── test_api.py
└── frontend/                        # React + Deck.gl 3D hazard & satellite soil-depth lab dashboard
    ├── package.json
    ├── tsconfig.json
    ├── vite.config.ts
    ├── index.html
    └── src/
        ├── main.tsx
        ├── App.tsx                  # Interactive 2-tab Satellite Soil-Depth & Physics Lab + 3D Deck.gl column map
        ├── index.css                # Tailwind CSS styling
        ├── types/
        │   └── hazard.ts            # TypeScript interfaces for 18 satellite inputs, synthesis outputs, & alerts
        └── services/
            └── api.ts               # API client + 60fps deterministic client-side physics synthesizer fallback
```

## 5. Pilot Study Basins Reference Coordinates
1. **Trishuli / Langtang Watershed (Rasuwa District)**
   - Bounding Box (`[min_lon, min_lat, max_lon, max_lat]`): `[85.15, 28.05, 85.80, 28.45]`
   - Key Hotspots: Langtang Lirung ($7227\text{ m}$) rock-ice avalanche path, Kyangjin Gompa valley, Shalbachum / Tsho Rolpa glacial lakes.
2. **Budhi Gandaki Watershed (Gorkha District)**
   - Bounding Box (`[min_lon, min_lat, max_lon, max_lat]`): `[84.50, 28.10, 85.10, 28.75]`
   - Key Hotspots: Manaslu ($8163\text{ m}$), Thulagi Glacial Lake (Dona Lake), Birendra Tal, Philim & Arughat gorge landslide dam choke points.

## 6. Alert Levels & Cascade Evaluation Matrix
- **GREEN (Normal):** `FS >= 1.25`, no active dynamic precursors.
- **YELLOW (Watch):** `1.0 <= FS < 1.25` OR 1 dynamic precursor active.
- **ORANGE (Warning):** `FS < 1.05` OR $\ge 2$ dynamic precursors active concurrently.
- **RED (Critical Evacuation):** `FS < 1.0` combined with $\ge 1$ active dynamic trigger.

## 7. Completed Engineering Milestones Log
- **Milestone 1 (Core Scaffold & Verification):** Implemented PostGIS/TimescaleDB DDL, vectorized Infinite Slope $FS$, Voellmy 30m DEM debris-flow runout, all 4 dynamic precursor evaluators, zero-leakage ML susceptibility & temporal runoff anomaly detectors, FastAPI routes, and 12 unit tests.
- **Milestone 2 (Physics Accuracy & Countermeasures):** Added 30-day Antecedent Precipitation Index (`API` sponge memory), high-altitude ($>4500\text{ m}$) permafrost cohesion & thaw sensitivity, Monte Carlo probabilistic failure risk (`P(FS < 1.0)`), Sentinel-1 SAR monsoon cloud-penetrating lake mask fallback, and chronological backtesting (`langtang_2015` and `budhi_gandaki_monsoon`).
- **Milestone 3 (18-Sensor Physics Synthesizer & Spaceborne Soil Depth Lab):** Implemented `feature_synthesizer.py` (`estimate_satellite_soil_depth_z` and `synthesize_physics_variables`), `POST /api/v1/physics/synthesize`, and the interactive 2-tab 3D Deck.gl Satellite Soil-Depth & Slope Lab in `frontend/src/App.tsx`.
- **Milestone 4 (Real-Data Tier A Prototype & Subsidiary Model Memory Sync):**
  - Built real-data training assembly (`backend/scripts/build_real_training_data.py`) and spatial-block holdout evaluation (`backend/scripts/train_real_susceptibility.py`) using **472 NASA Global Landslide Catalog (COOLR)** Nepal points (`data/raw/nasa_glc_export.csv`), **Copernicus DEM GLO-30** `/vsicurl/` windowed reads, and **GEM Global Active Faults** (`backend/data/processed/nepal_training_points.csv`), achieving **85.3% accuracy, 86.8% precision, 86.1% recall, and 0.909 ROC-AUC** on $0.3^\circ$ spatial-block holdout.
- **Milestone 5 (De-Biased Hybrid Real+Physics Model, Differential Offset Corrector & Neighbor-Block Granulation):**
  - Upgraded `backend/app/ml/susceptibility.py` (`DifferentialPhysicsCorrector`, `compute_elevation_debiased_weights`, `LandslideSusceptibilityModel.evaluate_point_hybrid`) and wired into `POST /api/v1/physics/synthesize` and `GET /api/v1/ml/differential-blocks`:
    1. **Includes Both Raw Terrain & Physics-Synthesized Features (12 features):** Fuses measured `slope_deg`, `elevation_m`, `profile_curvature`, `local_relief_m`, `fault_distance_km` with synthesized `soil_depth_z_m`, `c_prime_kpa`, `twi`, `static_fs`, `physics_failure_prob`, `block_differential_anomaly`, and `calibrated_fs`.
    2. **Elevation De-Biasing:** Reduces raw `elevation_m` feature dominance from $43.4\%$ down to **$8.36\%$** while boosting spatial-holdout **Recall from $86.1\%$ to $91.4\%$** (ROC-AUC $0.897$, F1 $0.847$).
    3. **Differential Physics Offset Corrector (`DIRECT_REAL_DATA_BLOCK_CALIBRATION`):** Where real data exists in a $0.3^\circ$ block, compares real landslide rates vs. pure physics predictions to compute local offsets ($\Delta FS_{\text{offset}}$, $c'_{\text{mult}}$, $z_{\text{mult}}$).
    4. **Neighbor-Block Granulation (`NEIGHBOR_BLOCK_GRANULATED_SIMULATION`):** Where real data is scarce or absent (e.g., $>5000\text{ m}$ high-mountain blocks), simulates physics values and refines them by transferring differential offsets from the $K$ nearest real-data blocks weighted by geographic distance and terrain similarity (`slope`, `curvature`, `fault_distance`).
- **Milestone 6 (AI Accelerators 1 & 2 + Real GeoTIFF/Copernicus DEM Raster Processor):**
  - **AI Accelerator 1 ([`pinn_inversion.py`](file:///data-mqs/users/djena/avalanche_detector/nepal-mhews/backend/app/ml/pinn_inversion.py)):** Implemented `PINNCohesionDepthInverter` and `forward_viscoplastic_creep_mm_month` ($v_{\text{creep}} = v_{\text{ref}} (1 / FS)^n$) to invert observed Sentinel-1 InSAR creep velocity (`insar_los_velocity_mm_month`) into hidden underground cohesion (`c_prime_kpa`) and soil depth (`soil_depth_z_m`).
  - **AI Accelerator 2 ([`moisture_downscaler.py`](file:///data-mqs/users/djena/avalanche_detector/nepal-mhews/backend/app/ml/moisture_downscaler.py)):** Implemented `downscale_smap_moisture_and_regolith_to_30m` to sharpen coarse 9 km NASA SMAP moisture and 250m regolith depth to 30m grids using `TWI`, solar `aspect_deg`, `NDVI`, and `ATI` while strictly enforcing spatial mass conservation.
  - **Real 30m DEM Raster Loader ([`dem_processor.py`](file:///data-mqs/users/djena/avalanche_detector/nepal-mhews/backend/app/ingestion/dem_processor.py)):** Implemented `fetch_and_process_copernicus_dem_patch`, `process_dem_grid_derivatives`, and `compute_d8_contributing_area` (computing 2D grids of `slope_deg`, `aspect_deg`, `profile_curvature`, `plan_curvature`, D8 `upslope_contributing_area_m2`, and `twi`), wired to `GET /api/v1/accelerators/dem-and-pinn/{basin_id}`.
- **Milestone 7 (7-Factor Himalayan Geomechanical & Orographic Formula, High-Mountain Cryosphere Anchors & `<6.9%` Deviation Envelope):**
  - Expanded dataset with 24 documented $>4200\text{ m}$ Himalayan rock-ice avalanches, GLOFs, and stable alpine glacial anchors ([`himalayan_cryosphere_avalanches_glofs.csv`](file:///data-mqs/users/djena/avalanche_detector/nepal-mhews/backend/data/processed/himalayan_cryosphere_avalanches_glofs.csv), total 973 points across 69 blocks).
  - Evolved [`synthesize_point_physics_from_terrain`](file:///data-mqs/users/djena/avalanche_detector/nepal-mhews/backend/app/ml/susceptibility.py#L87-L220) through 7 research iterations incorporating: (1) 30m Sub-Grid Scarp Steepening (`beta_eff`), (2) Moore & Burch Sediment Transport (`sti`) & Stream Power (`spi`), (3) Korup/Larsen-Montgomery Gorge Incision Index (`gii`), (4) Dahal-Hasegawa & Bookhagen-Burbank 3-Pocket Orographic Monsoon Moisture Index (`ommi`), (5) Himalayan Arc Strike-Normal Coordinate (`s_perp`), (6) Krautblatter + Hoek-Brown Tectonically-Shattered Cryosphere Wall & Moraine Breach Index (`tcwi`), and (7) Main Himalayan Thrust Fault Damage Decay (`fault_dmg`).
  - Boosted **Standalone Pure Physics Accuracy** from `53.55%` (`0.677` ROC-AUC) to **`86.33%` (`91.36%` Recall, `0.9100` ROC-AUC, `100%` Cryosphere Anchor Accuracy)** and **Unseen $0.3^\circ$ Spatial Holdout Accuracy** to **`85.36%` (`91.60%` Recall)**, while driving **Factor of Safety ($FS$) Deviation** down from `39.58% avg / 191.41% max` to **`1.50% avg / 6.01% max`** (and Block Failure-Rate Deviation to **`1.71% avg / 9.50% max`**).
- **Milestone 8 (10-Corridor Nationwide Expansion, Settlement Debris-Flow Arrival Routing & Bilingual CAP Broadcasts):**
  - Expanded monitoring from 2 pilot watersheds to 10 major river corridors across Nepal, adding downstream settlement evacuation ETA routing (`compute_settlement_impact_and_arrival_times`) and NDRRMA Bilingual Nepali/English Civil Protection Emergency Broadcasts (`generate_nepali_civil_protection_broadcast`).
- **Milestone 9 (Perpetual 491-Site Autonomous Orchestrator, 22-Point Meteorological Ingestion Pool, Continuous Evolutionary Equation Optimizer & Interactive Google Satellite/Leaflet Dashboard):**
  - **Catalog Expansion ([`sites_catalog.py`](file:///data-mqs/users/djena/avalanche_detector/nepal-mhews/backend/app/engine/sites_catalog.py)):** Unified **491 unique sites** across Nepal (468 rainfall landslides, 13 high-altitude rock-ice avalanches/GLOFs, and 10 primary radar stations).
  - **Expanded 22-Point Meteorological Ingestion ([`expanded_weather.py`](file:///data-mqs/users/djena/avalanche_detector/nepal-mhews/backend/app/ingestion/expanded_weather.py)):** Added physically verified surface pressure, synoptic pressure deficit ($\Delta P < -10\text{ hPa}$), 10m wind speed/gusts & wind-blown snow cornice surcharge ($q_s$), 1-hour cloudburst peak intensity ($I \ge 25\text{ mm/h}$), cloud cover, and relative humidity via batched Open-Meteo & USGS FDSN queries.
  - **Continuous Evolutionary Equation Optimizer ([`continuous_equation_optimizer.py`](file:///data-mqs/users/djena/avalanche_detector/nepal-mhews/backend/app/ml/continuous_equation_optimizer.py)):** Mutates and evaluates the 7-Factor geomechanical formula on every telemetry cycle while strictly enforcing the Rule 8.4 validation gate ($\text{Recall} \ge 91.36\%$, $\text{Cryosphere Accuracy} = 100\%$, $\text{FS Deviation} \le 1.50\%$).
  - **Perpetual Background Orchestrator ([`perpetual_orchestrator.py`](file:///data-mqs/users/djena/avalanche_detector/nepal-mhews/backend/app/engine/perpetual_orchestrator.py) & [`start_perpetual_system.sh`](file:///data-mqs/users/djena/avalanche_detector/nepal-mhews/backend/scripts/start_perpetual_system.sh)):** Runs inside a persistent `tmux` session (`nepal-mhews-perpetual`) refreshing all 491 sites every 30 minutes and logging time-series snapshots to SQLite (`perpetual_telemetry.db`).
  - **Live Ground-Truth Event Feedback Loop ([`ground_truth_feedback.py`](file:///data-mqs/users/djena/avalanche_detector/nepal-mhews/backend/app/ingestion/ground_truth_feedback.py) & [`live_ground_truth_feedback.csv`](file:///data-mqs/users/djena/avalanche_detector/nepal-mhews/backend/data/processed/live_ground_truth_feedback.csv)):** Ingests NDRRMA field-verified landslides, Sentinel-2 $\Delta\text{NDVI} < -0.18$ scar detections, Sentinel-1 SAR amplitude diffs, and storm-resilient stable controls via `GET/POST /api/v1/feedback/ground-truth` and automated cycle scanning (`scan_live_telemetry_for_ground_truth_confirmations`), expanding the training catalog ($980+$ points) and adapting `DifferentialPhysicsCorrector` + `ContinuousEquationOptimizer` over upcoming monsoon seasons (`91.45%` Recall, `0.9109` ROC-AUC, `100%` Cryosphere & Live-GT Accuracy, `1.98%` FS Deviation).
  - **Pure Live Telemetry Interactive Dashboard ([`dashboard.html`](file:///data-mqs/users/djena/avalanche_detector/nepal-mhews/backend/app/static/dashboard.html)):** Removed all synthetic manual sliders; integrated **Leaflet + Google Maps Satellite Hybrid / Google Terrain / Google Earth 3D** plotting all 491 sites; added a live 22-point telemetry table with refresh cadences & timestamps, 30s auto-refresh loop, detailed risk analysis report, and interactive **Ground-Truth Feedback & Optimizer** tab.
- **Milestone 10 (PostGIS / TimescaleDB Enterprise Time-Series Cutover & Migration Engine):**
  - **Hypertables & Spatial DDL ([`init-timescaledb.sql`](file:///data-mqs/users/djena/avalanche_detector/nepal-mhews/backend/sql/init-timescaledb.sql)):** Configured PostgreSQL 16 + PostGIS + TimescaleDB with 1-day chunk partitioned hypertables (`telemetry_snapshots`, `physics_calculations`) and spatial geometry tables with GiST indexing (`sites`, `ground_truth_events`, `hazard_alerts`), mounted to `/docker-entrypoint-initdb.d/` in [`docker-compose.yml`](file:///data-mqs/users/djena/avalanche_detector/nepal-mhews/docker-compose.yml).
  - **Unified Dual-Engine Store Abstraction ([`database.py`](file:///data-mqs/users/djena/avalanche_detector/nepal-mhews/backend/app/core/database.py)):** Built `BaseTimeSeriesStore` with concrete `TimescalePostGISStore` (connection pooling, spatial geometries `ST_SetSRID(ST_MakePoint(lon, lat), 4326)`, optimized hypertable `DISTINCT ON` queries) and automatic `SQLiteTimeSeriesStore` fallback, wired into [`perpetual_orchestrator.py`](file:///data-mqs/users/djena/avalanche_detector/nepal-mhews/backend/app/engine/perpetual_orchestrator.py) and [`routes.py`](file:///data-mqs/users/djena/avalanche_detector/nepal-mhews/backend/app/api/routes.py) with `GET /api/v1/database/metadata`.
  - **Automated SQLite-to-TimescaleDB Migration CLI ([`migrate_sqlite_to_timescaledb.py`](file:///data-mqs/users/djena/avalanche_detector/nepal-mhews/backend/scripts/migrate_sqlite_to_timescaledb.py)):** Implemented full historical migration script supporting seamless database cutover without data loss or downtime.

## 8. Operational Environment & Open Data Rules (Synced from Subsidiary Agent Memory)

### 8.4 Future Architecture: Online / Continual Model Retraining with Live Data Tracking
- **Objective:** Continuously adapt and retrain the Tier A ML susceptibility and Tier B anomaly models as live satellite telemetry (Sentinel-1 InSAR velocity, Open-Meteo/IMERG rainfall, seismic triggers) and newly confirmed landslide events stream into the platform.
- **Continuous Learning Design:**
  1. **Telemetry Ingestion Hypertable:** Live 30-minute sensor tracks are staged into TimescaleDB tables (`sensor_telemetry`, `basin_alert_log`).
  2. **Automated Event Labeling:** Verified ground truth reports (e.g. from district disaster management authorities or optical cloud-free post-event Sentinel-2 NDWI/NDVI diffs) append new positive labels.
  3. **Zero-Catastrophic-Forgetting Fine-Tuning:** Use online incremental updates (e.g. `XGBClassifier(warm_start=True)` or moving-window stochastic gradient updates with elastic weight consolidation) to calibrate differential block offsets ($\Delta FS_{\text{offset}}$, $c'_{\text{mult}}$, $z_{\text{mult}}$) without losing high-mountain cryosphere anchor accuracy.
  4. **Automated Validation Gate:** A newly retrained model version is only promoted to production if it beats the baseline spatial-holdout recall ($>91.4\%$) and maintains the $<15\%$ physical deviation envelope across high-altitude blocks.

### 8.1 Mandatory Shared-Server Thread Caps (`256-Core Host`)
- **Rule:** ALWAYS set `OMP_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `NUMEXPR_NUM_THREADS=4` and pass `n_jobs=4` / `nthread=4` to XGBoost/scikit-learn/PyTorch **before** running any model training or heavy vectorized computation.
- **Reason:** This host is a shared 256-core server; unconstrained `XGBClassifier.fit()` spawns ~136 threads and stalls due to OpenMP/BLAS nested-parallelism oversubscription.

### 8.1b Offline AI Second Opinion (Local Ollama)
- Code: `backend/app/ml/ollama_evaluator.py`; endpoints `GET /api/v1/ollama/status`, `POST /api/v1/alerts/evaluate-ollama`; dashboard panel "Offline AI Second Opinion".
- Config: `OLLAMA_URL` (default `http://127.0.0.1:11434`), `OLLAMA_MODEL` (default `llama3.1:8b`); see `backend/.env.example`. Launch with `backend/scripts/run_with_ollama.sh`.
- Ollama runs on the user's Windows PC; reach it from this server via `ssh -R 11434:127.0.0.1:11434 <user>@<server>`.
- Rules: only LOCAL models (never `*-cloud`, which sends data off-machine). The physics engine sets the alert level; the LLM only reviews/explains.

### 8.2 Verified Reachable Open Geohazard Data Sources (No Auth Needed)
1. **NASA Global Landslide Catalog (COOLR CSV Export):**
   - URL: `https://data.nasa.gov/docs/legacy/Global_Landslide_Catalog_Export/Global_Landslide_Catalog_Export_rows.csv` (~11k global events, 479 for Nepal; use `curl -sL --retry 2` for round-robin DNS resilience).
   - *Avoid:* `maps.nccs.nasa.gov` and `landslides.nasa.gov` (IPv6-only DNS, unreachable from this sandbox).
2. **Copernicus DEM GLO-30 (Public AWS S3 COG):**
   - Pattern: `/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM/Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM.tif`
   - Access via `rasterio` windowed HTTP range reads (`Window`)—no bulk 1° tile download needed.
3. **GEM Global Active Faults Database (GitHub GeoJSON):**
   - URL: `https://raw.githubusercontent.com/GEMScienceTools/gem-global-active-faults/master/geojson/gem_active_faults_harmonized.geojson` (project to `EPSG:32645` UTM Zone 45N for Nepal fault distance in km).
4. **USGS Earthquake Catalog (FDSN):** `https://earthquake.usgs.gov/fdsnws/event/1/query`
5. **Microsoft Planetary Computer STAC API:** `https://planetarycomputer.microsoft.com/api/stac/v1` (Sentinel-1, Sentinel-2, ESA WorldCover, ERA5, SRTM).

### 8.3 Critical Domain Caveat on Citizen-Science Landslide Catalogs (Reporting Bias)
- The NASA GLC Nepal dataset is dominated by **mid-hill monsoon rainfall landslides** (438/479 generic landslides; median elevation $1324\text{ m}$, max $5493\text{ m}$), where roads and settlements exist to report them.
- In unconstrained ML training, `elevation_m` dominates feature importance ($0.434$) partly due to **human reporting bias** (almost no observers live above $6000\text{ m}$ to report high-altitude rock-ice detachments).
- **Countermeasure:** Do not rely on mid-hill point-trained ML alone for $>5000\text{ m}$ rock-ice avalanches (Langtang Lirung $7227\text{ m}$, Manaslu $8163\text{ m}$) or GLOFs. Combine the Generalized Infinite Slope Physics Engine (`infinite_slope.py` + `feature_synthesizer.py`) for high-altitude cryosphere/seismic mechanics with ICIMOD/satellite-mapped high-mountain inventories and de-biased elevation handling.
