import React, { useEffect, useMemo, useState } from "react";
import DeckGL from "@deck.gl/react";
import { ColumnLayer } from "@deck.gl/layers";
import {
  DEFAULT_SATELLITE_INPUTS,
  FALLBACK_BASINS,
  buildInteractiveBasinGrid,
  computeLocalSatelliteSynthesis,
  synthesizeSatellitePhysics,
} from "./services/api";
import {
  BasinMetadata,
  GridFeature,
  SatelliteSensorInputs,
  SynthesizedPhysicsResponse,
} from "./types/hazard";

type LayerMode = "calibrated_fs" | "raw_fs" | "differential_anomaly" | "soil_depth";

const REGIME_BADGES: Record<string, { label: string; style: string }> = {
  STABLE: {
    label: "GREEN • STABLE (FS ≥ 1.25)",
    style: "bg-emerald-500/20 text-emerald-300 border-emerald-500/50",
  },
  MARGINAL_WATCH: {
    label: "YELLOW/ORANGE • WATCH (1.0 ≤ FS < 1.25)",
    style: "bg-amber-500/20 text-amber-300 border-amber-500/50",
  },
  CRITICAL_FAILURE: {
    label: "RED • CRITICAL EVACUATION (FS < 1.0)",
    style: "bg-red-600/30 text-red-200 border-red-500 animate-pulse",
  },
};

export default function App() {
  const [selectedBasin, setSelectedBasin] = useState<BasinMetadata>(FALLBACK_BASINS[0]);
  const [activeTab, setActiveTab] = useState<"soil_depth" | "hydro_cryo">("soil_depth");
  const [inputs, setInputs] = useState<SatelliteSensorInputs>(DEFAULT_SATELLITE_INPUTS);
  const [synthesis, setSynthesis] = useState<SynthesizedPhysicsResponse>(() =>
    computeLocalSatelliteSynthesis(DEFAULT_SATELLITE_INPUTS)
  );
  const [layerMode, setLayerMode] = useState<LayerMode>("calibrated_fs");

  useEffect(() => {
    let cancelled = false;
    synthesizeSatellitePhysics(inputs).then((res) => {
      if (!cancelled) setSynthesis(res);
    });
    return () => {
      cancelled = true;
    };
  }, [inputs]);

  const gridFeatures = useMemo(
    () => buildInteractiveBasinGrid(selectedBasin, inputs),
    [selectedBasin, inputs]
  );

  const updateField = <K extends keyof SatelliteSensorInputs>(
    key: K,
    val: SatelliteSensorInputs[K]
  ) => {
    setInputs((prev) => ({ ...prev, [key]: val }));
  };

  const applyPreset = (preset: "forested_safe" | "langtang_2015" | "monsoon_surge") => {
    if (preset === "forested_safe") {
      setSelectedBasin(FALLBACK_BASINS[0]);
      setInputs({
        ...DEFAULT_SATELLITE_INPUTS,
        latitude: 28.18,
        longitude: 85.32,
        elevation_m: 2850,
        slope_deg: 28,
        aspect_deg: 170,
        profile_curvature: -0.2,
        sentinel2_ndvi: 0.72,
        gedi_canopy_height_m: 22.0,
        diurnal_lst_range_c: 19.0,
        sar_l_to_c_band_penetration_ratio: 1.35,
        distance_to_fault_km: 11.0,
        imerg_rain_72h_mm: 35,
        antecedent_rain_30d_api_mm: 25,
        pdd_c_days_7d: 2.0,
        snow_water_equivalent_mm: 0,
        bedrock_pga_g: 0.0,
      });
    } else if (preset === "langtang_2015") {
      setSelectedBasin(FALLBACK_BASINS[0]);
      setInputs({
        ...DEFAULT_SATELLITE_INPUTS,
        latitude: 28.232,
        longitude: 85.518,
        elevation_m: 5680,
        slope_deg: 46,
        aspect_deg: 195,
        profile_curvature: -0.62,
        sentinel2_ndvi: 0.02,
        gedi_canopy_height_m: 0.0,
        diurnal_lst_range_c: 9.0,
        sar_l_to_c_band_penetration_ratio: 1.05,
        distance_to_fault_km: 0.8,
        imerg_rain_72h_mm: 40,
        antecedent_rain_30d_api_mm: 30,
        pdd_c_days_7d: 14.0,
        diurnal_freeze_thaw_cycles_30d: 26,
        snow_water_equivalent_mm: 340,
        bedrock_pga_g: 0.26,
        insar_los_velocity_mm_month: 24.0,
      });
    } else {
      setSelectedBasin(FALLBACK_BASINS[1]);
      setInputs({
        ...DEFAULT_SATELLITE_INPUTS,
        latitude: 28.596,
        longitude: 84.636,
        elevation_m: 4620,
        slope_deg: 38.5,
        aspect_deg: 185,
        profile_curvature: -0.85,
        plan_curvature: -0.65,
        sentinel2_ndvi: 0.18,
        gedi_canopy_height_m: 2.5,
        diurnal_lst_range_c: 22.0,
        sar_l_to_c_band_penetration_ratio: 1.45,
        distance_to_fault_km: 1.5,
        imerg_rain_72h_mm: 235,
        antecedent_rain_30d_api_mm: 210,
        pdd_c_days_7d: 15.5,
        snow_water_equivalent_mm: 90,
        bedrock_pga_g: 0.04,
      });
    }
  };

  const syn = synthesis.synthesized_variables;
  const diff = synthesis.differential_calibration;
  const badge = REGIME_BADGES[synthesis.stability_regime] || REGIME_BADGES.STABLE;

  const layers = [
    new ColumnLayer<GridFeature>({
      id: "susceptibility-30m-columns",
      data: gridFeatures,
      diskResolution: 12,
      radius: 460,
      extruded: true,
      pickable: true,
      elevationScale: 1,
      getPosition: (d) => [d.geometry.coordinates[0], d.geometry.coordinates[1]],
      getElevation: (d) => {
        if (layerMode === "soil_depth") return d.properties.soil_depth_z_m * 450;
        if (layerMode === "differential_anomaly")
          return Math.max(200, Math.abs(d.properties.differential_anomaly) * 2800);
        return d.properties.elevation_m * 0.4;
      },
      getFillColor: (d) => {
        if (layerMode === "soil_depth") {
          const ratio = Math.min(1, d.properties.soil_depth_z_m / 4.0);
          return [Math.round(56 + 180 * ratio), Math.round(189 - 60 * ratio), 248, 215];
        }
        if (layerMode === "differential_anomaly") {
          // Purple/Magenta for positive differential anomaly (real failure > pure physics), Cyan for negative
          return d.properties.data_coverage_mode === "DIRECT_REAL_DATA_BLOCK_CALIBRATION"
            ? [217, 70, 239, 230]
            : [56, 189, 248, 200];
        }
        const fsVal = layerMode === "raw_fs" ? d.properties.fs : d.properties.calibrated_fs;
        return fsVal < 1.0
          ? [239, 68, 68, 225]
          : fsVal < 1.25
          ? [245, 158, 11, 205]
          : [16, 185, 129, 185];
      },
      updateTriggers: {
        getElevation: [layerMode, gridFeatures],
        getFillColor: [layerMode, gridFeatures],
      },
    }),
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col lg:flex-row">
      {/* Left Interactive Satellite Synthesis & Differential Calibration Sidebar */}
      <aside className="w-full lg:w-[500px] border-r border-slate-800 p-5 flex flex-col gap-3.5 bg-slate-900/90 overflow-y-auto max-h-screen">
        <div>
          <span className="text-[11px] uppercase tracking-widest text-sky-400 font-semibold">
            Himalayan-MHEWS • Hybrid Real + Physics + Differential Engine
          </span>
          <h1 className="text-lg font-bold mt-0.5">
            Satellite Soil-Depth &amp; Differential Calibration Lab
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Fuses 18 satellite signals with 973 real Nepal landslide &amp; &gt;4500m cryosphere avalanche/GLOF anchor points.
          </p>
        </div>

        {/* Historical & Field Scenario Presets */}
        <div className="grid grid-cols-3 gap-1.5 text-xs">
          <button
            onClick={() => applyPreset("forested_safe")}
            className="p-2 rounded-lg border border-emerald-500/40 bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-200 font-medium text-left"
          >
            🌲 Forested Pine Slope
          </button>
          <button
            onClick={() => applyPreset("langtang_2015")}
            className="p-2 rounded-lg border border-red-500/40 bg-red-500/10 hover:bg-red-500/20 text-red-200 font-medium text-left"
          >
            🏔️ Langtang 2015 (5680m)
          </button>
          <button
            onClick={() => applyPreset("monsoon_surge")}
            className="p-2 rounded-lg border border-amber-500/40 bg-amber-500/10 hover:bg-amber-500/20 text-amber-200 font-medium text-left"
          >
            🌊 Birendra Tal / Manaslu
          </button>
        </div>

        {/* Control Category Tabs */}
        <div className="grid grid-cols-2 gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs">
          <button
            onClick={() => setActiveTab("soil_depth")}
            className={`py-1.5 px-2 rounded-md font-semibold transition ${
              activeTab === "soil_depth"
                ? "bg-sky-500/20 text-sky-300 border border-sky-500/40"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            1. Soil Depth (z) Satellites
          </button>
          <button
            onClick={() => setActiveTab("hydro_cryo")}
            className={`py-1.5 px-2 rounded-md font-semibold transition ${
              activeTab === "hydro_cryo"
                ? "bg-sky-500/20 text-sky-300 border border-sky-500/40"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            2. Rain, Ice-Glue &amp; Quakes
          </button>
        </div>

        {/* Sliders */}
        {activeTab === "soil_depth" ? (
          <div className="space-y-2.5 bg-slate-950/70 p-3.5 rounded-xl border border-slate-800 text-xs">
            <div>
              <div className="flex justify-between mb-0.5">
                <span className="text-slate-300">1. DEM Slope Angle (β)</span>
                <span className="font-mono text-sky-300">{inputs.slope_deg}°</span>
              </div>
              <input
                type="range"
                min={8}
                max={62}
                step={0.5}
                value={inputs.slope_deg}
                onChange={(e) => updateField("slope_deg", parseFloat(e.target.value))}
                className="w-full accent-sky-500"
              />
            </div>

            <div>
              <div className="flex justify-between mb-0.5">
                <span className="text-slate-300">2. Astronomical Solar Aspect (180°=South Sun)</span>
                <span className="font-mono text-sky-300">{inputs.aspect_deg}°</span>
              </div>
              <input
                type="range"
                min={0}
                max={360}
                step={5}
                value={inputs.aspect_deg}
                onChange={(e) => updateField("aspect_deg", parseFloat(e.target.value))}
                className="w-full accent-sky-500"
              />
            </div>

            <div>
              <div className="flex justify-between mb-0.5">
                <span className="text-slate-300">3. MODIS/ECOSTRESS Day-Night ΔT (Thermal Inertia)</span>
                <span className="font-mono text-sky-300">{inputs.diurnal_lst_range_c}°C</span>
              </div>
              <input
                type="range"
                min={4}
                max={30}
                step={0.5}
                value={inputs.diurnal_lst_range_c}
                onChange={(e) => updateField("diurnal_lst_range_c", parseFloat(e.target.value))}
                className="w-full accent-sky-500"
              />
            </div>

            <div>
              <div className="flex justify-between mb-0.5">
                <span className="text-slate-300">4. DEM Bowl Curvature (Negative = Gully Traps Dirt)</span>
                <span className="font-mono text-sky-300">{inputs.profile_curvature}</span>
              </div>
              <input
                type="range"
                min={-1.5}
                max={1.5}
                step={0.05}
                value={inputs.profile_curvature}
                onChange={(e) => updateField("profile_curvature", parseFloat(e.target.value))}
                className="w-full accent-sky-500"
              />
            </div>

            <div>
              <div className="flex justify-between mb-0.5">
                <span className="text-slate-300">5. NASA GEDI LiDAR Canopy Height (Root Floor)</span>
                <span className="font-mono text-sky-300">{inputs.gedi_canopy_height_m} m</span>
              </div>
              <input
                type="range"
                min={0}
                max={35}
                step={0.5}
                value={inputs.gedi_canopy_height_m}
                onChange={(e) => updateField("gedi_canopy_height_m", parseFloat(e.target.value))}
                className="w-full accent-sky-500"
              />
            </div>

            <div>
              <div className="flex justify-between mb-0.5">
                <span className="text-slate-300">6. NISAR L-Band / S1 C-Band Penetration Ratio</span>
                <span className="font-mono text-sky-300">
                  {inputs.sar_l_to_c_band_penetration_ratio.toFixed(2)}x
                </span>
              </div>
              <input
                type="range"
                min={0.6}
                max={1.9}
                step={0.05}
                value={inputs.sar_l_to_c_band_penetration_ratio}
                onChange={(e) =>
                  updateField("sar_l_to_c_band_penetration_ratio", parseFloat(e.target.value))
                }
                className="w-full accent-sky-500"
              />
            </div>
          </div>
        ) : (
          <div className="space-y-2.5 bg-slate-950/70 p-3.5 rounded-xl border border-slate-800 text-xs">
            <div>
              <div className="flex justify-between mb-0.5">
                <span className="text-slate-300">1. Elevation (&gt;4500m Permafrost + Cryosphere Anchors)</span>
                <span className="font-mono text-sky-300">{inputs.elevation_m} m</span>
              </div>
              <input
                type="range"
                min={1500}
                max={6800}
                step={50}
                value={inputs.elevation_m}
                onChange={(e) => updateField("elevation_m", parseFloat(e.target.value))}
                className="w-full accent-sky-500"
              />
            </div>

            <div>
              <div className="flex justify-between mb-0.5">
                <span className="text-slate-300">2. Warm Melt Index (PDD °C-days)</span>
                <span className="font-mono text-sky-300">{inputs.pdd_c_days_7d} °C-days</span>
              </div>
              <input
                type="range"
                min={0}
                max={25}
                step={0.5}
                value={inputs.pdd_c_days_7d}
                onChange={(e) => updateField("pdd_c_days_7d", parseFloat(e.target.value))}
                className="w-full accent-sky-500"
              />
            </div>

            <div>
              <div className="flex justify-between mb-0.5">
                <span className="text-slate-300">3. NASA IMERG 72h Rain + 30d Sponge API (mm)</span>
                <span className="font-mono text-sky-300">
                  {inputs.imerg_rain_72h_mm} + {inputs.antecedent_rain_30d_api_mm} mm
                </span>
              </div>
              <input
                type="range"
                min={0}
                max={280}
                step={5}
                value={inputs.imerg_rain_72h_mm}
                onChange={(e) => updateField("imerg_rain_72h_mm", parseFloat(e.target.value))}
                className="w-full accent-sky-500"
              />
            </div>

            <div>
              <div className="flex justify-between mb-0.5">
                <span className="text-slate-300">4. Distance to Active Fault (km)</span>
                <span className="font-mono text-sky-300">{inputs.distance_to_fault_km} km</span>
              </div>
              <input
                type="range"
                min={0.1}
                max={15}
                step={0.2}
                value={inputs.distance_to_fault_km}
                onChange={(e) => updateField("distance_to_fault_km", parseFloat(e.target.value))}
                className="w-full accent-sky-500"
              />
            </div>

            <div>
              <div className="flex justify-between mb-0.5">
                <span className="text-slate-300">5. ICESat-2 / MODIS Snowpack Load (SWE mm)</span>
                <span className="font-mono text-sky-300">{inputs.snow_water_equivalent_mm} mm</span>
              </div>
              <input
                type="range"
                min={0}
                max={800}
                step={20}
                value={inputs.snow_water_equivalent_mm}
                onChange={(e) =>
                  updateField("snow_water_equivalent_mm", parseFloat(e.target.value))
                }
                className="w-full accent-sky-500"
              />
            </div>

            <div>
              <div className="flex justify-between mb-0.5">
                <span className="text-slate-300">6. USGS Bedrock Earthquake Shaking (PGA / g)</span>
                <span className="font-mono text-sky-300">{inputs.bedrock_pga_g} g</span>
              </div>
              <input
                type="range"
                min={0}
                max={0.4}
                step={0.01}
                value={inputs.bedrock_pga_g}
                onChange={(e) => updateField("bedrock_pga_g", parseFloat(e.target.value))}
                className="w-full accent-sky-500"
              />
            </div>
          </div>
        )}

        {/* Live Differential Calibration & Raw vs. Calibrated FS Readout */}
        <div className="space-y-2.5 bg-slate-950/95 p-3.5 rounded-xl border border-slate-800">
          <div className="flex items-center justify-between">
            <span className="text-[11px] uppercase text-slate-400 font-semibold">
              Calibrated Stability Regime
            </span>
            <span className={`px-2.5 py-0.5 rounded-full text-[11px] font-bold border ${badge.style}`}>
              {badge.label}
            </span>
          </div>

          {/* Raw Physics FS vs Differential-Calibrated FS */}
          <div className="grid grid-cols-4 gap-2 text-center">
            <div className="bg-slate-900 p-2 rounded-lg border border-slate-800">
              <div className="text-[10px] text-slate-400">Raw Physics FS</div>
              <div className="text-base font-mono font-bold text-slate-300">
                {synthesis.deterministic_fs.toFixed(2)}
              </div>
            </div>
            <div className="bg-slate-900 p-2 rounded-lg border border-fuchsia-500/40">
              <div className="text-[10px] text-fuchsia-300">Calibrated FS</div>
              <div
                className={`text-base font-mono font-bold ${
                  synthesis.calibrated_fs < 1.0
                    ? "text-red-400"
                    : synthesis.calibrated_fs < 1.25
                    ? "text-amber-300"
                    : "text-emerald-400"
                }`}
              >
                {synthesis.calibrated_fs.toFixed(2)}
              </div>
            </div>
            <div className="bg-slate-900 p-2 rounded-lg border border-slate-800">
              <div className="text-[10px] text-slate-400">Calibrated z</div>
              <div className="text-base font-mono font-bold text-sky-300">
                {diff.calibrated_z_m}m
              </div>
            </div>
            <div className="bg-slate-900 p-2 rounded-lg border border-slate-800">
              <div className="text-[10px] text-slate-400">Hybrid ML Risk</div>
              <div className="text-base font-mono font-bold text-amber-300">
                {(synthesis.hybrid_susceptibility_score * 100).toFixed(1)}%
              </div>
            </div>
          </div>

          {/* Differential Block Calibration Mode & Offset Details */}
          <div className="p-2.5 rounded-lg bg-slate-900/90 border border-slate-800 text-[11px] space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Block Calibration Mode:</span>
              <span
                className={`font-mono font-semibold px-1.5 py-0.5 rounded ${
                  diff.data_coverage_mode === "DIRECT_REAL_DATA_BLOCK_CALIBRATION"
                    ? "bg-fuchsia-500/20 text-fuchsia-300 border border-fuchsia-500/40"
                    : "bg-sky-500/20 text-sky-300 border border-sky-500/40"
                }`}
              >
                {diff.data_coverage_mode === "DIRECT_REAL_DATA_BLOCK_CALIBRATION"
                  ? `DIRECT REAL BLOCK (${diff.supporting_real_samples} pts)`
                  : "NEIGHBOR-BLOCK GRANULATED"}
              </span>
            </div>
            <div className="flex justify-between text-slate-300">
              <span>Differential Anomaly (Δ) &amp; FS Offset:</span>
              <span className="font-mono text-fuchsia-300">
                Δ = {diff.block_differential_anomaly >= 0 ? "+" : ""}
                {diff.block_differential_anomaly} → ΔFS = {diff.delta_fs_offset}
              </span>
            </div>
            <div className="flex justify-between text-slate-300">
              <span>Glue (Raw → Calibrated):</span>
              <span className="font-mono">
                {syn.c_prime_kpa} → {diff.calibrated_c_prime_kpa} kPa (+Suction {syn.matric_suction_kpa})
              </span>
            </div>
          </div>
        </div>
      </aside>

      {/* Right 3D Deck.gl Watershed Viewport */}
      <main className="flex-1 relative min-h-[540px]">
        <DeckGL
          initialViewState={{
            longitude: selectedBasin.center[0],
            latitude: selectedBasin.center[1],
            zoom: 10.2,
            pitch: 52,
            bearing: -15,
          }}
          controller={true}
          layers={layers}
        />

        {/* Top Map Overlay Controls: Basin Selector + 4-Mode 3D Layer Switcher */}
        <div className="absolute top-4 left-4 right-4 flex flex-wrap gap-2 justify-between pointer-events-none">
          <div className="flex gap-1.5 bg-slate-900/95 border border-slate-800 p-1.5 rounded-xl text-xs pointer-events-auto">
            {FALLBACK_BASINS.map((b) => (
              <button
                key={b.id}
                onClick={() => setSelectedBasin(b)}
                className={`px-3 py-1.5 rounded-lg font-semibold transition ${
                  selectedBasin.id === b.id
                    ? "bg-sky-500 text-slate-950"
                    : "text-slate-300 hover:bg-slate-800"
                }`}
              >
                {b.name}
              </button>
            ))}
          </div>

          <div className="flex gap-1 bg-slate-900/95 border border-slate-800 p-1.5 rounded-xl text-xs pointer-events-auto">
            {(
              [
                ["calibrated_fs", "Calibrated FS (Real+Physics)"],
                ["raw_fs", "Raw Physics FS"],
                ["differential_anomaly", "Differential Anomaly (Δ)"],
                ["soil_depth", "Soil Depth (z)"],
              ] as [LayerMode, string][]
            ).map(([mode, label]) => (
              <button
                key={mode}
                onClick={() => setLayerMode(mode)}
                className={`px-2.5 py-1.5 rounded-lg font-semibold transition ${
                  layerMode === mode
                    ? "bg-fuchsia-500/30 text-fuchsia-200 border border-fuchsia-500/50"
                    : "text-slate-300 hover:bg-slate-800"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* Bottom Legend & Hotspots */}
        <div className="absolute bottom-5 right-5 bg-slate-900/95 border border-slate-800 p-3.5 rounded-xl text-xs space-y-1.5 max-w-xs">
          <div className="font-semibold text-slate-200">{selectedBasin.name} Anchors</div>
          {selectedBasin.key_hotspots.map((spot) => (
            <div key={spot} className="text-slate-400">
              • {spot}
            </div>
          ))}
          <div className="pt-1 border-t border-slate-800 text-[11px] text-fuchsia-300">
            Purple Δ columns = Direct Real-Data Calibrated Block • Cyan = Neighbor-Granulated Block
          </div>
        </div>
      </main>
    </div>
  );
}
