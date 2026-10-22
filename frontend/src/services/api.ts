import {
  BasinMetadata,
  DataCoverageMode,
  GridFeature,
  SatelliteSensorInputs,
  SynthesizedPhysicsResponse,
} from "../types/hazard";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "/api/v1";

export const FALLBACK_BASINS: BasinMetadata[] = [
  {
    id: "langtang",
    name: "Trishuli / Langtang Watershed",
    district: "Rasuwa",
    bbox: [85.15, 28.05, 85.80, 28.45],
    center: [85.515, 28.215],
    max_elevation_m: 7227,
    key_hotspots: [
      "Langtang Lirung Rock-Ice Face (7227m)",
      "Kyangjin Gompa Debris Cone",
      "Shalbachum Glacial Lake",
    ],
  },
  {
    id: "budhi_gandaki",
    name: "Budhi Gandaki Watershed",
    district: "Gorkha",
    bbox: [84.50, 28.10, 85.10, 28.75],
    center: [84.63, 28.54],
    max_elevation_m: 8163,
    key_hotspots: [
      "Thulagi Glacial Lake (Dona Lake)",
      "Birendra Tal (Manaslu Glacier)",
      "Philim Gorge Landslide Dam Choke Point",
    ],
  },
];

export const DEFAULT_SATELLITE_INPUTS: SatelliteSensorInputs = {
  latitude: 28.232,
  longitude: 85.518,
  local_relief_m: 320,
  elevation_m: 4650,
  slope_deg: 36,
  aspect_deg: 195,
  profile_curvature: -0.35,
  plan_curvature: -0.25,
  upslope_contributing_area_m2: 4500,
  sentinel2_ndvi: 0.32,
  gedi_canopy_height_m: 6.5,
  sand_fraction: 0.48,
  clay_fraction: 0.16,
  coarse_rock_fragment_fraction: 0.35,
  dry_bulk_density_g_cm3: 1.55,
  distance_to_fault_km: 2.4,
  modis_lst_mean_c: -1.5,
  diurnal_lst_range_c: 16.0,
  surface_albedo: 0.22,
  diurnal_freeze_thaw_cycles_30d: 14,
  pdd_c_days_7d: 6.0,
  snow_water_equivalent_mm: 180,
  imerg_rain_72h_mm: 115,
  antecedent_rain_30d_api_mm: 95,
  smap_rootzone_moisture_vwc: 0.31,
  sar_l_to_c_band_penetration_ratio: 1.15,
  bedrock_pga_g: 0.08,
  insar_los_velocity_mm_month: 14.5,
};

/**
 * Reference 0.3° calibration blocks from NASA GLC + High-Mountain Avalanche/GLOF inventory
 */
const CALIBRATED_NEPAL_BLOCKS: Array<{
  latIdx: number;
  lonIdx: number;
  centerLat: number;
  centerLon: number;
  samples: number;
  anomaly: number;
}> = [
  { latIdx: 94, lonIdx: 285, centerLat: 28.23, centerLon: 85.54, samples: 6, anomaly: 0.038 }, // Langtang + Cryosphere Anchors
  { latIdx: 93, lonIdx: 285, centerLat: 28.04, centerLon: 85.55, samples: 9, anomaly: 0.052 }, // Melamchi / Rasuwa South
  { latIdx: 95, lonIdx: 282, centerLat: 28.56, centerLon: 84.61, samples: 5, anomaly: 0.034 }, // Manaslu / Birendra Tal
  { latIdx: 94, lonIdx: 282, centerLat: 28.31, centerLon: 84.72, samples: 11, anomaly: 0.058 }, // Budhi Gandaki Gorge
  { latIdx: 94, lonIdx: 279, centerLat: 28.25, centerLon: 83.85, samples: 23, anomaly: 0.068 }, // Annapurna / Kaski
];

export function computeLocalSatelliteSynthesis(
  raw: SatelliteSensorInputs
): SynthesizedPhysicsResponse {
  const slopeRad = (Math.min(Math.max(raw.slope_deg, 0.5), 85.0) * Math.PI) / 180.0;
  const specificArea = Math.max(raw.upslope_contributing_area_m2 / 30.0, 1.0);
  const twi = Math.log(specificArea / Math.max(Math.tan(slopeRad), 0.01));

  const aspectRad = (raw.aspect_deg * Math.PI) / 180.0;
  const southness = 0.5 * (1.0 - Math.cos(aspectRad));

  const ati =
    (1.0 - Math.min(Math.max(raw.surface_albedo, 0.05), 0.85)) /
    Math.max(raw.diurnal_lst_range_c, 2.0);
  const thermalMantleFactor = Math.min(Math.max(1.35 - 5.5 * ati, 0.35), 1.35);
  const hillslopeBase = (0.4 + 3.1 * Math.exp(-0.042 * raw.slope_deg)) * thermalMantleFactor;
  const curvatureAccum = Math.min(
    Math.max(-0.35 * raw.profile_curvature - 0.25 * raw.plan_curvature, -0.45),
    1.1
  );
  const twiDeposition = 0.08 * Math.max(twi - 4.0, 0.0);
  const solarWeathering = 0.22 * southness;
  const sarPenetration =
    0.25 * Math.min(Math.max(raw.sar_l_to_c_band_penetration_ratio - 1.0, -0.4), 0.8);
  const lidarRootFloor =
    raw.gedi_canopy_height_m > 1.0 ? 0.35 * Math.pow(raw.gedi_canopy_height_m, 0.55) : 0.3;

  const zRaw =
    hillslopeBase + curvatureAccum + twiDeposition + solarWeathering + sarPenetration;
  const z_m = Math.min(Math.max(Math.max(zRaw, lidarRootFloor), 0.3), 6.0);

  const c_mineral_kpa =
    3.5 + 14.0 * raw.clay_fraction + 4.0 * raw.coarse_rock_fragment_fraction;
  const c_root_kpa =
    raw.elevation_m < 4300 && raw.sentinel2_ndvi > 0.15
      ? 12.0 *
        Math.pow(raw.sentinel2_ndvi, 1.4) *
        Math.min(Math.max(raw.gedi_canopy_height_m / 15.0, 0.1), 1.5)
      : 0.0;

  let c_ice_kpa = 0.0;
  if (raw.elevation_m >= 4500) {
    const freezeStrength = Math.max(0.0, 18.0 - 1.15 * Math.max(0.0, raw.modis_lst_mean_c + 2.0));
    const thawLoss = 0.95 * raw.pdd_c_days_7d;
    const frostFatigue = 0.22 * raw.diurnal_freeze_thaw_cycles_30d;
    c_ice_kpa = Math.max(0.0, freezeStrength - thawLoss - frostFatigue);
  }

  const faultIntegrity = 1.0 - 0.4 * Math.exp(-raw.distance_to_fault_km / 2.5);
  const creepSoftening =
    1.0 / (1.0 + 0.018 * Math.max(0.0, raw.insar_los_velocity_mm_month - 5.0));
  const c_prime_kpa = Math.max(
    0.5,
    (c_mineral_kpa * faultIntegrity + c_root_kpa + c_ice_kpa) * creepSoftening
  );

  const basePhi =
    24.0 +
    12.0 * raw.sand_fraction +
    9.5 * raw.coarse_rock_fragment_fraction -
    14.0 * raw.clay_fraction;
  const phi_prime_deg = Math.min(
    Math.max(basePhi * (0.85 + 0.15 * faultIntegrity), 16.0),
    44.0
  );

  const porosity = Math.min(Math.max(1.0 - raw.dry_bulk_density_g_cm3 / 2.65, 0.18), 0.52);
  const snowmeltMm = raw.pdd_c_days_7d * 4.5 * (1.0 + 0.2 * southness);
  const effectiveWaterMm =
    raw.imerg_rain_72h_mm + 0.35 * raw.antecedent_rain_30d_api_mm + snowmeltMm;
  const twiConc = Math.min(Math.max(twi / 6.0, 0.6), 2.2);
  const infilZw = (effectiveWaterMm / 1000.0 / porosity) * twiConc;
  const smapZw = Math.min(
    Math.max((raw.smap_rootzone_moisture_vwc / porosity) * z_m * 0.6, 0.0),
    z_m
  );
  const z_w_m = Math.min(Math.max(Math.max(infilZw, smapZw), 0.0), z_m);
  const saturation_ratio = z_w_m / Math.max(z_m, 1e-6);

  const matric_suction_kpa =
    6.0 *
    saturation_ratio *
    Math.pow(1.0 - saturation_ratio, 1.5) *
    (1.0 + 2.0 * raw.clay_fraction);

  const gamma_kn_m3 = raw.dry_bulk_density_g_cm3 * 9.81 + porosity * saturation_ratio * 9.81;
  const snowLoad = (raw.snow_water_equivalent_mm / 1000.0) * 9.81;
  const canopyLoad = raw.elevation_m < 4300 ? 0.08 * raw.gedi_canopy_height_m : 0.0;
  const surface_surcharge_q_kpa = snowLoad + canopyLoad;

  const ridgeAmp = Math.min(
    Math.max(
      1.0 +
        0.25 * Math.max(0.0, raw.profile_curvature) +
        0.2 * Math.max(0.0, (raw.elevation_m - 2500.0) / 4000.0),
      1.0
    ),
    1.75
  );
  const seismic_kh = Math.min(Math.max(raw.bedrock_pga_g * ridgeAmp, 0.0), 0.65);

  const phiRad = (phi_prime_deg * Math.PI) / 180.0;
  const cosB = Math.cos(slopeRad);
  const sinB = Math.sin(slopeRad);
  const cos2B = cosB * cosB;
  const totalVert = gamma_kn_m3 * z_m + surface_surcharge_q_kpa;
  const normalStress = Math.max(
    0.0,
    (totalVert - 9.81 * z_w_m) * cos2B - totalVert * seismic_kh * sinB * cosB
  );
  const resisting = c_prime_kpa + matric_suction_kpa + normalStress * Math.tan(phiRad);
  const driving = Math.max(1e-9, totalVert * (sinB * cosB + seismic_kh * cos2B));
  const fs = resisting / driving;

  // Differential Block Calibration & Neighbor Granulation (<15% bounded empirical-Bayes envelope)
  const latIdx = Math.floor(raw.latitude / 0.3);
  const lonIdx = Math.floor(raw.longitude / 0.3);
  const direct = CALIBRATED_NEPAL_BLOCKS.find(
    (b) => b.latIdx === latIdx && b.lonIdx === lonIdx
  );

  let mode: DataCoverageMode = "NEIGHBOR_BLOCK_GRANULATED_SIMULATION";
  let samples = 0;
  let diffAnom = 0.052;
  if (direct && direct.samples >= 4) {
    mode = "DIRECT_REAL_DATA_BLOCK_CALIBRATION";
    samples = direct.samples;
    diffAnom = direct.anomaly + 0.0012 * (raw.slope_deg - 34.0);
  } else {
    // Distance-weighted interpolation from nearest calibrated blocks
    let wSum = 0;
    let aSum = 0;
    for (const b of CALIBRATED_NEPAL_BLOCKS) {
      const dist = Math.hypot(raw.latitude - b.centerLat, raw.longitude - b.centerLon);
      const w = 1.0 / Math.max(dist, 0.08);
      wSum += w;
      aSum += w * b.anomaly;
    }
    diffAnom = aSum / Math.max(wSum, 1e-6) - 0.012 * Math.sin(raw.latitude * 12.0);
  }

  diffAnom = Math.min(Math.max(diffAnom, -0.095), 0.095);
  const relFsShift = Math.min(Math.max(-0.72 * diffAnom, -0.085), 0.085);
  const calibrated_fs = Math.min(Math.max(fs * (1.0 + relFsShift), 0.25), 4.0);
  const deltaFsOffset = calibrated_fs - fs;
  const calibrated_c = c_prime_kpa * Math.min(Math.max(1.0 - 0.65 * diffAnom, 0.92), 1.08);
  const calibrated_z = z_m * Math.min(Math.max(1.0 + 0.45 * diffAnom, 0.94), 1.06);

  const hybridProb = Math.min(
    0.99,
    Math.max(0.01, 1.0 / (1.0 + Math.exp(4.8 * (calibrated_fs - 1.12))))
  );
  const failureProb = Math.min(
    100.0,
    Math.max(0.0, 100.0 / (1.0 + Math.exp(8.5 * (calibrated_fs - 1.0))))
  );

  return {
    synthesized_variables: {
      c_prime_kpa: Number(c_prime_kpa.toFixed(2)),
      c_mineral_kpa: Number(c_mineral_kpa.toFixed(2)),
      c_root_kpa: Number(c_root_kpa.toFixed(2)),
      c_ice_kpa: Number(c_ice_kpa.toFixed(2)),
      matric_suction_kpa: Number(matric_suction_kpa.toFixed(2)),
      phi_prime_deg: Number(phi_prime_deg.toFixed(1)),
      gamma_kn_m3: Number(gamma_kn_m3.toFixed(2)),
      z_m: Number(z_m.toFixed(2)),
      apparent_thermal_inertia: Number(ati.toFixed(4)),
      lidar_root_floor_m: Number(lidarRootFloor.toFixed(2)),
      z_w_m: Number(z_w_m.toFixed(2)),
      saturation_ratio: Number(saturation_ratio.toFixed(2)),
      surface_surcharge_q_kpa: Number(surface_surcharge_q_kpa.toFixed(2)),
      seismic_kh: Number(seismic_kh.toFixed(3)),
      ridge_amplification_factor: Number(ridgeAmp.toFixed(2)),
      twi: Number(twi.toFixed(2)),
      porosity: Number(porosity.toFixed(2)),
    },
    deterministic_fs: Number(fs.toFixed(3)),
    calibrated_fs: Number(calibrated_fs.toFixed(3)),
    hybrid_susceptibility_score: Number(hybridProb.toFixed(3)),
    differential_calibration: {
      data_coverage_mode: mode,
      supporting_real_samples: samples,
      block_differential_anomaly: Number(diffAnom.toFixed(3)),
      delta_fs_offset: Number(deltaFsOffset.toFixed(3)),
      calibrated_fs: Number(calibrated_fs.toFixed(3)),
      calibrated_c_prime_kpa: Number(calibrated_c.toFixed(2)),
      calibrated_z_m: Number(calibrated_z.toFixed(2)),
    },
    failure_probability_pct: Number(failureProb.toFixed(1)),
    stability_regime:
      calibrated_fs < 1.0
        ? "CRITICAL_FAILURE"
        : calibrated_fs < 1.25
        ? "MARGINAL_WATCH"
        : "STABLE",
  };
}

export async function synthesizeSatellitePhysics(
  inputs: SatelliteSensorInputs
): Promise<SynthesizedPhysicsResponse> {
  try {
    const res = await fetch(`${API_BASE}/physics/synthesize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(inputs),
    });
    if (!res.ok) throw new Error("Fallback to local synthesizer");
    return await res.json();
  } catch {
    return computeLocalSatelliteSynthesis(inputs);
  }
}

export function buildInteractiveBasinGrid(
  basin: BasinMetadata,
  baseInputs: SatelliteSensorInputs
): GridFeature[] {
  const [minLon, minLat, maxLon, maxLat] = basin.bbox;
  const features: GridFeature[] = [];
  const rows = 8;
  const cols = 8;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const lon = minLon + 0.08 + ((maxLon - minLon - 0.16) * c) / (cols - 1);
      const lat = minLat + 0.06 + ((maxLat - minLat - 0.12) * r) / (rows - 1);
      const cellSlope = Math.min(
        65,
        Math.max(12, baseInputs.slope_deg + ((r + c) % 5 - 2) * 3.5)
      );
      const cellElev = baseInputs.elevation_m + (r - 3.5) * 280;
      const cellSyn = computeLocalSatelliteSynthesis({
        ...baseInputs,
        latitude: lat,
        longitude: lon,
        slope_deg: cellSlope,
        elevation_m: cellElev,
      });

      features.push({
        type: "Feature",
        geometry: {
          type: "Point",
          coordinates: [Number(lon.toFixed(5)), Number(lat.toFixed(5)), cellElev],
        },
        properties: {
          cell_id: `${basin.id}_${r}_${c}`,
          elevation_m: Math.round(cellElev),
          slope_deg: Number(cellSlope.toFixed(1)),
          soil_depth_z_m: cellSyn.synthesized_variables.z_m,
          water_table_zw_m: cellSyn.synthesized_variables.z_w_m,
          fs: cellSyn.deterministic_fs,
          calibrated_fs: cellSyn.calibrated_fs,
          differential_anomaly: cellSyn.differential_calibration.block_differential_anomaly,
          data_coverage_mode: cellSyn.differential_calibration.data_coverage_mode,
          supporting_real_samples: cellSyn.differential_calibration.supporting_real_samples,
          critical: cellSyn.calibrated_fs < 1.0,
        },
      });
    }
  }
  return features;
}
