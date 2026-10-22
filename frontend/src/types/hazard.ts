export type AlertLevel = "GREEN" | "YELLOW" | "ORANGE" | "RED";

export type DataCoverageMode =
  | "DIRECT_REAL_DATA_BLOCK_CALIBRATION"
  | "SCARCE_DATA_HYBRID_GRANULATION"
  | "NEIGHBOR_BLOCK_GRANULATED_SIMULATION";

export interface BasinMetadata {
  id: string;
  name: string;
  district: string;
  bbox: [number, number, number, number];
  center: [number, number];
  max_elevation_m: number;
  key_hotspots: string[];
}

export interface GridFeature {
  type: "Feature";
  geometry: {
    type: "Point";
    coordinates: [number, number, number];
  };
  properties: {
    cell_id: string;
    elevation_m: number;
    slope_deg: number;
    soil_depth_z_m: number;
    water_table_zw_m: number;
    fs: number;
    calibrated_fs: number;
    differential_anomaly: number;
    data_coverage_mode: DataCoverageMode;
    supporting_real_samples: number;
    critical: boolean;
  };
}

export interface SatelliteSensorInputs {
  latitude: number;
  longitude: number;
  local_relief_m: number;
  elevation_m: number;
  slope_deg: number;
  aspect_deg: number;
  profile_curvature: number;
  plan_curvature: number;
  upslope_contributing_area_m2: number;
  sentinel2_ndvi: number;
  gedi_canopy_height_m: number;
  sand_fraction: number;
  clay_fraction: number;
  coarse_rock_fragment_fraction: number;
  dry_bulk_density_g_cm3: number;
  distance_to_fault_km: number;
  modis_lst_mean_c: number;
  diurnal_lst_range_c: number;
  surface_albedo: number;
  diurnal_freeze_thaw_cycles_30d: number;
  pdd_c_days_7d: number;
  snow_water_equivalent_mm: number;
  imerg_rain_72h_mm: number;
  antecedent_rain_30d_api_mm: number;
  smap_rootzone_moisture_vwc: number;
  sar_l_to_c_band_penetration_ratio: number;
  bedrock_pga_g: number;
  insar_los_velocity_mm_month: number;
}

export interface DifferentialCalibrationInfo {
  data_coverage_mode: DataCoverageMode;
  supporting_real_samples: number;
  block_differential_anomaly: number;
  delta_fs_offset: number;
  calibrated_fs: number;
  calibrated_c_prime_kpa: number;
  calibrated_z_m: number;
}

export interface SynthesizedPhysicsResponse {
  synthesized_variables: {
    c_prime_kpa: number;
    c_mineral_kpa: number;
    c_root_kpa: number;
    c_ice_kpa: number;
    matric_suction_kpa: number;
    phi_prime_deg: number;
    gamma_kn_m3: number;
    z_m: number;
    apparent_thermal_inertia: number;
    lidar_root_floor_m: number;
    z_w_m: number;
    saturation_ratio: number;
    surface_surcharge_q_kpa: number;
    seismic_kh: number;
    ridge_amplification_factor: number;
    twi: number;
    porosity: number;
  };
  deterministic_fs: number;
  calibrated_fs: number;
  hybrid_susceptibility_score: number;
  differential_calibration: DifferentialCalibrationInfo;
  failure_probability_pct: number;
  stability_regime: string;
}

export interface CascadeAlertResponse {
  basin_id: string;
  basin_name: string;
  alert_level: AlertLevel;
  static_fs: number;
  dynamic_fs: number;
  ml_susceptibility_score: number;
  active_triggers: string[];
  precursor_details: Record<string, unknown>;
  runout_summary?: {
    runout_length_m: number;
    peak_velocity_m_s: number;
    affected_cells_30m: number;
  } | null;
}
