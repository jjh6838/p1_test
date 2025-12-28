"""
Shared configuration constants across all project scripts.
Change values here and they will be used by all dependent scripts.

Used by:
  - process_country_supply.py
  - process_country_siting.py
  - p1_d_viable_solar.py
  - p1_e_viable_wind.py
  - p1_f_viable_hydro.py
  - combine_one_results.py
  - combine_global_results.py
"""

# =============================================================================
# COMMON SETTINGS
# =============================================================================
COMMON_CRS = "EPSG:4326"  # WGS84 for input/output
ANALYSIS_YEAR = 2030  # Year for supply-demand analysis: 2024, 2030, or 2050
DEMAND_TYPES = ["Solar", "Wind", "Hydro", "Other Renewables", "Nuclear", "Fossil"]

# Supply factor for sensitivity analysis: each facility supplies X% of its capacity
# Default value used when running single scenario (no --run-all-scenarios flag)
# When --run-all-scenarios is used, this is overridden with [1.0, 0.9, 0.8, 0.7, 0.6]
SUPPLY_FACTOR = 1.0

# =============================================================================
# GRID RESOLUTION (for population centroids and CMIP6 outputs)
# =============================================================================
# GHS-POP native resolution is 30 arcsec (~1 km)
# POP_AGGREGATION_FACTOR aggregates native pixels into larger cells
# Final resolution = 30 * POP_AGGREGATION_FACTOR arcsec
#
# Examples:
#   Factor 10 → 300 arcsec (~9 km at equator)
#   Factor 5  → 150 arcsec (~4.5 km at equator)
#   Factor 2  → 60 arcsec (~1.8 km at equator)
#
# IMPORTANT: After changing POP_AGGREGATION_FACTOR, you MUST regenerate:
#   1. CMIP6 solar outputs: python p1_d_viable_solar.py --process-only
#   2. CMIP6 wind outputs:  python p1_e_viable_wind.py --process-only
#   3. CMIP6 hydro outputs: python p1_f_viable_hydro.py --process-only
#   4. All country supply outputs: python process_country_supply.py <ISO3>
#   5. Combined GPKGs: python combine_one_results.py <ISO3>

POP_AGGREGATION_FACTOR = 10

# Derived constants (do not modify directly)
GHS_POP_NATIVE_RESOLUTION_ARCSEC = 30
TARGET_RESOLUTION_ARCSEC = GHS_POP_NATIVE_RESOLUTION_ARCSEC * POP_AGGREGATION_FACTOR

# =============================================================================
# NETWORK / GRID SETTINGS (process_country_supply.py)
# =============================================================================
GRID_STITCH_DISTANCE_KM = 30  # Distance threshold (km) for stitching raw grid segments
NODE_SNAP_TOLERANCE_M = 100  # Snap distance (meters, UTM) for clustering nearby grid nodes
MAX_CONNECTION_DISTANCE_M = 50000  # Max distance (meters) for connecting facilities/centroids to grid
FACILITY_SEARCH_RADIUS_KM = 300  # Max radius (km) to search for facilities from each centroid

# =============================================================================
# SITING ANALYSIS SETTINGS (process_country_siting.py)
# =============================================================================
CLUSTER_RADIUS_KM = 50  # K-means clustering radius parameter
CLUSTER_MIN_SAMPLES = 1  # Minimum samples for DBSCAN geographic component detection
GRID_DISTANCE_THRESHOLD_KM = 50  # Threshold (km) to classify clusters as "remote" vs "near grid"
DROP_PERCENTAGE = 0.01  # Drop bottom X% of settlements by demand (0.01 = 1%)

# =============================================================================
# LAND COVER FILTERING SETTINGS (p1_c, p1_d, p1_e, p1_f)
# =============================================================================
# ESA CCI Land Cover 2022 class codes for viable energy siting
# See: https://cds.climate.copernicus.eu/datasets/satellite-land-cover
#
# Solar viable classes:
#   10-40: Cropland (rainfed, irrigated, mosaic)
#   130: Grassland
#   150: Sparse vegetation
#   200: Bare areas
LANDCOVER_VALID_SOLAR = [10, 20, 30, 40, 130, 150, 200]

# Wind viable classes (same as solar - open terrain)
LANDCOVER_VALID_WIND = [10, 20, 30, 40, 130, 150, 200]

# =============================================================================
# RESOURCE VIABILITY THRESHOLDS (p1_d_viable_solar.py, p1_e_viable_wind.py)
# =============================================================================
# Minimum PVOUT (kWh/kWp/day) for viable solar centroids
# Global Solar Atlas typical range: 2.5-6.0 kWh/kWp/day
SOLAR_PVOUT_THRESHOLD = 3.0

# Minimum WPD at 100m (W/m²) for viable wind centroids
# Global Wind Atlas typical range: 100-1000 W/m²
#
# NOTE: This threshold is set much lower than the typical 150 W/m² viability
# cutoff due to TWO systematic underestimation effects in our methodology:
#
# ═══════════════════════════════════════════════════════════════════════════
# ISSUE 1: Jensen's inequality — E[U³] >> (E[U])³
# ═══════════════════════════════════════════════════════════════════════════
# Wind power depends on U³, but we compute WPD from monthly-mean speeds:
#   WPD_computed = 0.5 * ρ * (mean(U))³
#   WPD_true     = 0.5 * ρ * mean(U³)
#
# Wind speed distributions are right-skewed; a few high-wind hours contribute
# disproportionately to energy. The ratio E[U³]/(E[U])³ is the Energy Pattern
# Factor (EPF), typically 1.4–2.0 for wind sites.
#
# References:
#   Justus, C.G., Hargraves, W.R., Mikhail, A., Graber, D. (1978).
#   "Methods for Estimating Wind Speed Frequency Distributions."
#   Journal of Applied Meteorology, 17(3), 350-353.
#   doi:10.1175/1520-0450(1978)017<0350:MFEWSF>2.0.CO;2
#   → EPF ≈ 1.9 for Weibull shape parameter k=2
#
#   Carta, J.A., Ramirez, P., Velazquez, S. (2009).
#   "A review of wind speed probability distributions used in wind energy analysis."
#   Renewable and Sustainable Energy Reviews, 13(5), 933-955.
#   doi:10.1016/j.rser.2008.05.005
#   → EPF = 1.4–2.0 depending on site variability
#
# ═══════════════════════════════════════════════════════════════════════════
# ISSUE 2: Vector-mean cancellation — sqrt(ū² + v̄²) << mean(sqrt(u² + v²))
# ═══════════════════════════════════════════════════════════════════════════
# ERA5 monthly-means provide mean u and v components separately. We compute:
#   U_computed = sqrt(mean(u)² + mean(v)²)
#
# But the true scalar mean speed is:
#   U_true = mean(sqrt(u² + v²))
#
# When wind direction varies within the month, the vector components cancel
# (e.g., easterlies and westerlies average to near-zero u), severely under-
# estimating scalar speed. This effect is worst in:
#   - Monsoon regions (seasonal wind reversals)
#   - Coastal areas (land-sea breeze oscillations)
#   - Mountain passes (diurnal flow reversals)
#   - Trade wind boundaries
#
# References:
#   Pryor, S.C., Barthelmie, R.J. (2010).
#   "Climate change impacts on wind energy: A review."
#   Renewable and Sustainable Energy Reviews, 14(1), 430-437.
#   doi:10.1016/j.rser.2009.07.028
#   → Documents directional variability effects on wind resource assessment
#
#   Staffell, I., Pfenninger, S. (2016).
#   "Using bias-corrected reanalysis to simulate current and future wind power output."
#   Energy, 114, 1224-1239.
#   doi:10.1016/j.energy.2016.08.068
#   → Shows reanalysis wind speed biases of 15-30% in complex terrain
#
# ═══════════════════════════════════════════════════════════════════════════
# COMBINED CORRECTION FACTOR
# ═══════════════════════════════════════════════════════════════════════════
# Conservative estimates:
#   - Jensen's inequality (EPF):     1.5–2.0×
#   - Vector cancellation:           1.5–3.0× (regime-dependent)
#   - Combined:                      ~3–6× underestimation
#
# With 150 W/m² as the typical viability threshold for true WPD:
#   threshold_computed = 150 / (correction_factor)
#   threshold_computed = 150 / 6 ≈ 25 W/m²  (conservative)
#
# A computed WPD of 25 W/m² likely corresponds to 75–150 W/m² true WPD.
WIND_WPD_THRESHOLD = 25

# Hydro viable classes (water-adjacent areas):
#   160: Tree cover, flooded, fresh or brackish water
#   170: Tree cover, flooded, saline water
#   180: Shrub or herbaceous cover, flooded
#   210: Water bodies
LANDCOVER_VALID_HYDRO = [160, 170, 180, 210]

# =============================================================================
# HYDROATLAS RIVER BUFFER SETTINGS (p1_f_viable_hydro.py)
# =============================================================================
# Buffer distance (meters) around suitable river reaches for hydro siting
HYDRO_RIVER_BUFFER_M = 5000  # 5 km buffer

# Minimum runoff threshold (mm/year) for viable hydro centroids
# Areas with runoff below this threshold are excluded from viable centroids output
HYDRO_RUNOFF_THRESHOLD_MM = 100
