"""
Shared configuration constants across all project scripts.
Change values here and they will be used by all dependent scripts.

Used by:
  - process_country_supply.py
  - process_country_siting.py
  - p1_c_cmip6_solar.py
  - p1_d_cmip6_wind.py
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
#   1. CMIP6 solar outputs: python p1_c_cmip6_solar.py --process-only
#   2. CMIP6 wind outputs:  python p1_d_cmip6_wind.py --process-only
#   3. All country supply outputs: python process_country_supply.py <ISO3>
#   4. Combined GPKGs: python combine_one_results.py <ISO3>

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
