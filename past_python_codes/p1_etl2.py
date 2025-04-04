import geopandas as gpd
import rasterio
from rasterio.features import shapes
from rasterio.mask import mask
from shapely.geometry import shape, Point, LineString, Polygon
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import os

# File paths
energy_facilities_path = r"KOR\wri_powerplants\wri-powerplants__KOR.gpkg"  # Energy facilities (nodes)
grid_lines_path = r"KOR\gridfinder\gridfinder-version_1-grid-kor.gpkg"      # Grid (lines)
population_raster_path = r"KOR\jrc_ghsl\GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0__KOR.tif"  # Population raster
south_korea_boundary_path = r"GADM\gadm41_KOR.gpkg"                         # South Korea boundary
output_gpkg_path = r"outputs\korea_integrated_dataset_v2.gpkg"              # Output GeoPackage

# Step 1: Load energy facilities, grid lines, and South Korea boundary
energy_facilities = gpd.read_file(energy_facilities_path)
grid_lines = gpd.read_file(grid_lines_path)
korea_boundary = gpd.read_file(south_korea_boundary_path)

# Ensure CRS matches across all layers
common_crs = korea_boundary.crs
energy_facilities = energy_facilities.to_crs(common_crs)
grid_lines = grid_lines.to_crs(common_crs)

# Step 2: Clip energy facilities and grid lines to South Korea boundary
energy_facilities_korea = gpd.clip(energy_facilities, korea_boundary)
grid_lines_korea = gpd.clip(grid_lines, korea_boundary)

# Step 3: Load population raster and extract centroids within South Korea
with rasterio.open(population_raster_path) as src:
    population_crs = src.crs

    # Reproject Korea boundary to raster CRS
    korea_boundary_reproj = korea_boundary.to_crs(population_crs)

    # Mask raster with South Korea boundary
    masked_population, masked_transform = mask(
        src, [geom for geom in korea_boundary_reproj.geometry], crop=True
    )

# Extract centroids and population values of high-value population cells
threshold = 0  # Modify as needed
centroids = []
values = []

# Iterate over raster cells
for geom, val in shapes(masked_population[0], transform=masked_transform):
    # Check if val is a dictionary
    if isinstance(val, dict):
        # Replace 'value' with the correct key if different
        actual_value = val['value']
    else:
        actual_value = val
    if actual_value > threshold:
        centroid = shape(geom).centroid
        centroids.append(centroid)
        values.append(actual_value)

# Create a GeoDataFrame with centroids and population values
population_centroids_gdf = gpd.GeoDataFrame(
    {'geometry': centroids, 'population': values},
    crs=population_crs
).to_crs(common_crs)

# Step 4: Generate Voronoi polygons weighted by population values
points = np.array([(point.x, point.y) for point in population_centroids_gdf.geometry])
weights = np.array(values)
vor = Voronoi(points)

# Create Voronoi polygons
voronoi_polygons = []
for region in vor.regions:
    if not -1 in region and len(region) > 0:
        polygon = Polygon([vor.vertices[i] for i in region])
        voronoi_polygons.append(polygon)

# Create a GeoDataFrame for Voronoi polygons
voronoi_gdf = gpd.GeoDataFrame(
    {'geometry': voronoi_polygons},
    crs=common_crs
)

# Step 5: Reproject layers to a projected CRS for distance calculations
projected_crs = "EPSG:3857"  # Example projected CRS
energy_facilities_proj = energy_facilities_korea.to_crs(projected_crs)
grid_lines_proj = grid_lines_korea.to_crs(projected_crs)
population_centroids_proj = population_centroids_gdf.to_crs(projected_crs)

# Step 6: Connect energy facilities to grid and population centroids
connections = []

# Connect energy facilities to grid
for _, facility in energy_facilities_proj.iterrows():
    distances = grid_lines_proj.distance(facility.geometry)
    nearest_line = grid_lines_proj.loc[distances.idxmin()].geometry
    nearest_point = nearest_line.interpolate(nearest_line.project(facility.geometry))
    connections.append(LineString([facility.geometry, nearest_point]))

# Connect population centroids to grid
for _, centroid in population_centroids_proj.iterrows():
    distances = grid_lines_proj.distance(centroid.geometry)
    nearest_line = grid_lines_proj.loc[distances.idxmin()].geometry
    nearest_point = nearest_line.interpolate(nearest_line.project(centroid.geometry))
    connections.append(LineString([centroid.geometry, nearest_point]))

# Convert connections to GeoDataFrame
connections_gdf = gpd.GeoDataFrame(
    {"geometry": connections},
    crs=projected_crs
).to_crs(common_crs)

# Reproject layers back to common CRS
energy_facilities_korea = energy_facilities_proj.to_crs(common_crs)
grid_lines_korea = grid_lines_proj.to_crs(common_crs)
population_centroids_gdf = population_centroids_proj.to_crs(common_crs)

# Step 7: Save all layers to GeoPackage
layer_data = {
    "energy_facilities_korea": energy_facilities_korea,
    "grid_lines_korea": grid_lines_korea,
    "population_centroids": population_centroids_gdf,
    "connections": connections_gdf,
    "voronoi_polygons": voronoi_gdf,
}

# Write layers to GeoPackage
if os.path.exists(output_gpkg_path):
    os.remove(output_gpkg_path)

for layer_name, gdf in layer_data.items():
    gdf.to_file(output_gpkg_path, layer=layer_name, driver="GPKG")

print(f"Korea-specific integrated dataset saved to {output_gpkg_path}")