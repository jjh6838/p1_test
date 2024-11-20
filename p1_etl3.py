import geopandas as gpd
import rasterio
from rasterio.features import shapes
from rasterio.mask import mask
from shapely.geometry import shape, Point, LineString, Polygon
from scipy.spatial import Voronoi
import numpy as np
import os
import pandas as pd

# File paths
energy_facilities_path = r"KOR\wri_powerplants\wri-powerplants__KOR.gpkg"
grid_lines_path = r"KOR\gridfinder\gridfinder-version_1-grid-kor.gpkg"
population_raster_path = r"KOR\jrc_ghsl\GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0__KOR.tif"
south_korea_boundary_path = r"GADM\gadm41_KOR.gpkg"
output_gpkg_path = r"outputs\korea_integrated_dataset_v3.gpkg"

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
    if isinstance(val, dict):
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
        if polygon.is_valid:
            voronoi_polygons.append(polygon)

# Create a GeoDataFrame for Voronoi polygons
voronoi_gdf = gpd.GeoDataFrame(
    {'geometry': voronoi_polygons},
    crs=common_crs
)

# Use spatial indexing to identify Voronoi polygons near the boundary
boundary_sindex = korea_boundary.sindex
voronoi_sindex = voronoi_gdf.sindex

# Clip only the Voronoi polygons that intersect with the boundary
clipped_voronoi_polygons = []
for idx, polygon in voronoi_gdf.iterrows():
    possible_matches_index = list(boundary_sindex.intersection(polygon.geometry.bounds))
    possible_matches = korea_boundary.iloc[possible_matches_index]
    if possible_matches.intersects(polygon.geometry).any():
        clipped_polygon = polygon.geometry.intersection(korea_boundary.unary_union)
        if not clipped_polygon.is_empty:
            clipped_voronoi_polygons.append(clipped_polygon)
    else:
        clipped_voronoi_polygons.append(polygon.geometry)

# Create a GeoDataFrame for clipped Voronoi polygons
clipped_voronoi_gdf = gpd.GeoDataFrame(
    {'geometry': clipped_voronoi_polygons},
    crs=common_crs
)

# Step 5: Reproject layers to a projected CRS for distance calculations
projected_crs = "EPSG:3857"
energy_facilities_proj = energy_facilities_korea.to_crs(projected_crs)
grid_lines_proj = grid_lines_korea.to_crs(projected_crs)
population_centroids_proj = population_centroids_gdf.to_crs(projected_crs)

# Step 6: Connect energy facilities to grid and population centroids
connections = []

# Use spatial indexing for faster distance calculations
grid_lines_proj_sindex = grid_lines_proj.sindex

# Connect energy facilities to grid
for _, facility in energy_facilities_proj.iterrows():
    possible_matches_index = list(grid_lines_proj_sindex.intersection(facility.geometry.bounds))
    possible_matches = grid_lines_proj.iloc[possible_matches_index]
    distances = possible_matches.distance(facility.geometry)
    nearest_line = possible_matches.loc[distances.idxmin()].geometry
    nearest_point = nearest_line.interpolate(nearest_line.project(facility.geometry))
    connection_line = LineString([facility.geometry, nearest_point])
    connections.append(connection_line)

# Create a GeoDataFrame for the new lines from energy facilities
new_lines_gdf = gpd.GeoDataFrame(
    {"geometry": connections},
    crs=projected_crs
)

# Combine grid lines and new lines from energy facilities
combined_lines_gdf = gpd.GeoDataFrame(
    pd.concat([grid_lines_proj, new_lines_gdf], ignore_index=True),
    crs=projected_crs
)

# Use spatial indexing for combined lines
combined_lines_sindex = combined_lines_gdf.sindex

# Connect population centroids to grid and new lines from energy facilities
for _, centroid in population_centroids_proj.iterrows():
    possible_matches_index = list(combined_lines_sindex.intersection(centroid.geometry.bounds))
    possible_matches = combined_lines_gdf.iloc[possible_matches_index]
    distances = possible_matches.distance(centroid.geometry)
    nearest_line = possible_matches.loc[distances.idxmin()].geometry
    nearest_point = nearest_line.interpolate(nearest_line.project(centroid.geometry))
    connections.append(LineString([centroid.geometry, nearest_point]))

# Convert connections to GeoDataFrame
connections_gdf = gpd.GeoDataFrame(
    {"geometry": connections},
    crs=projected_crs
).to_crs(common_crs)

# Reproject layers back to common CRS
energy_facilities_korea = energy_facilities_proj.to_crs(common_crs)
combined_lines_korea = combined_lines_gdf.to_crs(common_crs)
population_centroids_gdf = population_centroids_proj.to_crs(common_crs)

# Step 7: Save all layers to GeoPackage
layer_data = {
    "energy_facilities_korea": energy_facilities_korea,
    "grid_lines_korea": combined_lines_korea,
    "population_centroids": population_centroids_gdf,
    "connections": connections_gdf,
    "voronoi_polygons": clipped_voronoi_gdf,
}

# Write layers to GeoPackage
if os.path.exists(output_gpkg_path):
    os.remove(output_gpkg_path)

for layer_name, gdf in layer_data.items():
    gdf.to_file(output_gpkg_path, layer=layer_name, driver="GPKG")

print(f"Korea-specific integrated dataset saved to {output_gpkg_path}")