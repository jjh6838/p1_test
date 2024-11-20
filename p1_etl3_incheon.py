import warnings
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from rasterio.mask import mask
from shapely.geometry import shape, Point, LineString, Polygon
from shapely.ops import nearest_points
from scipy.spatial import Voronoi
import numpy as np
import os
import pandas as pd

# Suppress specific user warnings
warnings.filterwarnings("ignore", message="You are attempting to write an empty DataFrame to file")

# File paths
energy_facilities_path = r"KOR\wri_powerplants\wri-powerplants__KOR.gpkg"
grid_lines_path = r"KOR\gridfinder\gridfinder-version_1-grid-kor.gpkg"
population_raster_path = r"KOR\jrc_ghsl\GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0__KOR.tif"
south_korea_boundary_path = r"GADM\gadm41_KOR.gpkg"
output_gpkg_path = r"outputs\korea_integrated_dataset_v3_incheon.gpkg"

try:
    # Step 1: Load energy facilities, grid lines, and South Korea boundary
    energy_facilities = gpd.read_file(energy_facilities_path)
    grid_lines = gpd.read_file(grid_lines_path)
    korea_boundary = gpd.read_file(south_korea_boundary_path, layer='ADM_ADM_1')
except Exception as e:
    print(f"Error reading files: {e}")
    raise

# Filter the boundary data to include only "Incheon"
regions_of_interest = ["Incheon"]
filtered_boundary = korea_boundary[korea_boundary['NAME_1'].isin(regions_of_interest)]

# Ensure CRS matches across all layers
common_crs = korea_boundary.crs
print(f"Common CRS: {common_crs}")  # Debug statement
energy_facilities = energy_facilities.to_crs(common_crs)
grid_lines = grid_lines.to_crs(common_crs)

# Step 2: Clip energy facilities and grid lines to filtered boundary
energy_facilities_korea = gpd.clip(energy_facilities, filtered_boundary)
grid_lines_korea = gpd.clip(grid_lines, filtered_boundary)

print(f"Energy facilities CRS: {energy_facilities_korea.crs}")  # Debug statement
print(f"Grid lines CRS: {grid_lines_korea.crs}")  # Debug statement
print(f"Filtered boundary CRS: {filtered_boundary.crs}")  # Debug statement

# Step 3: Load population raster and extract centroids within filtered boundary
try:
    with rasterio.open(population_raster_path) as src:
        population_crs = src.crs
        print(f"Population raster CRS: {population_crs}")  # Debug statement

        # Reproject filtered boundary to raster CRS
        filtered_boundary_reproj = filtered_boundary.to_crs(population_crs)
        print(f"Filtered boundary reprojected CRS: {filtered_boundary_reproj.crs}")  # Debug statement

        # Mask raster with filtered boundary
        masked_population, masked_transform = mask(
            src, [geom for geom in filtered_boundary_reproj.geometry], crop=True
        )
except Exception as e:
    print(f"Error processing raster: {e}")
    raise

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

print(f"Population centroids CRS: {population_centroids_gdf.crs}")  # Debug statement

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

print(f"Energy facilities projected CRS: {energy_facilities_proj.crs}")  # Debug statement
print(f"Grid lines projected CRS: {grid_lines_proj.crs}")  # Debug statement
print(f"Population centroids projected CRS: {population_centroids_proj.crs}")  # Debug statement

# Step 6: Connect energy facilities to grid and population centroids
connections_energy = []
connections_pop = []

# Connect energy facilities to grid
for _, facility in energy_facilities_proj.iterrows():
    distances = grid_lines_proj.distance(facility.geometry)
    nearest_line = grid_lines_proj.loc[distances.idxmin()].geometry
    nearest_point = nearest_points(facility.geometry, nearest_line)[1]
    connections_energy.append(LineString([facility.geometry, nearest_point]))

# Connect population centroids to grid or energy facilities
for _, centroid in population_centroids_proj.iterrows():
    # Calculate distances to grid lines
    distances_grid = grid_lines_proj.distance(centroid.geometry)
    nearest_grid_line = grid_lines_proj.loc[distances_grid.idxmin()].geometry
    nearest_grid_point = nearest_points(centroid.geometry, nearest_grid_line)[1]
    distance_to_grid = centroid.geometry.distance(nearest_grid_point)
    
    # Calculate distances to energy facilities
    distances_facilities = energy_facilities_proj.distance(centroid.geometry)
    nearest_facility = energy_facilities_proj.loc[distances_facilities.idxmin()].geometry
    nearest_facility_point = nearest_points(centroid.geometry, nearest_facility)[1]
    distance_to_facility = centroid.geometry.distance(nearest_facility_point)
    
    # Choose the shorter connection and only add if shorter than 10 km
    if distance_to_grid < distance_to_facility and distance_to_grid <= 10000:
        connections_pop.append(LineString([centroid.geometry, nearest_grid_point]))
    elif distance_to_facility <= 10000:
        connections_pop.append(LineString([centroid.geometry, nearest_facility_point]))

# Convert connections to GeoDataFrame
connections_energy_gdf = gpd.GeoDataFrame(
    {"geometry": connections_energy},
    crs=projected_crs
).to_crs(common_crs)

connections_pop_gdf = gpd.GeoDataFrame(
    {"geometry": connections_pop},
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
    "connections_energy": connections_energy_gdf,
    "connections_pop": connections_pop_gdf,
    "voronoi_polygons": voronoi_gdf,
}

# Write layers to GeoPackage
if os.path.exists(output_gpkg_path):
    os.remove(output_gpkg_path)

try:
    for layer_name, gdf in layer_data.items():
        if not gdf.empty:
            print(f"Writing layer {layer_name} to GeoPackage")
            gdf.to_file(output_gpkg_path, layer=layer_name, driver="GPKG")
        else:
            print(f"Layer {layer_name} is empty and will not be written to GeoPackage")
except Exception as e:
    print(f"An error occurred: {e}")
    raise