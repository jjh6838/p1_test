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
energy_data_csv_path = r"ember_energy_data\yearly_full_release_long_format.csv"
energy_facilities_path = r"re_data\Global-Integrated-Power-June-2024.xlsx"
grid_lines_path = r"KOR\gridfinder\gridfinder-version_1-grid-kor.gpkg"
population_raster_path = r"KOR\jrc_ghsl\GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0__KOR.tif"
south_korea_boundary_path = r"GADM\gadm41_KOR.gpkg"
# Output GeoPackage path
output_gpkg_path = r"outputs\korea_integrated_dataset_v3_jeju.gpkg"


# Read the energy data CSV and calculate the sum of "Value" for the specified conditions
energy_data = pd.read_csv(energy_data_csv_path)
filtered_energy_data = energy_data[
    (energy_data['Country code'] == 'KOR') &
    (energy_data['Year'] == 2023) &
    (energy_data['Category'] == 'Capacity') &
    (energy_data['Subcategory'] == 'Fuel') &
    (energy_data['Unit'] == 'GW')
]
total_capacity = filtered_energy_data['Value'].sum()
print(f"Total Capacity for KOR in 2023: {total_capacity} GW")

# Read the energy facilities data from the Excel file
energy_facilities_df = pd.read_excel(energy_facilities_path, sheet_name="Powerfacilities")
filtered_energy_facilities_df = energy_facilities_df[
    (energy_facilities_df['Country/area'] == 'South Korea') &
    (energy_facilities_df['Status'] == 'operating')
]
total_capacity_mw = filtered_energy_facilities_df['Capacity (MW)'].sum()
print(f"Total Capacity of operating facilities in South Korea: {total_capacity_mw} MW")

# Convert the filtered DataFrame to a GeoDataFrame
energy_facilities = gpd.GeoDataFrame(
    filtered_energy_facilities_df,
    geometry=gpd.points_from_xy(filtered_energy_facilities_df['Longitude'], filtered_energy_facilities_df['Latitude']),
    crs="EPSG:4326"
)

# Step 1: Load grid lines and South Korea boundary
grid_lines = gpd.read_file(grid_lines_path)
korea_boundary = gpd.read_file(south_korea_boundary_path, layer='ADM_ADM_1')

# Filter the boundary data to include only "Incheon"
regions_of_interest = ['Jeju']
filtered_boundary = korea_boundary[korea_boundary['NAME_1'].isin(regions_of_interest)]

# Ensure CRS matches across all layers
common_crs = korea_boundary.crs
print(f"Common CRS: {common_crs}")  # Debug statement
energy_facilities = energy_facilities.to_crs(common_crs)
grid_lines = grid_lines.to_crs(common_crs)

# Step 2: Clip energy facilities and grid lines to filtered boundary
energy_facilities_korea = gpd.clip(energy_facilities, filtered_boundary)
grid_lines_korea = gpd.clip(grid_lines, filtered_boundary)

print(f"Total Capacity of operating facilities in South Korea: {energy_facilities_korea['Capacity (MW)'].sum()} MW")


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

# Calculate the bounding box of the filtered boundary
bounding_box = filtered_boundary.total_bounds
bbox_polygon = Polygon([
    (bounding_box[0], bounding_box[1]),
    (bounding_box[0], bounding_box[3]),
    (bounding_box[2], bounding_box[3]),
    (bounding_box[2], bounding_box[1])
])

# Create Voronoi polygons
voronoi_polygons = []
for region in vor.regions:
    if not -1 in region and len(region) > 0:
        polygon = Polygon([vor.vertices[i] for i in region])
        clipped_polygon = polygon.intersection(bbox_polygon)
        voronoi_polygons.append(clipped_polygon)

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

# Rule 1: Establish comprehensive connections
def establish_comprehensive_connections(connections_energy_gdf, grid_lines_korea, connections_pop_gdf):
    # Combine all connections into a single GeoDataFrame
    combined_connections = gpd.GeoDataFrame(pd.concat([connections_energy_gdf, grid_lines_korea, connections_pop_gdf], ignore_index=True))
    
    # Ensure the CRS is consistent
    combined_connections = combined_connections.to_crs(connections_energy_gdf.crs)
    
    return combined_connections

# Establish comprehensive connections
comprehensive_connections_gdf = establish_comprehensive_connections(connections_energy_gdf, grid_lines_korea, connections_pop_gdf)

#Rules 2-3



# Step 7: Save all layers to GeoPackage
layer_data = {
    "energy_facilities_korea": energy_facilities_korea,
    "grid_lines_korea": grid_lines_korea,
    "population_centroids": population_centroids_gdf,
    "connections_energy": connections_energy_gdf,
    "connections_pop": connections_pop_gdf,
    "voronoi_polygons": voronoi_gdf,
    "connections_all": comprehensive_connections_gdf
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