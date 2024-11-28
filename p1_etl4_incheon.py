import warnings
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from rasterio.mask import mask
from shapely.geometry import shape, Point, LineString, Polygon, MultiLineString
from shapely.ops import nearest_points
from scipy.spatial import Voronoi
import numpy as np
import os
import pandas as pd
import networkx as nx
from shapely.ops import split, linemerge
from itertools import combinations

# Suppress specific user warnings
# warnings.filterwarnings("ignore", message="You are attempting to write an empty DataFrame to file")

# File paths
energy_data_csv_path = r"ember_energy_data\yearly_full_release_long_format.csv"
energy_facilities_path = r"re_data\Global-Integrated-Power-June-2024.xlsx"
grid_lines_path = r"KOR\gridfinder\gridfinder-version_1-grid-kor.gpkg"
population_raster_path = r"KOR\jrc_ghsl\GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0__KOR.tif"
south_korea_boundary_path = r"GADM\gadm41_KOR.gpkg"
# Output GeoPackage path
output_gpkg_path = r"outputs\korea_integrated_dataset_v4_jeju.gpkg"

# Read the energy facilities data from the Excel file
energy_facilities_df = pd.read_excel(energy_facilities_path, sheet_name="Powerfacilities")
filtered_energy_facilities_df = energy_facilities_df[
    (energy_facilities_df['Country/area'] == 'South Korea') &
    (energy_facilities_df['Status'] == 'operating')
]
# Convert the filtered DataFrame to a GeoDataFrame
energy_facilities = gpd.GeoDataFrame(
    filtered_energy_facilities_df,
    geometry=gpd.points_from_xy(filtered_energy_facilities_df['Longitude'], filtered_energy_facilities_df['Latitude']),
    crs="EPSG:4326"
)

# Step 1: Load grid lines and South Korea boundary
grid_lines = gpd.read_file(grid_lines_path)
korea_boundary = gpd.read_file(south_korea_boundary_path, layer='ADM_ADM_1')

# Filter the boundary data and set common CRS
regions_of_interest = ['Jeju']
filtered_boundary = korea_boundary[korea_boundary['NAME_1'].isin(regions_of_interest)]
common_crs = korea_boundary.crs
projected_crs = "EPSG:3857"  # For distance calculations

# Ensure CRS matches across all layers
energy_facilities = energy_facilities.to_crs(common_crs)
grid_lines = grid_lines.to_crs(common_crs)

# Step 2: Clip energy facilities and grid lines to filtered boundary
energy_facilities_korea = gpd.clip(energy_facilities, filtered_boundary)
grid_lines_korea = gpd.clip(grid_lines, filtered_boundary)

print(f"Total Capacity of operating facilities in South Korea: {energy_facilities_korea['Capacity (MW)'].sum()} MW")

# Step 3: Load population raster and extract centroids
with rasterio.open(population_raster_path) as src:
    filtered_boundary_reproj = filtered_boundary.to_crs(src.crs)
    masked_population, masked_transform = mask(
        src, [geom for geom in filtered_boundary_reproj.geometry], crop=True
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
    crs=src.crs
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
energy_facilities_proj = energy_facilities_korea.to_crs(projected_crs)
grid_lines_proj = grid_lines_korea.to_crs(projected_crs)
population_centroids_proj = population_centroids_gdf.to_crs(projected_crs)

print(f"Energy facilities projected CRS: {energy_facilities_proj.crs}")  # Debug statement
print(f"Grid lines projected CRS: {grid_lines_proj.crs}")  # Debug statement
print(f"Population centroids projected CRS: {population_centroids_proj.crs}")  # Debug statement


# New Step: Create Network Graph
def create_network_graph(facilities_gdf, grid_lines_gdf, centroids_gdf):
    # Initialize empty graph
    G = nx.Graph()
    
    # 1. Process grid lines to ensure single linestrings
    single_lines = []
    for _, row in grid_lines_gdf.iterrows():
        if isinstance(row.geometry, MultiLineString):
            single_lines.extend(list(row.geometry.geoms))  # Use .geoms to iterate over MultiLineString
        else:
            single_lines.append(row.geometry)
    
    # 2. Collect all nodes
    nodes = set()

    # 3. Create sets of node positions for each type
    facility_nodes = set()
    pop_centroid_nodes = set()
    grid_line_nodes = set()
    
    for point in facilities_gdf.geometry:
        facility_nodes.add((point.x, point.y))
    
    for point in centroids_gdf.geometry:
        pop_centroid_nodes.add((point.x, point.y))
        
    for line in single_lines:
        grid_line_nodes.add((line.coords[0][0], line.coords[0][1]))
        grid_line_nodes.add((line.coords[-1][0], line.coords[-1][1]))

    # 4. Add nodes to the graph with a 'type' attribute
    for node in facility_nodes:
        G.add_node(node, pos=node, type='facility')
    for node in pop_centroid_nodes:
        G.add_node(node, pos=node, type='pop_centroid')
    for node in grid_line_nodes:
        G.add_node(node, pos=node, type='grid_line')
    
    # 5. Create edges along existing grid lines
    for line in single_lines:
        coords = list(line.coords)
        for i in range(len(coords) - 1):
            start_node = coords[i]
            end_node = coords[i + 1]
            G.add_edge(start_node, end_node, weight=Point(start_node).distance(Point(end_node)))

    # 6. Connect facilities and centroids to the nearest grid line
    for facility in facilities_gdf.geometry:
        nearest_line = grid_lines_gdf.distance(facility).idxmin()
        nearest_point = nearest_points(facility, grid_lines_gdf.loc[nearest_line].geometry)[1]
        G.add_edge((facility.x, facility.y), (nearest_point.x, nearest_point.y), weight=facility.distance(nearest_point))

    for centroid in centroids_gdf.geometry:
        nearest_line = grid_lines_gdf.distance(centroid).idxmin()
        nearest_point = nearest_points(centroid, grid_lines_gdf.loc[nearest_line].geometry)[1]
        distance = centroid.distance(nearest_point)
        if distance <= 10000:  # 50 km in meters
            G.add_edge((centroid.x, centroid.y), (nearest_point.x, nearest_point.y), weight=distance)
    
    # Ensure all nodes have a 'type' attribute
    for node, data in G.nodes(data=True):
        if 'type' not in data:
            data['type'] = 'grid_line'
    
    return G
    
# Create network graph
network_graph = create_network_graph(energy_facilities_proj, grid_lines_proj, population_centroids_proj)


# Convert network graph to GeoDataFrame for visualization and storage
graph_nodes = gpd.GeoDataFrame(
    {
        'geometry': [Point(node) for node in network_graph.nodes()],
        'type': [data['type'] for node, data in network_graph.nodes(data=True)]
    },
    crs=projected_crs
)

graph_edges = gpd.GeoDataFrame(
    {
        'geometry': [LineString([Point(edge[0]), Point(edge[1])]) for edge in network_graph.edges()],
        'length': [LineString([Point(edge[0]), Point(edge[1])]).length for edge in network_graph.edges()]
    },
    crs=projected_crs
)

# Find shortest paths from population centroids to the nearest facility
def find_shortest_paths(network_graph, num_centroids=10):
    shortest_paths = []
    pop_centroid_nodes = [node for node, data in network_graph.nodes(data=True) if data['type'] == 'pop_centroid']
    facility_nodes = [node for node, data in network_graph.nodes(data=True) if data['type'] == 'facility']
    
    for i, centroid in enumerate(pop_centroid_nodes):
        if i >= num_centroids:
            break
        try:
            # Find the shortest path to any facility node
            lengths, paths = nx.single_source_dijkstra(network_graph, centroid)
            nearest_facility = min(facility_nodes, key=lambda node: lengths.get(node, float('inf')))
            shortest_paths.append((centroid, nearest_facility, paths[nearest_facility], lengths[nearest_facility]))
        except Exception as e:
            print(f"Error finding path for centroid {centroid}: {e}")
    
    return shortest_paths

# Get the shortest paths for the first 20 centroids
shortest_paths = find_shortest_paths(network_graph, num_centroids=10)

# Print the shortest paths for the first 20 centroids
for centroid, facility, path, length in shortest_paths:
    print(f"Centroid {centroid} -> Facility {facility}: Path {path}, Length {length}")

# Step 7: Save all layers to GeoPackage
# First ensure all layers have matching CRS
layer_data = {
    "energy_facilities_korea": energy_facilities_proj.to_crs(common_crs),
    "grid_lines_korea": grid_lines_proj.to_crs(common_crs),
    "population_centroids": population_centroids_proj.to_crs(common_crs),
    "voronoi_polygons": voronoi_gdf.to_crs(common_crs),
    "network_nodes": graph_nodes.to_crs(common_crs),
    "network_edges": graph_edges.to_crs(common_crs),
    # Convert shortest paths to GeoDataFrame
    "shortest_paths": gpd.GeoDataFrame(
        {"geometry": [LineString([Point(coord) for coord in path]) for _, _, path, _ in shortest_paths]},
        crs=projected_crs
    ).to_crs(common_crs)
}

# Write layers to GeoPackage
if os.path.exists(output_gpkg_path):
    os.remove(output_gpkg_path)

try:
    for layer_name, gdf in layer_data.items():
        if isinstance(gdf, gpd.GeoDataFrame) and not gdf.empty:
            print(f"Writing layer {layer_name} to GeoPackage")
            gdf.to_file(output_gpkg_path, layer=layer_name, driver="GPKG")
        else:
            print(f"Layer {layer_name} is empty or not a GeoDataFrame and will not be written to GeoPackage")
except Exception as e:
    print(f"An error occurred: {e}")
    raise