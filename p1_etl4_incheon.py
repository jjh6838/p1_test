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

# Define CRS constants
GEOGRAPHIC_CRS = "EPSG:4326"  # For input/output and geographic operations
PROJECTED_CRS = "EPSG:3857"   # For distance calculations

# Convert the filtered DataFrame to a GeoDataFrame
energy_facilities = gpd.GeoDataFrame(
    filtered_energy_facilities_df,
    geometry=gpd.points_from_xy(filtered_energy_facilities_df['Longitude'], filtered_energy_facilities_df['Latitude']),
    crs=GEOGRAPHIC_CRS
)

# Create a dictionary to map facility coordinates to GEM IDs
energy_facilities_proj = energy_facilities.to_crs(PROJECTED_CRS)
facility_gem_ids = dict(zip(
    [(x, y) for x, y in zip(energy_facilities_proj['geometry'].x, energy_facilities_proj['geometry'].y)],
    energy_facilities['GEM unit/phase ID']
))

# Step 1: Load grid lines and South Korea boundary
grid_lines = gpd.read_file(grid_lines_path)
korea_boundary = gpd.read_file(south_korea_boundary_path, layer='ADM_ADM_1')

# Filter the boundary data and set common CRS
regions_of_interest = ['Jeju']
filtered_boundary = korea_boundary[korea_boundary['NAME_1'].isin(regions_of_interest)]
common_crs = korea_boundary.crs

# Ensure consistent CRS for clipping operations
energy_facilities_clip = energy_facilities.to_crs(filtered_boundary.crs)
grid_lines_clip = grid_lines.to_crs(filtered_boundary.crs)

# Step 2: Clip energy facilities and grid lines to filtered boundary
energy_facilities = gpd.clip(energy_facilities_clip, filtered_boundary)
grid_lines = gpd.clip(grid_lines_clip, filtered_boundary)

print(f"Total Capacity of operating facilities in South Korea: {energy_facilities['Capacity (MW)'].sum()} MW")

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
energy_facilities_proj = energy_facilities.to_crs(PROJECTED_CRS)
grid_lines_proj = grid_lines.to_crs(PROJECTED_CRS)
population_centroids_proj = population_centroids_gdf.to_crs(PROJECTED_CRS)

print(f"Energy facilities projected CRS: {energy_facilities_proj.crs}")  # Debug statement
print(f"Grid lines projected CRS: {grid_lines_proj.crs}")  # Debug statement
print(f"Population centroids projected CRS: {population_centroids_proj.crs}")  # Debug statement


from shapely.ops import unary_union
from shapely.ops import split as split_line

def split_intersecting_edges(lines):
    """Split lines at their intersections and return all unique segments."""
    # Merge all lines
    merged = unary_union(lines)
    
    # Find all intersection points
    if isinstance(merged, (LineString, MultiLineString)):
        # Create a list of all segments
        segments = []
        if isinstance(merged, LineString):
            segments.append(merged)
        else:
            segments.extend(list(merged.geoms))
            
        # Split each segment at intersection points
        final_segments = []
        for segment in segments:
            coords = list(segment.coords)
            for i in range(len(coords) - 1):
                final_segments.append(LineString([coords[i], coords[i + 1]]))
                
        return final_segments
    return []

def create_network_graph(facilities_gdf, grid_lines_gdf, centroids_gdf):
    # Initialize empty graph
    G = nx.Graph()
    
    # 1. Process grid lines and split at intersections
    single_lines = []
    for _, row in grid_lines_gdf.iterrows():
        if isinstance(row.geometry, MultiLineString):
            single_lines.extend(list(row.geometry.geoms))
        else:
            single_lines.append(row.geometry)
    
    # Split lines at intersections
    split_lines = split_intersecting_edges(single_lines)
    print(f"Original line count: {len(single_lines)}, After splitting: {len(split_lines)}")
    
    # 2. Create nodes at all line endpoints and intersections
    nodes = set()
    for line in split_lines:
        coords = list(line.coords)
        nodes.add(coords[0])
        nodes.add(coords[-1])
    
    # 3. Add facility and centroid nodes
    facility_nodes = set((point.x, point.y) for point in facilities_gdf.geometry)
    pop_centroid_nodes = set((point.x, point.y) for point in centroids_gdf.geometry)
    
    # 4. Add all nodes to the graph with their types
    for node in nodes:
        G.add_node(node, pos=node, type='grid_line')
    for node in facility_nodes:
        G.add_node(node, pos=node, type='facility')
    for node in pop_centroid_nodes:
        G.add_node(node, pos=node, type='pop_centroid')
    
    # 5. Create edges from split lines
    for line in split_lines:
        coords = list(line.coords)
        G.add_edge(coords[0], coords[-1], weight=line.length)
    
    # 6. Connect facilities and centroids to nearest grid nodes
    for point_gdf, point_type in [(facilities_gdf, 'facility'), (centroids_gdf, 'pop_centroid')]:
        for point in point_gdf.geometry:
            point_coord = (point.x, point.y)
            
            # Find nearest node in the grid network
            grid_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'grid_line']
            if grid_nodes:
                nearest_node = min(grid_nodes, 
                                 key=lambda n: Point(n).distance(point))
                
                # Add edge if within distance threshold (10km for centroids)
                distance = Point(nearest_node).distance(point)
                if point_type == 'pop_centroid' and distance > 10000:
                    continue
                G.add_edge(point_coord, nearest_node, weight=distance)
    
    return G

# Create network graph
network_graph = create_network_graph(energy_facilities_proj, grid_lines_proj, population_centroids_proj)


# Convert network graph to GeoDataFrame for visualization and storage
graph_nodes = gpd.GeoDataFrame(
    {
        'geometry': [Point(node) for node in network_graph.nodes()],
        'type': [data['type'] for node, data in network_graph.nodes(data=True)]
    },
    crs=PROJECTED_CRS
)

graph_edges = gpd.GeoDataFrame(
    {
        'geometry': [LineString([Point(edge[0]), Point(edge[1])]) for edge in network_graph.edges()],
        'length': [LineString([Point(edge[0]), Point(edge[1])]).length for edge in network_graph.edges()]
    },
    crs=PROJECTED_CRS
)

# Find shortest paths from population centroids to the nearest facility
def find_shortest_paths(network_graph, num_centroids=None):
    shortest_paths = []
    pop_centroid_nodes = [node for node, data in network_graph.nodes(data=True) if data['type'] == 'pop_centroid']
    facility_nodes = [node for node, data in network_graph.nodes(data=True) if data['type'] == 'facility']
    
    # Limit number of centroids if specified
    if num_centroids is not None:
        pop_centroid_nodes = pop_centroid_nodes[:num_centroids]
    
    print(f"Processing {len(pop_centroid_nodes)} centroids and {len(facility_nodes)} facilities...")
    
    for i, centroid in enumerate(pop_centroid_nodes):
        try:
            # Find shortest paths to all facilities
            shortest_distance = float('inf')
            shortest_path = None
            nearest_facility = None
            
            for facility in facility_nodes:
                try:
                    # Use nx.shortest_path_length and nx.shortest_path instead of single_source_dijkstra
                    path_length = nx.shortest_path_length(network_graph, centroid, facility, weight='weight')
                    if path_length < shortest_distance:
                        shortest_distance = path_length
                        shortest_path = nx.shortest_path(network_graph, centroid, facility, weight='weight')
                        nearest_facility = facility
                except nx.NetworkXNoPath:
                    continue
            
            if shortest_path is not None:
                shortest_paths.append((centroid, nearest_facility, shortest_path, shortest_distance))
                if i % 500 == 0:  # Progress update every 500 centroids
                    print(f"Processed {i+1}/{len(pop_centroid_nodes)} centroids")
        except Exception as e:
            print(f"Error finding path for centroid {centroid}: {e}")
    
    print(f"Successfully found {len(shortest_paths)} paths")
    return shortest_paths

# Update the shortest paths call
shortest_paths = find_shortest_paths(network_graph)  # Remove the num_centroids limit

# Update the shortest paths GeoDataFrame creation
shortest_paths_gdf = gpd.GeoDataFrame(
    {
        "geometry": [LineString([Point(coord) for coord in path]) for _, _, path, _ in shortest_paths],
        "distance": [distance for _, _, _, distance in shortest_paths],
        "facility_gem_id": [facility_gem_ids.get(facility, 'unknown') for _, facility, _, _ in shortest_paths]
    }, 
    crs=PROJECTED_CRS
)

# Step 7: Save all layers to GeoPackage
# First ensure all layers have matching CRS
layer_data = {
    "energy_facilities": energy_facilities_proj.to_crs(common_crs),
    "grid_lines": grid_lines_proj.to_crs(common_crs),
    "population_centroids": population_centroids_proj.to_crs(common_crs),
    "voronoi_polygons": voronoi_gdf.to_crs(common_crs),
    "network_nodes": graph_nodes.to_crs(common_crs),
    "network_edges": graph_edges.to_crs(common_crs),
    "shortest_paths": shortest_paths_gdf.to_crs(common_crs)
}


# Visualization using GeoPandas and Matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch  # Add this import

# Before visualization, ensure all layers are in the same CRS (use common_crs for visualization)
visualization_crs = common_crs

# Convert all layers to visualization CRS
voronoi_gdf = voronoi_gdf.to_crs(visualization_crs)
grid_lines_proj = grid_lines_proj.to_crs(visualization_crs)
energy_facilities_proj = energy_facilities_proj.to_crs(visualization_crs)
population_centroids_proj = population_centroids_proj.to_crs(visualization_crs)
graph_nodes = graph_nodes.to_crs(visualization_crs)
graph_edges = graph_edges.to_crs(visualization_crs)
shortest_paths_gdf = shortest_paths_gdf.to_crs(visualization_crs)

# Create a figure and axis with specific bounds
fig, ax = plt.subplots(figsize=(12, 12))

# Get the bounds from filtered_boundary
bounds = filtered_boundary.to_crs(visualization_crs).total_bounds

# Plot Voronoi polygons
voronoi_gdf.plot(ax=ax, color='lightblue', edgecolor='none', alpha=0.5, label='Voronoi Polygons')

# Plot grid lines
grid_lines_proj.plot(ax=ax, color='gray', linewidth=1, label='Grid Lines')

# Plot energy facilities
energy_facilities_proj.plot(ax=ax, color='red', markersize=50, marker='^', label='Facilities')

# Plot population centroids
population_centroids_proj.plot(ax=ax, color='green', markersize=20, marker='o', label='Centroids')

# Plot network edges
graph_edges.plot(ax=ax, color='blue', linewidth=0.5, alpha=0.7, label='Network Connections')

# Plot network nodes
graph_nodes[graph_nodes['type'] == 'grid_line'].plot(ax=ax, color='blue', markersize=10, marker='.', label='Grid Line Nodes')
graph_nodes[graph_nodes['type'] == 'facility'].plot(ax=ax, color='red', markersize=50, marker='^', label='Facility Nodes')
graph_nodes[graph_nodes['type'] == 'pop_centroid'].plot(ax=ax, color='green', markersize=20, marker='o', label='Centroid Nodes')

# Plot shortest paths
shortest_paths_gdf.plot(ax=ax, color='red', linewidth=2, linestyle='--', alpha=0.7, label='Shortest Paths')

# Customize the legend
legend_elements = [
    Line2D([0], [0], marker='^', color='w', label='Facilities', markerfacecolor='red', markersize=15),
    Line2D([0], [0], marker='o', color='w', label='Centroids', markerfacecolor='green', markersize=10),
    Line2D([0], [0], marker='.', color='w', label='Grid Line Nodes', markerfacecolor='blue', markersize=10),
    Line2D([0], [0], color='blue', lw=1, label='Network Connections'),
    Line2D([0], [0], color='red', lw=2, linestyle='--', label='Shortest Paths'),
    Patch(facecolor='lightblue', edgecolor='none', label='Voronoi Polygons'),
    Patch(facecolor='gray', edgecolor='none', label='Grid Lines')
]
ax.legend(handles=legend_elements, loc='upper right')

# Set the plot bounds
ax.set_xlim([bounds[0], bounds[2]])
ax.set_ylim([bounds[1], bounds[3]])

# Add titles and labels
ax.set_title('Comprehensive Network Visualization', fontsize=16)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()

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
