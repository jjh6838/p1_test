# 1. Initial Setup
# 1.1 Import required libraries
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

# 1.2 Suppress warnings
warnings.filterwarnings("ignore", message="You are attempting to write an empty DataFrame to file")
warnings.filterwarnings("ignore", message="Unknown extension is not supported and will be removed")  # Add this line

# 1.3 Define file paths
energy_data_csv_path = r"ember_energy_data\yearly_full_release_long_format.csv"
energy_facilities_path = r"re_data\Global-Integrated-Power-June-2024.xlsx"
grid_lines_path = r"KOR\gridfinder\gridfinder-version_1-grid-kor.gpkg"
population_raster_path = r"KOR\jrc_ghsl\GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0__KOR.tif"
south_korea_boundary_path = r"GADM\gadm41_KOR.gpkg"
# Output GeoPackage path
output_gpkg_path = r"outputs\korea_integrated_dataset_v4_jeju.gpkg"

# 2. Data Loading and Processing
# 2.1 Load and filter energy facilities
# Read the energy facilities data from the Excel file
energy_facilities_df = pd.read_excel(energy_facilities_path, sheet_name="Powerfacilities")

# First filter for South Korea and operating status
filtered_energy_facilities_df = energy_facilities_df[
    (energy_facilities_df['Country/area'] == 'South Korea') &
    (energy_facilities_df['Status'] == 'operating')
]

# Group by GEM location ID and merge
def merge_facilities(group):
    first_row = group.iloc[0].copy()
    first_row['Capacity (MW)'] = group['Capacity (MW)'].sum()
    first_row['Num of Merged Units'] = len(group)
    return first_row

filtered_energy_facilities_df = (filtered_energy_facilities_df
    .groupby('GEM location ID', as_index=False)
    .apply(merge_facilities, include_groups=False)  # Add include_groups=False
    .reset_index(drop=True))

# 2.2 Define CRS constants
GEOGRAPHIC_CRS = "EPSG:4326"  # For input/output and geographic operations
PROJECTED_CRS = "EPSG:3857"   # For distance calculations

# 2.3 Capacity adjustment functions
# Calculate proportion-adjusted capacity based on larger values between GEM and Ember
def adjust_capacity_proportionally(facilities_df):
    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    facilities_df = facilities_df.copy()
    
    # Import values from p1_ember_analysis1
    import importlib
    ember_analysis = importlib.import_module('p1_ember_analysis1')
    
    # Import capacity values and conversion rates
    target_capacities = {
        "bioenergy": ember_analysis.larger_value_bioenergy,
        "coal": ember_analysis.larger_value_coal,
        "hydropower": ember_analysis.larger_value_hydro,
        "nuclear": ember_analysis.larger_value_nuclear,
        "solar": ember_analysis.larger_value_solar,
        "wind": ember_analysis.larger_value_wind,
        "geothermal": ember_analysis.larger_value_geothermal,
        "oil/gas": ember_analysis.larger_value_oilgas
    }
    
    # Import conversion rates
    conversion_rates = {
        "bioenergy": ember_analysis.conversion_rate_bioenergy,
        "coal": ember_analysis.conversion_rate_coal,
        "hydropower": ember_analysis.conversion_rate_hydro,
        "nuclear": ember_analysis.conversion_rate_nuclear,
        "solar": ember_analysis.conversion_rate_solar,
        "wind": ember_analysis.conversion_rate_wind,
        "geothermal": ember_analysis.conversion_rate_geothermal,
        "oil/gas": ember_analysis.conversion_rate_oilgas
    }

    # Calculate scaling factors and apply adjustments (existing code)
    type_capacities = facilities_df.groupby('Type')['Capacity (MW)'].sum()
    scaling_factors = {}
    for type_ in target_capacities:
        current_capacity = type_capacities.get(type_, 0)
        target_capacity = target_capacities[type_]
        scaling_factors[type_] = target_capacity / current_capacity if current_capacity > 0 else 1.0
    
    # Apply scaling factors and calculate MWh
    facilities_df.loc[:, 'Original_Capacity_MW'] = facilities_df['Capacity (MW)']
    facilities_df.loc[:, 'Scaling_Factor'] = facilities_df['Type'].map(scaling_factors)
    facilities_df.loc[:, 'Adjusted_Capacity_MW'] = facilities_df['Capacity (MW)'] * facilities_df['Type'].map(scaling_factors)
    facilities_df.loc[:, 'Conversion_Rate'] = facilities_df['Type'].map(conversion_rates)
    facilities_df.loc[:, 'Annual_MWh'] = facilities_df['Adjusted_Capacity_MW'] * facilities_df['Conversion_Rate'] # MW to MWh for 1 year
    
    return facilities_df

# Apply the adjustment before converting to GeoDataFrame
filtered_energy_facilities_df = adjust_capacity_proportionally(filtered_energy_facilities_df)

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

# 3. Spatial Data Processing
# 3.1 Load and process boundary data
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


# 3.2 Process population raster data
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

# After creating population_centroids_gdf and before network graph creation, add:
def calculate_centroid_mwh(centroids_gdf, facilities_gdf):
    # Calculate total available MWh
    total_available_mwh = facilities_gdf['Annual_MWh'].sum()
    
    # Calculate total population
    total_population = centroids_gdf['population'].sum()
    
    # Calculate proportions and percentages
    centroids_gdf['proportion'] = centroids_gdf['population'] / total_population
    centroids_gdf['population_share_pct'] = centroids_gdf['proportion'] * 100  # Convert to percentage
    centroids_gdf['needed_mwh'] = centroids_gdf['proportion'] * total_available_mwh
    
    print(f"Total available MWh: {total_available_mwh:.2f}")
    print(f"Total population: {total_population}")
    
    return centroids_gdf

# Apply the calculation
population_centroids_gdf = calculate_centroid_mwh(population_centroids_gdf, energy_facilities)

# 3.3 Generate Voronoi polygons
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

# 3.4 Reproject layers for distance calculations
energy_facilities_proj = energy_facilities.to_crs(PROJECTED_CRS)
grid_lines_proj = grid_lines.to_crs(PROJECTED_CRS)
population_centroids_proj = population_centroids_gdf.to_crs(PROJECTED_CRS)

print(f"Energy facilities projected CRS: {energy_facilities_proj.crs}")
print(f"Grid lines projected CRS: {grid_lines_proj.crs}")
print(f"Population centroids projected CRS: {population_centroids_proj.crs}")

# 4. Network Analysis
# 4.1 Network creation functions
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

# 4.2 Create network graph
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


# 4.3 Process shortest paths
# Add index to population centroids for tracking
population_centroids_proj['centroid_id'] = population_centroids_proj.index

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

# Update the shortest paths GeoDataFrame creation with centroid IDs
shortest_paths_gdf = gpd.GeoDataFrame(
    {
        "geometry": [LineString([Point(coord) for coord in path]) for _, _, path, _ in shortest_paths],
        "distance": [distance for _, _, _, distance in shortest_paths],
        "facility_gem_id": [facility_gem_ids.get(facility, 'unknown') for _, facility, _, _ in shortest_paths],
        "centroid_coord": [centroid for centroid, _, _, _ in shortest_paths]
    }, 
    crs=PROJECTED_CRS
)

# Add centroid IDs by matching coordinates
shortest_paths_gdf['centroid_id'] = shortest_paths_gdf.apply(
    lambda row: population_centroids_proj[
        population_centroids_proj.geometry.apply(
            lambda point: (point.x, point.y) == row['centroid_coord']
        )
    ].index[0],
    axis=1
)

# 4.4 Process energy supply mapping
def process_energy_supply(shortest_paths_df, facilities_df, centroids_df):
    # Create a copy of facilities DataFrame to track remaining MWh
    facilities_remaining = facilities_df.copy()
    facilities_remaining['Remaining_MWh'] = facilities_remaining['Annual_MWh']
    
    # Create base supply status DataFrame with only necessary columns
    supply_status = gpd.GeoDataFrame({
        'geometry': centroids_df.geometry,
        'centroid_id': centroids_df.index,
        'population': centroids_df['population'],
        'needed_mwh': centroids_df['needed_mwh'],
        'filled_mwh_1': 0.0,
        'supply_status': 'not_filled',
        'facility_id': None  # Add new column
    }, crs=centroids_df.crs)
    
    # Sort paths by distance
    sorted_paths = shortest_paths_df.sort_values('distance')
    
    # Process each path
    for _, path in sorted_paths.iterrows():
        centroid_id = path['centroid_id']
        facility_id = path['facility_gem_id']
        
        # Get needed and available MWh
        needed_mwh = float(supply_status.loc[centroid_id, 'needed_mwh'])
        available_mwh = float(facilities_remaining.loc[
            facilities_remaining['GEM unit/phase ID'] == facility_id, 
            'Remaining_MWh'
        ].iloc[0])
        
        if available_mwh >= needed_mwh:
            supply_status.loc[centroid_id, 'filled_mwh_1'] = needed_mwh
            supply_status.loc[centroid_id, 'supply_status'] = 'filled'
            facilities_remaining.loc[
                facilities_remaining['GEM unit/phase ID'] == facility_id, 
                'Remaining_MWh'
            ] -= needed_mwh
            supply_status.loc[centroid_id, 'facility_id'] = facility_id  # Assign facility ID
        elif available_mwh > 0:
            supply_status.loc[centroid_id, 'filled_mwh_1'] = available_mwh
            supply_status.loc[centroid_id, 'supply_status'] = 'partially_filled'
            facilities_remaining.loc[
                facilities_remaining['GEM unit/phase ID'] == facility_id, 
                'Remaining_MWh'
            ] = 0
            supply_status.loc[centroid_id, 'facility_id'] = facility_id  # Assign facility ID
    
    return supply_status, facilities_remaining

# Ensure that 'energy_supply_status' includes the 'facility_id'
energy_supply_status, facilities_remaining = process_energy_supply(
    shortest_paths_gdf,
    energy_facilities_proj,
    population_centroids_proj
)

# After processing energy supply mapping, create a filtered GeoDataFrame for supplied paths
# Filter 'shortest_paths_gdf' to include only paths where the supply status is 'filled' or 'partially_filled'
supplied_shortest_paths_gdf = shortest_paths_gdf[
    shortest_paths_gdf['centroid_id'].isin(
        energy_supply_status[
            energy_supply_status['supply_status'].isin(['filled', 'partially_filled'])
        ]['centroid_id']
    )
]

# Ensure 'supplied_shortest_paths_gdf' is in the correct CRS
supplied_shortest_paths_gdf = supplied_shortest_paths_gdf.to_crs(common_crs)

# Update 'energy_facilities_proj' with 'Remaining_MWh' and calculate 'Supplied_MWh'
energy_facilities_proj = energy_facilities_proj.merge(
    facilities_remaining[['GEM unit/phase ID', 'Remaining_MWh']],
    on='GEM unit/phase ID',
    how='left'
)
energy_facilities_proj['Supplied_MWh'] = energy_facilities_proj['Annual_MWh'] - energy_facilities_proj['Remaining_MWh']

# 5. Data Visualization
# 5.1 Setup visualization parameters
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

# 5.2 Create plots
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
# plt.show()

# 6. Data Export
# 6.1 Prepare layers for export
# Step 7: Save all layers to GeoPackage
# First ensure all layers have matching CRS
layer_data = {
    "energy_facilities": energy_facilities_proj.to_crs(common_crs),
    "grid_lines": grid_lines_proj.to_crs(common_crs),
    "population_centroids": population_centroids_proj.to_crs(common_crs),  # Now includes needed_mwh
    "voronoi_polygons": voronoi_gdf.to_crs(common_crs),
    "network_nodes": graph_nodes.to_crs(common_crs),
    "network_edges": graph_edges.to_crs(common_crs),
    "shortest_paths": shortest_paths_gdf.to_crs(common_crs),
    "energy_supply_status": energy_supply_status.to_crs(common_crs),
    "supplied_shortest_paths": supplied_shortest_paths_gdf.to_crs(common_crs)
}

# Before exporting, clean up the GeoDataFrames to remove invalid data types
def clean_geodataframe(gdf):
    for column in gdf.columns:
        if gdf[column].dtype == object:
            # Check if the column contains tuples
            if gdf[column].apply(lambda x: isinstance(x, tuple)).any():
                print(f"Column '{column}' contains tuples and will be removed from {gdf.name}.")
                gdf = gdf.drop(columns=[column])
    return gdf

# Assign names to GeoDataFrames for identification
for layer_name, gdf in layer_data.items():
    if isinstance(gdf, gpd.GeoDataFrame):
        gdf.name = layer_name

# 6.2 Export to GeoPackage
# Write layers to GeoPackage
if os.path.exists(output_gpkg_path):
    os.remove(output_gpkg_path)

try:
    for layer_name, gdf in layer_data.items():
        if isinstance(gdf, gpd.GeoDataFrame) and not gdf.empty:
            # Clean the GeoDataFrame before writing
            gdf = clean_geodataframe(gdf)
            print(f"Writing layer {layer_name} to GeoPackage")
            gdf.to_file(output_gpkg_path, layer=layer_name, driver="GPKG")
        else:
            print(f"Layer {layer_name} is empty or not a GeoDataFrame and will not be written to GeoPackage")
except Exception as e:
    print(f"An error occurred: {e}")
    raise
