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
from shapely.ops import split as split_line  # Corrected import
import time
start_time = time.time()

# 1.2 Suppress warnings
warnings.filterwarnings("ignore", message="You are attempting to write an empty DataFrame to file")
warnings.filterwarnings("ignore", message="Unknown extension is not supported and will be removed")

# 1.3 Define file paths
energy_facilities_path = r"re_data\Global-Integrated-Power-February-2025-update-II.xlsx"
grid_lines_path = r"KOR\gridfinder\gridfinder-version_1-grid-kor.gpkg"
population_raster_path = r"KOR\jrc_ghsl\GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0__KOR.tif"
south_korea_boundary_path = r"GADM\gadm41_KOR.gpkg"
# Output GeoPackage path
output_gpkg_path = r"outputs\korea_integrated_dataset_v6_korea.gpkg"

# 1.4 Configuration: Toggle between original and prototyping approaches
USE_PROTOTYPING = True  # Set to True for prototyping, False for original

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
    
    # Import values from p1_c_ember_gem_2030_50.py
    import importlib
    ember_analysis = importlib.import_module('p1_c_ember_gem_2030') ###########################################
    
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

# Add this helper function after the facility_gem_ids creation
def get_facility_type(facility_id, facilities_df):
    if facility_id == 'unknown':
        return 'unknown'
    facility_mask = facilities_df['GEM unit/phase ID'] == facility_id
    return facilities_df.loc[facility_mask, 'Type'].iloc[0] if any(facility_mask) else 'unknown'

# 3. Spatial Data Processing
# 3.1 Load and process boundary data
# Step 1: Load grid lines and South Korea boundary
grid_lines = gpd.read_file(grid_lines_path)
korea_boundary = gpd.read_file(south_korea_boundary_path, layer='ADM_ADM_0')

# Filter the boundary data and set common CRS
regions_of_interest = ['KOR']
filtered_boundary = korea_boundary[korea_boundary['GID_0'].isin(regions_of_interest)]
common_crs = korea_boundary.crs

# Ensure consistent CRS for clipping operations
energy_facilities_clip = energy_facilities.to_crs(filtered_boundary.crs)
grid_lines_clip = grid_lines.to_crs(filtered_boundary.crs)

# Step 2: Clip energy facilities and grid lines to filtered boundary
energy_facilities = gpd.clip(energy_facilities_clip, filtered_boundary)
grid_lines = gpd.clip(grid_lines_clip, filtered_boundary)


# 3.2 Process population raster data
if USE_PROTOTYPING:
    # Prototyping approach: Aggregate population raster to a coarser resolution
    with rasterio.open(population_raster_path) as src:
        filtered_boundary_reproj = filtered_boundary.to_crs(src.crs)
        masked_population, masked_transform = mask(
            src, [geom for geom in filtered_boundary_reproj.geometry], crop=True
        )
        
        # Ensure dimensions are divisible by the aggregation factor
        aggregation_factor = 100  # Adjust this value as needed
        height, width = masked_population[0].shape

        # Calculate new dimensions that are divisible by the aggregation factor
        new_height = (height // aggregation_factor) * aggregation_factor
        new_width = (width // aggregation_factor) * aggregation_factor

        # Crop the array to the new dimensions
        cropped_population = masked_population[0][:new_height, :new_width]

        # Reshape and aggregate
        aggregated_population = cropped_population.reshape(
            new_height // aggregation_factor, aggregation_factor,
            new_width // aggregation_factor, aggregation_factor
        ).sum(axis=(1, 3))
        
        # Update the transform for the aggregated raster
        aggregated_transform = rasterio.Affine(
            masked_transform.a * aggregation_factor, masked_transform.b, masked_transform.c,
            masked_transform.d, masked_transform.e * aggregation_factor, masked_transform.f
        )

    # Extract centroids and population values of high-value population cells
    threshold = 0  # Modify as needed
    centroids = []
    values = []

    # Iterate over aggregated raster cells
    for geom, val in shapes(aggregated_population, transform=aggregated_transform):
        if val > threshold:
            centroid = shape(geom).centroid
            centroids.append(centroid)
            values.append(val)

    # Create a GeoDataFrame with centroids and population values
    population_centroids_gdf = gpd.GeoDataFrame(
        {'geometry': centroids, 'population': values},
        crs=src.crs
    ).to_crs(common_crs)

    print(f"Population centroids CRS (Prototyping): {population_centroids_gdf.crs}")  # Debug statement
else:
    # Original approach: Use the full-resolution population raster
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
        if val > threshold:
            centroid = shape(geom).centroid
            centroids.append(centroid)
            values.append(val)

    # Create a GeoDataFrame with centroids and population values
    population_centroids_gdf = gpd.GeoDataFrame(
        {'geometry': centroids, 'population': values},
        crs=src.crs
    ).to_crs(common_crs)

    print(f"Population centroids CRS (Original): {population_centroids_gdf.crs}")  # Debug statement

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
def find_shortest_paths(network_graph, pop_centroid_nodes=None, facility_nodes=None, num_centroids=None):
    shortest_paths = []
    if pop_centroid_nodes is None:
        pop_centroid_nodes = [node for node, data in network_graph.nodes(data=True) if data['type'] == 'pop_centroid']
    if facility_nodes is None:
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

# Before process_energy_supply call, add initial statistics
print("\n<Before Processing>")
total_available_mwh = energy_facilities_proj['Annual_MWh'].sum()
total_population = population_centroids_proj['population'].sum()
num_centroids = len(population_centroids_proj)
print(f"1. Total available MWh from Facilities: {total_available_mwh:,.2f}")
print(f"2. Total population: {total_population:,.0f}")
print(f"   Number of Centroids: {num_centroids}")

# Update the shortest paths call
shortest_paths = find_shortest_paths(network_graph)  # Remove the num_centroids limit

# Update shortest_paths_gdf creation
shortest_paths_gdf = gpd.GeoDataFrame(
    {
        "geometry": [LineString([Point(coord) for coord in path]) for _, _, path, _ in shortest_paths],
        "distance": [distance for _, _, _, distance in shortest_paths],
        "facility_gem_id": [facility_gem_ids.get(facility, 'unknown') for _, facility, _, _ in shortest_paths],
        "facility_type": [get_facility_type(facility_gem_ids.get(facility, 'unknown'), energy_facilities_proj) for _, facility, _, _ in shortest_paths],
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
    # If 'Remaining_MWh' already exists, use it as the starting point
    if 'Remaining_MWh' in facilities_df.columns:
        facilities_remaining['Remaining_MWh'] = facilities_df['Remaining_MWh']
    else:
        facilities_remaining['Remaining_MWh'] = facilities_remaining['Annual_MWh']
    
    # Create base supply status DataFrame
    supply_status = gpd.GeoDataFrame({
        'geometry': centroids_df.geometry,
        'centroid_id': centroids_df.index,
        'population': centroids_df['population'],
        'needed_mwh': centroids_df['needed_mwh'],
        'filled_mwh': 0.0,
        'supply_status': 'not_filled',
        'facility_id': None,
        'facility_type': None  # Add this line
    }, crs=centroids_df.crs)
    
    # Sort paths by distance
    sorted_paths = shortest_paths_df.sort_values('distance')
    
    # Process each path
    for _, path in sorted_paths.iterrows():
        centroid_id = path['centroid_id']
        facility_id = path['facility_gem_id']
        
        # Skip if centroid is already filled
        if supply_status.loc[centroid_id, 'supply_status'] == 'filled':
            continue

        # Get needed and available MWh
        needed_mwh = float(supply_status.loc[centroid_id, 'needed_mwh'])
        facility_mask = facilities_remaining['GEM unit/phase ID'] == facility_id
        
        if not any(facility_mask):
            continue
            
        available_mwh = float(facilities_remaining.loc[facility_mask, 'Remaining_MWh'].iloc[0])
        
        if available_mwh <= 0:
            continue

        if available_mwh >= needed_mwh:
            # Fully supply the centroid
            supply_status.loc[centroid_id, 'filled_mwh'] = needed_mwh
            supply_status.loc[centroid_id, 'supply_status'] = 'filled'
            facilities_remaining.loc[facility_mask, 'Remaining_MWh'] -= needed_mwh
            supply_status.loc[centroid_id, 'facility_id'] = facility_id
            supply_status.loc[centroid_id, 'facility_type'] = facilities_df.loc[facility_mask, 'Type'].iloc[0]  # Add this line
        else:
            # Partially supply the centroid
            supply_status.loc[centroid_id, 'filled_mwh'] = available_mwh
            supply_status.loc[centroid_id, 'supply_status'] = 'partially_filled'
            facilities_remaining.loc[facility_mask, 'Remaining_MWh'] = 0
            supply_status.loc[centroid_id, 'facility_id'] = facility_id
            supply_status.loc[centroid_id, 'facility_type'] = facilities_df.loc[facility_mask, 'Type'].iloc[0]  # Add this line
    
    return supply_status, facilities_remaining

# Ensure that 'energy_supply_status' includes the 'facility_id'
energy_supply_status, facilities_remaining = process_energy_supply(
    shortest_paths_gdf,
    energy_facilities_proj,
    population_centroids_proj
)

# Filter for supplied paths in first round
supplied_shortest_paths_gdf = shortest_paths_gdf[
    shortest_paths_gdf['centroid_id'].isin(
        energy_supply_status[
            energy_supply_status['supply_status'].isin(['filled', 'partially_filled'])
        ]['centroid_id']
    )
]

# After first round process_energy_supply, fix the order of calculations
energy_facilities_proj['Supplied_MWh'] = energy_facilities_proj.apply(
    lambda x: energy_supply_status[
        energy_supply_status['facility_id'] == x['GEM unit/phase ID']
    ]['filled_mwh'].sum(),
    axis=1
)
energy_facilities_proj['Remaining_MWh'] = energy_facilities_proj['Annual_MWh'] - energy_facilities_proj['Supplied_MWh']

# Now print the statistics
print("\n<After first round energy supply processing>")
total_supplied_mwh = energy_facilities_proj['Supplied_MWh'].sum()
total_remaining_mwh = energy_facilities_proj['Remaining_MWh'].sum()
filled_centroids = energy_supply_status[energy_supply_status['supply_status'] == 'filled']
supplied_population = filled_centroids['population'].sum()
num_supplied_centroids = len(filled_centroids)
print(f"1. Total Supplied MWh: {total_supplied_mwh:,.2f}")
print(f"   Remaining MWh: {total_remaining_mwh:,.2f}")
print(f"2. Supplied Population: {supplied_population:,.0f}")
print(f"   Number of Supplied Centroids: {num_supplied_centroids}")

# After first round energy supply mapping, add detailed print statements
print("\n<After first round energy supply processing>")
print("Facility level statistics:")
for _, facility in energy_facilities_proj.iterrows():
    facility_id = facility['GEM unit/phase ID']
    annual_mwh = facility['Annual_MWh']
    supplied_mwh = facility['Supplied_MWh']
    remaining_mwh = facility['Remaining_MWh']
    print(f"Facility {facility_id}:")
    print(f"  Annual MWh: {annual_mwh:,.2f}")
    print(f"  Supplied MWh: {supplied_mwh:,.2f}")
    print(f"  Remaining MWh: {remaining_mwh:,.2f}")

print("\n<Data pre-processing for second round: connecting components>")
# 4.5 Second Round of Energy Supply Mapping
# Create filtered copy of the network graph
network_graph_2 = network_graph.copy()

# Filter nodes for second round of path finding
unfilled_centroid_ids = energy_supply_status[energy_supply_status['supply_status'] != 'filled']['centroid_id']
unfilled_centroids_proj = population_centroids_proj[population_centroids_proj['centroid_id'].isin(unfilled_centroid_ids)]
remaining_facilities = facilities_remaining[facilities_remaining['Remaining_MWh'] > 0]
remaining_facilities_proj = energy_facilities_proj[energy_facilities_proj['GEM unit/phase ID'].isin(remaining_facilities['GEM unit/phase ID'])]

# Get coordinates for filtered nodes
unfilled_centroid_nodes = [(point.x, point.y) for point in unfilled_centroids_proj.geometry]
facilities_remaining_nodes = [(point.x, point.y) for point in remaining_facilities_proj.geometry]

# Remove nodes that aren't grid nodes, unfilled centroids, or facilities with remaining energy
nodes_to_keep = set()
for node, data in network_graph_2.nodes(data=True):
    if (data['type'] == 'grid_line' or
        (data['type'] == 'pop_centroid' and node in unfilled_centroid_nodes) or
        (data['type'] == 'facility' and node in facilities_remaining_nodes)):
        nodes_to_keep.add(node)

# Remove nodes not in nodes_to_keep
nodes_to_remove = set(network_graph_2.nodes()) - nodes_to_keep
network_graph_2.remove_nodes_from(nodes_to_remove)

# Identify connected components before connecting
components_before = list(nx.connected_components(network_graph_2))
num_components_before = len(components_before)
print(f"Number of connected components before connecting: {num_components_before}")

# Convert component sets to lists for indexing
components = [list(comp) for comp in components_before]

# Initialize a list to store edges to be added
edges_to_add = []

# For each pair of components, find the shortest connecting edge
for i in range(len(components)):
    for j in range(i + 1, len(components)):
        component_1 = components[i]
        component_2 = components[j]

        min_distance = float('inf')
        closest_pair = None

        # Find the closest pair of nodes between components
        for node1 in component_1:
            for node2 in component_2:
                distance = Point(node1).distance(Point(node2))
                if distance < min_distance:
                    min_distance = distance
                    closest_pair = (node1, node2)

        # If the distance is less than 10 km, add an edge
        if min_distance < 10000:  # 10 km in meters
            edges_to_add.append((closest_pair[0], closest_pair[1], {'weight': min_distance, 'edge_type': 'added'}))
            print(f"Connecting components {i} and {j} with edge length {min_distance:.2f} meters")

# Add the new edges to the graph with 'edge_type' attribute
network_graph_2.add_edges_from(edges_to_add)

# Recalculate connected components after connecting
components_after = list(nx.connected_components(network_graph_2))
num_components_after = len(components_after)
print(f"Number of connected components after connecting: {num_components_after}")

# Calculate number of components still unconnected
num_unconnected_components = num_components_after if num_components_after > 1 else 0
print(f"Number of still unconnected components after connecting: {num_unconnected_components}")

# Add new function for second round path finding
def find_all_facility_paths(network_graph, pop_centroid_nodes, facility_nodes):
    all_paths = []
    print(f"Processing {len(pop_centroid_nodes)} centroids and {len(facility_nodes)} facilities...")
    
    for i, centroid in enumerate(pop_centroid_nodes):
        try:
            # Find paths to all facilities
            for facility in facility_nodes:
                try:
                    path_length = nx.shortest_path_length(network_graph, centroid, facility, weight='weight')
                    path = nx.shortest_path(network_graph, centroid, facility, weight='weight')
                    all_paths.append((centroid, facility, path, path_length))
                except nx.NetworkXNoPath:
                    continue
            
            if i % 500 == 0:  # Progress update
                print(f"Processed {i+1}/{len(pop_centroid_nodes)} centroids")
        except Exception as e:
            print(f"Error finding paths for centroid {centroid}: {e}")
    
    print(f"Successfully found {len(all_paths)} paths")
    return all_paths

# Replace the second round path finding with all-facilities approach
second_shortest_paths = find_all_facility_paths(
    network_graph_2,
    pop_centroid_nodes=unfilled_centroid_nodes,
    facility_nodes=facilities_remaining_nodes
)

# Convert second shortest paths to GeoDataFrame (modified to handle multiple paths per centroid)
second_shortest_paths_gdf = gpd.GeoDataFrame(
    {
        "geometry": [LineString([Point(coord) for coord in path]) for _, _, path, _ in second_shortest_paths],
        "distance": [distance for _, _, _, distance in second_shortest_paths],
        "facility_gem_id": [facility_gem_ids.get(facility, 'unknown') for _, facility, _, _ in second_shortest_paths],
        "facility_type": [get_facility_type(facility_gem_ids.get(facility, 'unknown'), energy_facilities_proj) for _, facility, _, _ in second_shortest_paths],
        "centroid_coord": [centroid for centroid, _, _, _ in second_shortest_paths]
    },
    crs=PROJECTED_CRS
)

# Add centroid IDs by matching coordinates with unfilled_centroids_proj
second_shortest_paths_gdf['centroid_id'] = second_shortest_paths_gdf.apply(
    lambda row: unfilled_centroids_proj[
        unfilled_centroids_proj.geometry.apply(
            lambda point: (point.x, point.y) == row['centroid_coord']
        )
    ].index[0],
    axis=1
)

# Sort paths by distance to prioritize shorter connections
second_shortest_paths_gdf = second_shortest_paths_gdf.sort_values('distance')

# Update before second energy supply processing
unfilled_centroids_proj = population_centroids_proj[population_centroids_proj['centroid_id'].isin(unfilled_centroid_ids)].copy()

# Get the filled_mwh from first round energy supply status
first_round_filled = energy_supply_status[['centroid_id', 'needed_mwh', 'filled_mwh']].copy()
first_round_filled.rename(columns={'needed_mwh': 'needed_mwh_first_round'}, inplace=True)

# Update unfilled_centroids_proj with remaining needed_mwh
unfilled_centroids_proj = unfilled_centroids_proj.merge(
    first_round_filled,
    left_index=True,
    right_on='centroid_id',
    how='left'
)
unfilled_centroids_proj['needed_mwh'] = unfilled_centroids_proj['needed_mwh_first_round'] - unfilled_centroids_proj['filled_mwh']

# Process energy supply mapping with second shortest paths
second_energy_supply_status, second_facilities_remaining = process_energy_supply(
    second_shortest_paths_gdf,
    remaining_facilities,
    unfilled_centroids_proj
)

# Rename needed_mwh in second_energy_supply_status for clarity
second_energy_supply_status.rename(columns={'needed_mwh': 'needed_mwh_after_first_processing'}, inplace=True)
# Create 'second_supplied_shortest_paths_gdf' for supplied paths in the second run
second_supplied_shortest_paths_gdf = second_shortest_paths_gdf[
    second_shortest_paths_gdf['centroid_id'].isin(
        second_energy_supply_status[
            second_energy_supply_status['supply_status'].isin(['filled', 'partially_filled'])
        ]['centroid_id']
    )
]

# Create a GeoDataFrame for the edges of the second network
second_graph_edges = []
for u, v, data in network_graph_2.edges(data=True):
    edge_geom = LineString([Point(u), Point(v)])
    edge_type = data.get('edge_type', 'existing')
    second_graph_edges.append({'geometry': edge_geom, 'edge_type': edge_type})

second_graph_edges_gdf = gpd.GeoDataFrame(second_graph_edges, crs=PROJECTED_CRS)

# Create a GeoDataFrame for the nodes of the second network
second_graph_nodes = []
for node, data in network_graph_2.nodes(data=True):
    node_geom = Point(node)
    node_type = data.get('type', 'unknown')
    second_graph_nodes.append({'geometry': node_geom, 'type': node_type})

second_graph_nodes_gdf = gpd.GeoDataFrame(second_graph_nodes, crs=PROJECTED_CRS)

# After second round process_energy_supply, calculate Second_supplied_MWh and Second_remaining_MWh
energy_facilities_proj['Second_supplied_MWh'] = energy_facilities_proj.apply(
    lambda x: second_energy_supply_status[
        second_energy_supply_status['facility_id'] == x['GEM unit/phase ID']
    ]['filled_mwh'].sum(),
    axis=1
)
energy_facilities_proj['Second_remaining_MWh'] = energy_facilities_proj['Remaining_MWh'] - energy_facilities_proj['Second_supplied_MWh']

print("\n<After second round energy supply processing>")
total_supplied_mwh_second = energy_facilities_proj['Second_supplied_MWh'].sum()
total_remaining_mwh_final = energy_facilities_proj['Second_remaining_MWh'].sum()
filled_centroids_second = second_energy_supply_status[second_energy_supply_status['supply_status'] == 'filled']
supplied_population_second = filled_centroids_second['population'].sum()
num_supplied_centroids_second = len(filled_centroids_second)
print(f"1. Second Round Supplied MWh: {total_supplied_mwh_second:,.2f}")
print(f"   Final Remaining MWh: {total_remaining_mwh_final:,.2f}")
print(f"2. Supplied Population: {supplied_population_second:,.0f}")
print(f"   Number of Supplied Centroids: {num_supplied_centroids_second}")

# After second round energy supply mapping, add detailed print statements
print("\n<After second round energy supply processing>")
print("Facility level statistics:")
for _, facility in energy_facilities_proj.iterrows():
    facility_id = facility['GEM unit/phase ID']
    annual_mwh = facility['Annual_MWh']
    first_supplied = facility['Supplied_MWh']
    remaining_first = facility['Remaining_MWh']
    second_supplied = facility['Second_supplied_MWh']
    final_remaining = facility['Second_remaining_MWh']
    print(f"Facility {facility_id}:")
    print(f"  Annual MWh: {annual_mwh:,.2f}")
    print(f"  First Round Supplied: {first_supplied:,.2f}")
    print(f"  Remaining after First Round: {remaining_first:,.2f}")
    print(f"  Second Round Supplied: {second_supplied:,.2f}")
    print(f"  Final Remaining: {final_remaining:,.2f}")

# 5. Data Export
# 5.1 Prepare layers for export
layer_data = {
    "energy_facilities": energy_facilities_proj.to_crs(common_crs),
    "grid_lines": grid_lines_proj.to_crs(common_crs),
    "population_centroids": population_centroids_proj.to_crs(common_crs),
    "voronoi_polygons": voronoi_gdf.to_crs(common_crs),
    "network_nodes": graph_nodes.to_crs(common_crs),
    "network_edges": graph_edges.to_crs(common_crs),
    "shortest_paths": shortest_paths_gdf.to_crs(common_crs),
    "supplied_shortest_paths": supplied_shortest_paths_gdf.to_crs(common_crs),  # First round supplied paths
    "first_energy_supply_status": energy_supply_status.to_crs(common_crs),
    "second_network_nodes": second_graph_nodes_gdf.to_crs(common_crs),
    "second_network_edges": second_graph_edges_gdf.to_crs(common_crs),
    "second_shortest_paths": second_shortest_paths_gdf.to_crs(common_crs),
    "second_supplied_shortest_paths": second_supplied_shortest_paths_gdf.to_crs(common_crs),
    "second_energy_supply_status": second_energy_supply_status.to_crs(common_crs),
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

# 5.2 Export to GeoPackage
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

end_time = time.time()
execution_time = end_time - start_time
print(f"\nTotal execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
