import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from tqdm import tqdm  # Progress bar
import matplotlib.pyplot as plt

# Set your common CRS (WGS84)
COMMON_CRS = "EPSG:4326"

# Load admin1 boundaries with spatial filter to reduce loading time
print("Loading admin1 boundaries...")
# Use bbox filter to load only Jeju region data
test_bbox = [124.0, 33.0, 132.0, 43.0]  # [min_lon, min_lat, max_lon, max_lat] # Korea 
admin_boundaries = gpd.read_file('bigdata_gadm/gadm_410.gpkg', bbox=test_bbox)
print(admin_boundaries.head())

# Optionally simplify geometry for faster masking (tolerance in degrees, adjust as needed)
print("Simplifying geometry...")
admin_boundaries['geometry'] = admin_boundaries['geometry'].simplify(tolerance=0.001, preserve_topology=True)

# Load only the grid features that intersect Jeju (in native CRS), then reproject
print("Loading grid data...")
grid = gpd.read_file('bigdata_gridfinder/grid.gpkg', bbox=test_bbox)

# If CRS is not EPSG:4326, reproject after filtering
if grid.crs != COMMON_CRS:
    grid = grid.to_crs(COMMON_CRS)

# Load and mask population raster 
print("Loading population raster...")
with rasterio.open('bigdata_jrc_pop/GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0.tif') as src:
    # Use bounding box to window read only the relevant part
    minx, miny, maxx, maxy = test_bbox
    window = rasterio.windows.from_bounds(minx, miny, maxx, maxy, src.transform)
    pop_data = src.read(1, window=window)
    
    # Get the transform for the windowed data
    windowed_transform = rasterio.windows.transform(window, src.transform)
    
    # Calculate centroids for each cell
    print("Calculating cell centroids...")
    rows, cols = pop_data.shape
    centroids_x = []
    centroids_y = []
    
    for row in range(rows):
        for col in range(cols):
            # Convert pixel coordinates to geographic coordinates
            x, y = rasterio.transform.xy(windowed_transform, row, col)
            centroids_x.append(x)
            centroids_y.append(y)
    
    # Create centroids GeoDataFrame
    centroids_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(centroids_x, centroids_y),
        crs=COMMON_CRS
    )
    print("All datasets matched to the same geospatial framework (CRS and extent).")

    # Calculate population per centroid (cell)
    centroids_gdf["Population_centroid"] = pop_data.flatten()

    # Calculate total population in the bbox window
    total_national_population = centroids_gdf["Population_centroid"].sum()

    # Load national demand by type from Excel
    demand_df = pd.read_excel("outputs_processed_data/p1_a_ember_2024_30.xlsx")

    # Define demand types and years
    demand_types = ["Solar", "Wind", "Hydro", "Other Renewables", "Nuclear", "Fossil"]
    years = [2024, 2030, 2050]

    # Calculate and assign total demand for each centroid for each year
    for year in years:
        # Calculate total national demand for this year
        total_national_demand = 0
        for demand_type in demand_types:
            col = f"{demand_type}_{year}_MWh"
            if col in demand_df.columns:
                total_national_demand += demand_df[col].sum()
        
        # Calculate and assign demand for each centroid based on population proportion
        demand_col = f"Total_Demand_{year}_centroid"
        centroids_gdf[demand_col] = (
            centroids_gdf["Population_centroid"] / total_national_population * total_national_demand
        )

# Filter out centroids with zero population and print the first 10 rows
centroids_filtered = centroids_gdf[centroids_gdf["Population_centroid"] > 0]
print(centroids_filtered.head(10))

# Plot the Jeju boundary, grid, population raster, and centroids
fig, ax = plt.subplots(figsize=(8, 8))

# Plot population raster
raster_show = ax.imshow(
    pop_data,
    extent=(minx, maxx, miny, maxy),
    origin='upper',
    cmap='viridis',
    alpha=0.7
)

# Plot Jeju admin boundaries
admin_boundaries.boundary.plot(ax=ax, color='red', linewidth=2, label='Jeju Admin Boundary')

# Plot grid
grid.plot(ax=ax, color='white', linewidth=0.8, alpha=0.8, label='Grid')

# Plot centroids
centroids_gdf.plot(ax=ax, color='orange', markersize=1, alpha=0.5, label='Population Centroids')

plt.colorbar(raster_show, ax=ax, label='Population Density', shrink=0.8)
plt.title("Jeju: Population Distribution, Admin Boundaries, Grid, and Centroids", fontsize=14)
plt.xlabel("Longitude", fontsize=12)
plt.ylabel("Latitude", fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()