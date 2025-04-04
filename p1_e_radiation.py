import netCDF4 as nc
import numpy as np
import rasterio
from rasterio.transform import from_origin

# Define input NetCDF file and output GeoTIFF file
nc_file_path = r'C:\Users\wb626743\dphil_p1\p1_test\cmip6\radiation_kor_2030\rsds_Amon_EC-Earth3-CC_ssp245_r1i1p1f1_gr_20300116-20301216.nc'
tiff_file_path = r'C:\Users\wb626743\dphil_p1\p1_test\cmip6\radiation_kor_2030\cmip6_radiation_kor_2030_avg.tif'

# Open NetCDF file
dataset = nc.Dataset(nc_file_path)

# Print metadata and attributes to understand the data
print("NetCDF file metadata:")
print(dataset.__dict__)
print("\nVariables and their attributes:")
for var_name in dataset.variables:
    print(f"{var_name}: {dataset.variables[var_name].__dict__}")

# Extract the variable of interest (e.g., 'rsds')
data = dataset.variables['rsds'][:]
data = np.squeeze(data)  # Remove any unnecessary dimensions

# Calculate the average over the time dimension
data_avg = np.mean(data, axis=0)

# Get latitude and longitude values
lat = dataset.variables['lat'][:]
lon = dataset.variables['lon'][:]

# Ensure latitude is in descending order for correct orientation
if lat[0] < lat[-1]:
    lat = lat[::-1]
    data_avg = data_avg[::-1, :]

# Define raster transformation (origin, pixel size)
pixel_size_x = lon[1] - lon[0]
pixel_size_y = lat[0] - lat[1]  # Note the order to ensure correct orientation
transform = from_origin(lon.min(), lat.max(), pixel_size_x, pixel_size_y)

# Check for CRS, default to WGS 84 if not available
crs_wkt = dataset.getncattr('crs') if 'crs' in dataset.ncattrs() else "EPSG:4326"

# Write the averaged data to a GeoTIFF file with metadata
with rasterio.open(
    tiff_file_path,
    'w',
    driver='GTiff',
    height=data_avg.shape[0],  # Adjusted to match the dimensions of the data
    width=data_avg.shape[1],   # Adjusted to match the dimensions of the data
    count=1,
    dtype=data_avg.dtype,
    crs=crs_wkt,
    transform=transform,
    nodata=0  # Set no-data value
) as dst:
    dst.write(data_avg, 1)

print(f"GeoTIFF successfully saved at: {tiff_file_path}")