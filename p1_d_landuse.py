import netCDF4 as nc
import numpy as np
import rasterio
from rasterio.transform import from_origin

# Define input NetCDF file and output GeoTIFF file
nc_file_path = r'C:\Users\wb626743\dphil_p1\p1_test\cds_landcover\C3S-LC-L4-LCCS-Map-300m-P1Y-2022-v2.1.1.area-subset.38.61.131.87.33.11.124.6.nc'
tiff_file_path = r'C:\Users\wb626743\dphil_p1\p1_test\cds_landcover\cds_landcover_kor.tif'

# Open NetCDF file
dataset = nc.Dataset(nc_file_path)

# Extract the land cover classification variable
data = dataset.variables['lccs_class'][:]
data = np.squeeze(data)  # Remove any unnecessary dimensions

# Get latitude and longitude values
lat = dataset.variables['lat'][:]
lon = dataset.variables['lon'][:]

# Define raster transformation (origin, pixel size)
pixel_size_x = lon[1] - lon[0]
pixel_size_y = lat[0] - lat[1]  # Note the order to ensure correct orientation
transform = from_origin(lon.min(), lat.max(), pixel_size_x, pixel_size_y)

# Check for CRS, default to WGS 84 if not available
crs_wkt = dataset.variables['crs'].wkt if 'crs' in dataset.variables else "EPSG:4326"

# Extract legend metadata (flag values, meanings, colors)
flag_values = dataset.variables['lccs_class'].flag_values
flag_meanings = dataset.variables['lccs_class'].flag_meanings.split()
flag_colors = dataset.variables['lccs_class'].flag_colors.split()

# Write to GeoTIFF with metadata
with rasterio.open(
    tiff_file_path,
    'w',
    driver='GTiff',
    height=data.shape[0],
    width=data.shape[1],
    count=1,
    dtype=data.dtype,
    crs=crs_wkt,
    transform=transform,
    nodata=0  # Set no-data value
) as dst:
    dst.write(data, 1)
    dst.update_tags(
        flag_values=",".join(map(str, flag_values.tolist())),
        flag_meanings=",".join(flag_meanings),
        flag_colors=",".join(flag_colors),
        description="Land Cover Classification - South Korea",
        source="C3S-LC-L4-LCCS, Copernicus Climate Change Service"
    )

print(f"GeoTIFF successfully saved at: {tiff_file_path}")