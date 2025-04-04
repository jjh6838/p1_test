import arcpy

# Paths to your GeoPackage and geodatabase
gpkg_path = r"C:\Users\jjh68\dphil_p1\p1_arcgis\KOR\gridfinder\grid__KOR.gpkg"
geodatabase = r"C:\Users\jjh68\dphil_p1\p1_arcgis\KOR\gridfinder\grid__KOR.gdb"

# Name of the feature dataset within the geodatabase
feature_dataset = "EnergyNetwork"

# Set the workspace to your geodatabase
arcpy.env.workspace = geodatabase

# Create the feature dataset if it doesn't exist
if not arcpy.Exists(feature_dataset):
    spatial_ref = arcpy.SpatialReference(4326)  # Replace with your spatial reference
    arcpy.CreateFeatureDataset_management(geodatabase, feature_dataset, spatial_ref)

# List all feature classes in the GeoPackage
arcpy.env.workspace = gpkg_path
feature_classes = arcpy.ListFeatureClasses()

# Import each feature class into the geodatabase
for fc in feature_classes:
    input_fc = gpkg_path + "\\" + fc
    output_fc = geodatabase + "\\" + feature_dataset + "\\" + fc
    arcpy.FeatureClassToFeatureClass_conversion(input_fc, geodatabase + "\\" + feature_dataset, fc)

print("Import completed.")
