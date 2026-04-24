import os
import json
import pandas as pd

base = r"C:\Users\jjh68\dphil_p1\p1_test"
files = [
    r"bigdata_solar_cmip6/outputs/PVOUT_baseline_300arcsec.parquet",
    r"bigdata_solar_cmip6/outputs/PVOUT_2030_300arcsec.parquet",
    r"bigdata_solar_cmip6/outputs/PVOUT_2050_300arcsec.parquet",
    r"bigdata_solar_cmip6/outputs/PVOUT_DELTA_2030_300arcsec.parquet",
    r"bigdata_solar_cmip6/outputs/PVOUT_DELTA_2050_300arcsec.parquet",
    r"bigdata_solar_cmip6/outputs/PVOUT_UNCERTAINTY_2030_300arcsec.parquet",
    r"bigdata_solar_cmip6/outputs/PVOUT_UNCERTAINTY_2050_300arcsec.parquet",
    r"bigdata_wind_cmip6/outputs/WPD100_baseline_300arcsec.parquet",
    r"bigdata_wind_cmip6/outputs/WPD100_2030_300arcsec.parquet",
    r"bigdata_wind_cmip6/outputs/WPD100_2050_300arcsec.parquet",
    r"bigdata_wind_cmip6/outputs/WPD100_DELTA_2030_300arcsec.parquet",
    r"bigdata_wind_cmip6/outputs/WPD100_DELTA_2050_300arcsec.parquet",
    r"bigdata_wind_cmip6/outputs/WPD100_UNCERTAINTY_2030_300arcsec.parquet",
    r"bigdata_wind_cmip6/outputs/WPD100_UNCERTAINTY_2050_300arcsec.parquet",
    r"bigdata_hydro_cmip6/outputs/HYDRO_RUNOFF_baseline_300arcsec.parquet",
    r"bigdata_hydro_cmip6/outputs/HYDRO_RUNOFF_2030_300arcsec.parquet",
    r"bigdata_hydro_cmip6/outputs/HYDRO_RUNOFF_2050_300arcsec.parquet",
    r"bigdata_hydro_cmip6/outputs/HYDRO_RUNOFF_DELTA_2030_300arcsec.parquet",
    r"bigdata_hydro_cmip6/outputs/HYDRO_RUNOFF_DELTA_2050_300arcsec.parquet",
    r"bigdata_hydro_cmip6/outputs/HYDRO_RUNOFF_UNCERTAINTY_2030_300arcsec.parquet",
    r"bigdata_hydro_cmip6/outputs/HYDRO_RUNOFF_UNCERTAINTY_2050_300arcsec.parquet",
]

for rel in files:
    path = os.path.join(base, rel)
    print(f"FILE: {rel}")
    if not os.path.exists(path):
        print("STATUS: MISSING")
        print("---")
        continue
    try:
        df = pd.read_parquet(path)
        print("COLUMNS:", json.dumps(list(df.columns), ensure_ascii=False))
        if len(df) > 0:
            row = df.iloc[0].to_dict()
            print("SAMPLE_ROW:", json.dumps(row, ensure_ascii=False, default=str))
        else:
            print("SAMPLE_ROW: <EMPTY_FILE>")
    except Exception as e:
        print("STATUS: ERROR")
        print("ERROR:", repr(e))
    print("---")
