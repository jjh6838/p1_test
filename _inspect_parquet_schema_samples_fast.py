import os
import json
import pyarrow.parquet as pq

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

def to_jsonable(v):
    if isinstance(v, (bytes, bytearray)):
        return repr(v)
    return v

for rel in files:
    path = os.path.join(base, rel)
    print(f"FILE: {rel}")
    if not os.path.exists(path):
        print("STATUS: MISSING")
        print("---")
        continue
    try:
        pf = pq.ParquetFile(path)
        cols = pf.schema_arrow.names
        print("COLUMNS:", json.dumps(cols, ensure_ascii=False))

        sample = None
        for batch in pf.iter_batches(batch_size=1):
            if batch.num_rows > 0:
                d = batch.to_pydict()
                sample = {k: to_jsonable(v[0] if len(v) else None) for k, v in d.items()}
                break
        if sample is None:
            print("SAMPLE_ROW: <EMPTY_FILE>")
        else:
            print("SAMPLE_ROW:", json.dumps(sample, ensure_ascii=False, default=str))
    except Exception as e:
        print("STATUS: ERROR")
        print("ERROR:", repr(e))
    print("---")
