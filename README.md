# Parallel Supply Analysis — Cluster & Local Execution

Process all countries via 40 parallel SLURM jobs (cluster) or locally with automatic CPU detection.

## Status: ✅ Ready for Execution
- **Cluster**: 40 parallel SLURM scripts with tiered batching (1/2/4/8 countries per job)
- **Local**: Automatic core detection (8/16/72 cores), per-country execution
- **Internal parallelization**: Grid processing, facility distance calculations utilize all available cores

---

## Quick Start: Cluster Execution (Recommended for All Countries)

```bash
# 1) Generate 40 parallel scripts
python get_countries.py --create-parallel

# 2) Fix line endings & make executable (Linux/cluster)
sed -i 's/\r$//' submit_all_parallel.sh parallel_scripts/*.sh
chmod +x submit_all_parallel.sh parallel_scripts/*.sh

# 3) Submit all 40 parallel jobs (in waves of 8)
./submit_all_parallel.sh

# 4) Monitor progress
squeue -u $USER
tail -f outputs_global/logs/parallel_*.out

# 5) Combine results when jobs complete
sbatch submit_workflow.sh
```

**Key Files:**
- `submit_all_parallel.sh` - Submits all 40 jobs in waves of 8 (respects per-user limits)
- `parallel_scripts/submit_parallel_01.sh` through `submit_parallel_40.sh` - Individual job scripts
- `submit_workflow.sh` - Combination step (runs after parallel jobs complete)

**Tips:**
- Use `./script.sh` on Linux/macOS when executing from current directory
- Run `sed -i 's/\r$//'` on cluster if files prepared on Windows
- Each job automatically uses 64 CPUs via internal parallelization
- **Wave-based submission**: Script waits when 8 jobs running, submits next when slots open
- **Respects cluster limits**: Prevents queue congestion, better cluster etiquette

---

## Alternative: Local Execution (Small Datasets or Testing)

```bash
# Single country (auto-detects cores: 8/16/72)
conda activate p1_etl
python process_country_supply.py KOR

# Multiple countries (serial)
python process_country_supply.py USA CHN IND

# Combine results
python combine_global_results.py --input-dir outputs_per_country
```

**Performance (automatic scaling):**
- **Korea**: 20-25 mins (16-core laptop) | 35-45 mins (8-core laptop) | 15 mins (72-core cluster)
- **Japan**: 1-1.5 hours (16-core) | 2-3 hours (8-core) | <30 mins (72-core)
- **China**: 2-3 hours (16-core) | 4-6 hours (8-core) | <30 mins (72-core)

---

## Shell Scripts Overview

### **Master Submission Script**
- **`submit_all_parallel.sh`** - Submits all 40 jobs in controlled waves
  - Respects per-user job limits (max 8 simultaneous)
  - Monitors job count with `squeue -u $USER`
  - Waits for slots before submitting next wave
  - Prevents queue congestion, better cluster etiquette
  - Submission takes ~30-60 minutes, but ensures steady execution

### **Individual Job Scripts** (`parallel_scripts/submit_parallel_*.sh`)
40 scripts with tiered country allocation:

| Script Range | Countries/Job | Country Tier | Examples |
|--------------|---------------|--------------|----------|
| `01-07` | 1 | Tier 1 (Largest) | USA, CHN, IND, RUS, BRA, CAN, AUS |
| `08-19` | 2 | Tier 2 (Large) | ARG+KAZ, DZA+MEX, IDN+SDN, etc. |
| `20-33` | 4 | Tier 3 (Medium) | KOR+FRA+DEU+JPN, GBR+ESP+THA+VNM, etc. |
| `34-40` | 8 | Other (Small) | Island nations, small countries |

**Each script:**
- Requests 64 CPUs, 1024GB RAM (16GB/core)
- Partition assignment by tier:
  - **Tier 1** (7 scripts): Long partition, 168h time limit
  - **Tier 2** (~6 scripts): Medium partition, 48h time limit  
  - **Tier 3 + Other** (~27 scripts): Short partition, 12h time limit
- Activates conda environment: `conda activate p1_etl`
- Processes assigned countries: `python process_country_supply.py {ISO3}`
- Outputs to: `outputs_per_country/parquet/{scenario}/{layer}_{ISO3}.parquet`
  - Example: `outputs_per_country/parquet/2050_supply_100%/centroids_KOR.parquet`

### **Combination Script**
- **`submit_workflow.sh`** - Combines all country results into global outputs
  - **When to run**: After all 40 parallel jobs complete
  - **Resources**: 72 CPUs, 1152GB RAM
  - **Partition**: Short (12h sufficient for combination)
  - **What it does**: Runs `combine_global_results.py` to merge parquet files
  - **Auto-detects scenarios**: Scans `outputs_per_country/parquet/` for scenario subfolders
  - **Outputs**: `outputs_global/{scenario}_global.gpkg` (one per scenario)
    - Example: `outputs_global/2050_supply_100%_global.gpkg`


---

## Internal Parallelization (Automatic)

**Key Enhancement**: All operations now utilize available cores automatically

### **Automatic Core Detection**
```python
MAX_WORKERS = min(72, max(1, os.cpu_count() or 1))
# Result: 72 on cluster, 16 on laptop, 8 on older laptop
```

### **Parallelized Operations**
1. **Grid line splitting** (>1,000 lines)
   - Divides into batches: `max(50, num_lines // (MAX_WORKERS * 2))`
   - Processes batches in parallel with all cores
   
2. **Grid chunked processing** (>10,000 lines)
   - Creates chunks for spatial processing
   - Parallel execution with ThreadPoolExecutor
   
3. **Facility distance calculations** (>20 facilities, >1,000 centroids)
   - Divides facilities into batches: `len(facilities) // MAX_WORKERS`
   - Each batch runs Dijkstra pathfinding independently
   - Progress reporting every 10%

### **Smart Thresholds**
- Small datasets use serial processing (avoids threading overhead)
- Medium datasets use moderate parallelization
- Large datasets (Japan, China, USA) use full parallelization

**Performance Impact:**
- Korea: 8h+ → 15 mins (72-core cluster)
- Japan: 2.5h+ → <30 mins target (72-core cluster)
- China: Expected <30 mins (72-core cluster)

---

## Multi-threading Configuration

### **Environment Variables** (set automatically)
```bash
OMP_NUM_THREADS = allocated_cpus
MKL_NUM_THREADS = allocated_cpus  
OPENBLAS_NUM_THREADS = allocated_cpus
NUMEXPR_NUM_THREADS = allocated_cpus
```

### **No Manual Configuration Needed**
- Cluster: Automatically uses 72 cores per job
- Laptop: Automatically detects 8/16 cores
- Threading overhead avoided for small datasets

---

## Cluster Resources & Performance

### **Tiered Resource Allocation**

All jobs request full nodes for maximum performance:

| Country Tier | CPUs/Job | Memory | Partition | Time Limit | Countries/Job | Examples |
|--------------|----------|--------|-----------|------------|---------------|----------|
| **Tier 1** | 64 | 1024GB | Long | 168h (7 days) | 1 | USA, CHN, IND, RUS, BRA, CAN, AUS |
| **Tier 2** | 64 | 1024GB | Medium | 48h (2 days) | 2 | ARG+KAZ, DZA+MEX, IDN+SDN, LBY+SAU |
| **Tier 3** | 64 | 1024GB | Short | 12h | 4 | KOR+FRA+DEU+JPN, GBR+ESP+THA+VNM |
| **Other** | 64 | 1024GB | Short | 12h | 8 | Small island nations |

**Partition Strategy:**
- **Long** (few slots): Reserved for 7 largest countries requiring maximum time
- **Medium** (some slots): Used for ~6 large countries needing 2 days
- **Short** (many slots): Handles ~27 medium/small countries completing in hours

**Combination Step:**
- Memory: 1152GB (72 CPUs × 16GB/core)
- Partition: Short (12h is sufficient for combination)
- CPUs: 72

### **Expected Performance (64-core cluster)**
- **Tier 1 countries** (Long partition): 15-30 minutes, but 7 days available if needed
- **Tier 2 countries** (Medium partition): 10-20 minutes, 2 days available
- **Tier 3 countries** (Short partition): 5-15 minutes, completes well within 12h
- **Small countries** (Short partition): <5 minutes, completes within 12h

**Laptop Performance (16 cores):**
- Small: <10 minutes
- Medium: 20-30 minutes
- Large: 1-2 hours
- Very large: 2-3 hours

### **Performance Monitoring**
```bash
# Check CPU utilization during job
sstat -j <JOB_ID> --format=JobID,MaxRSS,AveCPU

# Check detailed resource usage
sacct -j <JOB_ID> --format=JobID,JobName,MaxRSS,Elapsed,CPUTime,CPUTimeRAW

# Monitor all parallel jobs
squeue -u $USER --format="%.10i %.9P %.20j %.8u %.8T %.10M %.6D %R"

# Check logs for parallelization messages
tail -f outputs_global/logs/parallel_01.out | grep "Using parallel processing"
```

---

## Troubleshooting

### **Fixed Issues ✅**
- Environment conflicts resolved in `environment.yml`
- Conda activation fixed in all 40 parallel scripts
- Unicode encoding fixed for Windows/Linux compatibility
- Geographic CRS warnings fixed with proper UTM projections
- Internal parallelization implemented (grid processing, facility distances)
- Automatic core detection for cluster/laptop execution

### **Common Issues**
1. **"No completed countries found"** (when running `submit_workflow.sh`)
   - Ensure all parallel jobs completed: `squeue -u $USER`
   - Check for parquet files: `find outputs_per_country -name "*.parquet"`
   - Review job logs: `tail outputs_global/logs/parallel_*.out`

2. **Re-running jobs overwrites existing data**
   - Existing parquet files are **overwritten** without backup
   - To preserve previous results: `mv outputs_per_country outputs_per_country_backup`
   - Or rename scenario subfolder: `mv outputs_per_country/parquet/2050_supply_100% outputs_per_country/parquet/2050_supply_100%_v1`
   - Partial re-runs create mixed datasets (old + new data)

3. **Line ending errors** (`'\r': command not found`)
   - Run on cluster: `sed -i 's/\r$//' submit_all_parallel.sh parallel_scripts/*.sh`
   - Or use Git: `git config core.autocrlf true`

4. **Permission denied**
   - Run: `chmod +x submit_all_parallel.sh parallel_scripts/*.sh`

5. **Memory errors** (rare with 1024GB allocation)
   - Check job output: `sacct -j <JOB_ID> --format=JobID,MaxRSS`

6. **Time limit exceeded**
   - Tier 1 (Long partition): 168h should be sufficient for all countries
   - Tier 2/3 (Medium/Short): If job exceeds time, consider moving to higher tier
   - Check logs to identify which country took longest

7. **Missing countries in output**
   - Workflow skips countries without GADM boundaries
   - Check `countries_list.txt` for expected countries
   - Some islands may have no grid data (handled gracefully)

### **Performance Issues**
- **Slow execution on laptop**: Expected for large countries, consider cluster
- **Underutilized cores**: Check logs for "Using parallel processing" messages
- **Long runtimes**: Ensure latest code with internal parallelization

---

## Monitoring & Logs

### **Active Job Monitoring**
```bash
# View running jobs
squeue -u $USER

# Monitor specific job output in real-time
tail -f outputs_global/logs/parallel_01.out

# Check all parallel job logs
ls outputs_global/logs/parallel_*.out

# Search for errors across all logs
grep -i error outputs_global/logs/parallel_*.out
```

### **Log Files Generated**
- `outputs_global/logs/parallel_{01-40}.out` - Individual parallel job outputs
- `outputs_global/logs/parallel_{01-40}.err` - Individual parallel job errors
- `outputs_global/logs/test_{JOBID}.out` - Combination step output (from `submit_workflow.sh`)
- `outputs_global/logs/test_{JOBID}.err` - Combination step errors

### **Key Log Messages**
Look for these indicators of successful parallelization:
```
[CONFIG] Using MAX_WORKERS=72 for parallel operations
Using parallel processing for 1234 facilities with 72 workers...
Using parallel processing for 56789 grid lines...
Processed 500/1234 facilities...
```

---
## Cluster Execution Guide

### **Step-by-Step Workflow**

```bash
# ═══════════════════════════════════════════════════════════
# STEP 1: Prepare environment (one-time setup)
# ═══════════════════════════════════════════════════════════
conda activate p1_etl

# ═══════════════════════════════════════════════════════════
# STEP 2: Generate parallel scripts
# ═══════════════════════════════════════════════════════════
python get_countries.py --create-parallel
# Creates: parallel_scripts/submit_parallel_01.sh through 40.sh
#          submit_all_parallel.sh (master script)

# ═══════════════════════════════════════════════════════════
# STEP 3: Transfer to cluster & fix line endings
# ═══════════════════════════════════════════════════════════
# On cluster:
sed -i 's/\r$//' submit_all_parallel.sh parallel_scripts/*.sh
chmod +x submit_all_parallel.sh parallel_scripts/*.sh

# ═══════════════════════════════════════════════════════════
# STEP 4: Submit all parallel jobs (40 jobs, ~8-12 hours)
# ═══════════════════════════════════════════════════════════
./submit_all_parallel.sh

# Monitor:
squeue -u $USER
watch -n 60 'squeue -u $USER | wc -l'  # Count running jobs

# ═══════════════════════════════════════════════════════════
# STEP 5: Check completion
# ═══════════════════════════════════════════════════════════
# Count completed countries (should be 211):
find outputs_per_country -name "*.parquet" -type f | wc -l

# Check for errors:
grep -i "error\|failed" outputs_global/logs/parallel_*.out

# ═══════════════════════════════════════════════════════════
# STEP 6: Combine results (after all jobs complete)
# ═══════════════════════════════════════════════════════════
sbatch submit_workflow.sh

# Monitor combination:
squeue -u $USER
tail -f outputs_global/logs/test_*.out

# ═══════════════════════════════════════════════════════════
# STEP 7: Verify outputs
# ═══════════════════════════════════════════════════════════
ls -lh outputs_global/*_global.gpkg
# Should show one GPKG per scenario, e.g.:
#   2030_supply_100%_global.gpkg
#   2050_supply_100%_global.gpkg
```

### **Expected Timeline**
- **Parallel processing**: 8-12 hours (40 simultaneous jobs)
- **Combination**: 1-2 hours
- **Total**: ~10-14 hours for all 211 countries

---

## Development & Testing

### **Test Single Country Locally**
```bash
conda activate p1_etl
python process_country_supply.py KOR  # Should complete in 20-45 mins on laptop
```

### **Test on Cluster**
```bash
# Edit submit_parallel_01.sh to test single country
sbatch parallel_scripts/submit_parallel_01.sh

# Monitor
squeue -u $USER
tail -f outputs_global/logs/parallel_01.out
```

### **Performance Profiling**
```bash
# Check log files for timing information:
grep "Completed in" outputs_global/logs/parallel_*.out

# Extract performance metrics:
grep "Using parallel processing" outputs_global/logs/parallel_*.out
```

---

## Data Schema

Each country output contains:
- `geometry` - Point geometry for population centroid (WGS84)
- `Population_centroid` - Population at this centroid
- `Total_Demand_2030_centroid` - Energy demand for 2030 (MWh)  
- `Total_Demand_2050_centroid` - Energy demand for 2050 (MWh)
- `GID_0` - ISO3 country code
- `distance_km` - Network distance to nearest facility
- `facility_type` - Type of power plant (coal, gas, solar, etc.)
- `facility_capacity` - Capacity in MW

---

## File Structure Summary

```
├── process_country_supply.py          # Main processing script (internal parallelization)
├── get_countries.py                   # Generates parallel scripts
├── combine_global_results.py          # Combines outputs
├── submit_all_parallel.sh             # Master SLURM submission
├── submit_workflow.sh                 # Combination step submission
├── Snakefile                          # Optional Snakemake workflow (combination only)
├── parallel_scripts/
│   ├── submit_parallel_01.sh          # Job 1: AUS (Tier 1)
│   ├── submit_parallel_02.sh          # Job 2: BRA (Tier 1)
│   └── ...submit_parallel_40.sh       # Job 40: Small countries
├── outputs_per_country/
│   └── parquet/
│       ├── 2030_supply_100%/          # Scenario-specific outputs
│       │   ├── centroids_KOR.parquet
│       │   ├── facilities_KOR.parquet
│       │   └── ...
│       └── 2050_supply_100%/
│           └── ...
└── outputs_global/
    ├── logs/                          # Job logs
    ├── 2030_supply_100%_global.gpkg   # Combined outputs per scenario
    └── 2050_supply_100%_global.gpkg
```
