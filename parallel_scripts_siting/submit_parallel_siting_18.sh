#!/bin/bash --login
#SBATCH --job-name=p18s_t3
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=25G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/siting_18_%j.out
#SBATCH --error=outputs_global/logs/siting_18_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting siting analysis script 18/24 (T3) at $(date)"
echo "[INFO] Processing 10 countries in this batch: MNE, MNG, MOZ, MRT, MUS, MWI, MYS, NAM, NCL, NER"
echo "[INFO] Tier: T3 | Memory: 25G | CPUs: 40 | Time: 12:00:00"

# --- directories ---
mkdir -p outputs_per_country outputs_global outputs_global/logs

# --- Conda bootstrap ---
export PATH=/soge-home/users/lina4376/miniconda3/bin:$PATH
source /soge-home/users/lina4376/miniconda3/etc/profile.d/conda.sh

conda --version
conda activate p1_etl || true

# Use the env's absolute python path
PY=/soge-home/users/lina4376/miniconda3/envs/p1_etl/bin/python
echo "[INFO] Using Python: $PY"
$PY -c 'import sys; print(sys.executable)'

# Process countries in this batch

echo "[INFO] Processing siting analysis for MNE (T3)..."
if $PY process_country_siting.py MNE; then
    echo "[SUCCESS] MNE siting analysis completed"
else
    echo "[ERROR] MNE siting analysis failed"
fi

echo "[INFO] Processing siting analysis for MNG (T3)..."
if $PY process_country_siting.py MNG; then
    echo "[SUCCESS] MNG siting analysis completed"
else
    echo "[ERROR] MNG siting analysis failed"
fi

echo "[INFO] Processing siting analysis for MOZ (T3)..."
if $PY process_country_siting.py MOZ; then
    echo "[SUCCESS] MOZ siting analysis completed"
else
    echo "[ERROR] MOZ siting analysis failed"
fi

echo "[INFO] Processing siting analysis for MRT (T3)..."
if $PY process_country_siting.py MRT; then
    echo "[SUCCESS] MRT siting analysis completed"
else
    echo "[ERROR] MRT siting analysis failed"
fi

echo "[INFO] Processing siting analysis for MUS (T3)..."
if $PY process_country_siting.py MUS; then
    echo "[SUCCESS] MUS siting analysis completed"
else
    echo "[ERROR] MUS siting analysis failed"
fi

echo "[INFO] Processing siting analysis for MWI (T3)..."
if $PY process_country_siting.py MWI; then
    echo "[SUCCESS] MWI siting analysis completed"
else
    echo "[ERROR] MWI siting analysis failed"
fi

echo "[INFO] Processing siting analysis for MYS (T3)..."
if $PY process_country_siting.py MYS; then
    echo "[SUCCESS] MYS siting analysis completed"
else
    echo "[ERROR] MYS siting analysis failed"
fi

echo "[INFO] Processing siting analysis for NAM (T3)..."
if $PY process_country_siting.py NAM; then
    echo "[SUCCESS] NAM siting analysis completed"
else
    echo "[ERROR] NAM siting analysis failed"
fi

echo "[INFO] Processing siting analysis for NCL (T3)..."
if $PY process_country_siting.py NCL; then
    echo "[SUCCESS] NCL siting analysis completed"
else
    echo "[ERROR] NCL siting analysis failed"
fi

echo "[INFO] Processing siting analysis for NER (T3)..."
if $PY process_country_siting.py NER; then
    echo "[SUCCESS] NER siting analysis completed"
else
    echo "[ERROR] NER siting analysis failed"
fi

echo "[INFO] Siting batch 18/24 (T3) completed at $(date)"
