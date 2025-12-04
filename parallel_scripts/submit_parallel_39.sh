#!/bin/bash --login
#SBATCH --job-name=p39_t5
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=28G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --output=outputs_global/logs/parallel_39_%j.out
#SBATCH --error=outputs_global/logs/parallel_39_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 39/40 (T5) at $(date)"
echo "[INFO] Processing 10 countries in this batch: TGO, THA, TJK, TKM, TLS, TON, TTO, TUN, TWN, TZA"
echo "[INFO] Tier: T5 | Memory: 28G | CPUs: 40 | Time: 12:00:00"

# --- directories ---
mkdir -p outputs_per_country outputs_global outputs_global/logs

# --- Conda bootstrap ---
export PATH=/soge-home/users/lina4376/miniconda3/bin:$PATH
source /soge-home/users/lina4376/miniconda3/etc/profile.d/conda.sh

conda --version
conda activate p1_etl || true

# Use the env's absolute python path to avoid activation issues in batch shells
PY=/soge-home/users/lina4376/miniconda3/envs/p1_etl/bin/python
echo "[INFO] Using Python: $PY"
$PY -c 'import sys; print(sys.executable)'

# Process countries in this batch

echo "[INFO] Processing TGO (T5)..."
if $PY process_country_supply.py TGO --output-dir outputs_per_country; then
    echo "[SUCCESS] TGO completed"
else
    echo "[ERROR] TGO failed"
fi

echo "[INFO] Processing THA (T5)..."
if $PY process_country_supply.py THA --output-dir outputs_per_country; then
    echo "[SUCCESS] THA completed"
else
    echo "[ERROR] THA failed"
fi

echo "[INFO] Processing TJK (T5)..."
if $PY process_country_supply.py TJK --output-dir outputs_per_country; then
    echo "[SUCCESS] TJK completed"
else
    echo "[ERROR] TJK failed"
fi

echo "[INFO] Processing TKM (T5)..."
if $PY process_country_supply.py TKM --output-dir outputs_per_country; then
    echo "[SUCCESS] TKM completed"
else
    echo "[ERROR] TKM failed"
fi

echo "[INFO] Processing TLS (T5)..."
if $PY process_country_supply.py TLS --output-dir outputs_per_country; then
    echo "[SUCCESS] TLS completed"
else
    echo "[ERROR] TLS failed"
fi

echo "[INFO] Processing TON (T5)..."
if $PY process_country_supply.py TON --output-dir outputs_per_country; then
    echo "[SUCCESS] TON completed"
else
    echo "[ERROR] TON failed"
fi

echo "[INFO] Processing TTO (T5)..."
if $PY process_country_supply.py TTO --output-dir outputs_per_country; then
    echo "[SUCCESS] TTO completed"
else
    echo "[ERROR] TTO failed"
fi

echo "[INFO] Processing TUN (T5)..."
if $PY process_country_supply.py TUN --output-dir outputs_per_country; then
    echo "[SUCCESS] TUN completed"
else
    echo "[ERROR] TUN failed"
fi

echo "[INFO] Processing TWN (T5)..."
if $PY process_country_supply.py TWN --output-dir outputs_per_country; then
    echo "[SUCCESS] TWN completed"
else
    echo "[ERROR] TWN failed"
fi

echo "[INFO] Processing TZA (T5)..."
if $PY process_country_supply.py TZA --output-dir outputs_per_country; then
    echo "[SUCCESS] TZA completed"
else
    echo "[ERROR] TZA failed"
fi

echo "[INFO] Batch 39/40 (T5) completed at $(date)"
