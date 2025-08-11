#!/bin/bash --login
#SBATCH --job-name=supply_p40_other
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=340G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --output=outputs_global/logs/parallel_40_%j.out
#SBATCH --error=outputs_global/logs/parallel_40_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "[INFO] Starting parallel script 40/40 (OTHER) at $(date)"
echo "[INFO] Processing 29 countries in this batch: RWA, SEN, SGP, SLB, SLE, SLV, SRB, SSD, STP, SUR, SVK, SVN, SWZ, SYC, SYR, TGO, TJK, TLS, TON, TTO, TUN, TWN, UGA, URY, VCT, VIR, VNM, VUT, WSM"
echo "[INFO] Tier: OTHER | Memory: 340G | CPUs: 72 | Time: 12:00:00"

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

echo "[INFO] Processing RWA (OTHER)..."
$PY process_country_supply.py RWA --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] RWA completed"
else
    echo "[ERROR] RWA failed"
fi

echo "[INFO] Processing SEN (OTHER)..."
$PY process_country_supply.py SEN --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SEN completed"
else
    echo "[ERROR] SEN failed"
fi

echo "[INFO] Processing SGP (OTHER)..."
$PY process_country_supply.py SGP --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SGP completed"
else
    echo "[ERROR] SGP failed"
fi

echo "[INFO] Processing SLB (OTHER)..."
$PY process_country_supply.py SLB --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SLB completed"
else
    echo "[ERROR] SLB failed"
fi

echo "[INFO] Processing SLE (OTHER)..."
$PY process_country_supply.py SLE --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SLE completed"
else
    echo "[ERROR] SLE failed"
fi

echo "[INFO] Processing SLV (OTHER)..."
$PY process_country_supply.py SLV --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SLV completed"
else
    echo "[ERROR] SLV failed"
fi

echo "[INFO] Processing SRB (OTHER)..."
$PY process_country_supply.py SRB --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SRB completed"
else
    echo "[ERROR] SRB failed"
fi

echo "[INFO] Processing SSD (OTHER)..."
$PY process_country_supply.py SSD --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SSD completed"
else
    echo "[ERROR] SSD failed"
fi

echo "[INFO] Processing STP (OTHER)..."
$PY process_country_supply.py STP --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] STP completed"
else
    echo "[ERROR] STP failed"
fi

echo "[INFO] Processing SUR (OTHER)..."
$PY process_country_supply.py SUR --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SUR completed"
else
    echo "[ERROR] SUR failed"
fi

echo "[INFO] Processing SVK (OTHER)..."
$PY process_country_supply.py SVK --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SVK completed"
else
    echo "[ERROR] SVK failed"
fi

echo "[INFO] Processing SVN (OTHER)..."
$PY process_country_supply.py SVN --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SVN completed"
else
    echo "[ERROR] SVN failed"
fi

echo "[INFO] Processing SWZ (OTHER)..."
$PY process_country_supply.py SWZ --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SWZ completed"
else
    echo "[ERROR] SWZ failed"
fi

echo "[INFO] Processing SYC (OTHER)..."
$PY process_country_supply.py SYC --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SYC completed"
else
    echo "[ERROR] SYC failed"
fi

echo "[INFO] Processing SYR (OTHER)..."
$PY process_country_supply.py SYR --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] SYR completed"
else
    echo "[ERROR] SYR failed"
fi

echo "[INFO] Processing TGO (OTHER)..."
$PY process_country_supply.py TGO --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] TGO completed"
else
    echo "[ERROR] TGO failed"
fi

echo "[INFO] Processing TJK (OTHER)..."
$PY process_country_supply.py TJK --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] TJK completed"
else
    echo "[ERROR] TJK failed"
fi

echo "[INFO] Processing TLS (OTHER)..."
$PY process_country_supply.py TLS --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] TLS completed"
else
    echo "[ERROR] TLS failed"
fi

echo "[INFO] Processing TON (OTHER)..."
$PY process_country_supply.py TON --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] TON completed"
else
    echo "[ERROR] TON failed"
fi

echo "[INFO] Processing TTO (OTHER)..."
$PY process_country_supply.py TTO --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] TTO completed"
else
    echo "[ERROR] TTO failed"
fi

echo "[INFO] Processing TUN (OTHER)..."
$PY process_country_supply.py TUN --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] TUN completed"
else
    echo "[ERROR] TUN failed"
fi

echo "[INFO] Processing TWN (OTHER)..."
$PY process_country_supply.py TWN --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] TWN completed"
else
    echo "[ERROR] TWN failed"
fi

echo "[INFO] Processing UGA (OTHER)..."
$PY process_country_supply.py UGA --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] UGA completed"
else
    echo "[ERROR] UGA failed"
fi

echo "[INFO] Processing URY (OTHER)..."
$PY process_country_supply.py URY --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] URY completed"
else
    echo "[ERROR] URY failed"
fi

echo "[INFO] Processing VCT (OTHER)..."
$PY process_country_supply.py VCT --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] VCT completed"
else
    echo "[ERROR] VCT failed"
fi

echo "[INFO] Processing VIR (OTHER)..."
$PY process_country_supply.py VIR --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] VIR completed"
else
    echo "[ERROR] VIR failed"
fi

echo "[INFO] Processing VNM (OTHER)..."
$PY process_country_supply.py VNM --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] VNM completed"
else
    echo "[ERROR] VNM failed"
fi

echo "[INFO] Processing VUT (OTHER)..."
$PY process_country_supply.py VUT --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] VUT completed"
else
    echo "[ERROR] VUT failed"
fi

echo "[INFO] Processing WSM (OTHER)..."
$PY process_country_supply.py WSM --output-dir outputs_per_country --threads 72
if [ $? -eq 0 ]; then
    echo "[SUCCESS] WSM completed"
else
    echo "[ERROR] WSM failed"
fi

echo "[INFO] Batch 40/40 (OTHER) completed at $(date)"
