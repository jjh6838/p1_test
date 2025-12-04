#!/bin/bash
# Submit all parallel siting analysis jobs

# --- Conda bootstrap ---
export PATH=/soge-home/users/lina4376/miniconda3/bin:$PATH
source /soge-home/users/lina4376/miniconda3/etc/profile.d/conda.sh

conda activate p1_etl

echo "[INFO] Submitting 24 parallel siting analysis jobs..."
echo "[INFO] SLURM will automatically queue and manage job execution"
echo ""

# Submit all jobs
for i in {01..24}; do
    echo "[$(date +%H:%M:%S)] Submitting siting job $i..."
    sbatch parallel_scripts_siting/submit_parallel_siting_${i}.sh
    sleep 1
done

echo ""
echo "[INFO] All 24 siting jobs submitted!"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  watch -n 60 'squeue -u \$USER'"
echo ""
echo "Check completion:"
echo "  find outputs_per_country/parquet -name 'siting_*.parquet' | wc -l"
echo ""
echo "Resource allocation summary (siting analysis):"
echo "  Tier 1 (CHN, USA):              1 country/script  | Medium partition (48h) | 98G, 40 CPUs"
echo "  Tier 2 (IND, CAN, etc.):        2 countries/script | Medium partition (24h) | 98G, 40 CPUs"
echo "  Tier 3 (all others):           11 countries/script | Short partition (12h)  | 28G, 40 CPUs"
