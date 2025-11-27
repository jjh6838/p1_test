#!/bin/bash
# Submit all parallel jobs immediately (SLURM will queue them automatically)

# --- Conda bootstrap ---
export PATH=/soge-home/users/lina4376/miniconda3/bin:$PATH
source /soge-home/users/lina4376/miniconda3/etc/profile.d/conda.sh

conda activate p1_etl

echo "[INFO] Submitting 40 parallel jobs..."
echo "[INFO] SLURM will automatically queue and manage job execution (max 8 running at once)"
echo ""

# Submit all jobs
for i in {01..40}; do
    echo "[$(date +%H:%M:%S)] Submitting job $i..."
    sbatch parallel_scripts/submit_parallel_${i}.sh
    sleep 1  # Small delay to avoid overwhelming scheduler
done

echo ""
echo "[INFO] All 40 jobs submitted!"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  watch -n 60 'squeue -u \$USER'"
echo ""
echo "Check completion:"
echo "  find outputs_per_country/parquet -name '*.parquet' | wc -l"
echo ""
echo "Resource allocation summary (tiered partition strategy - smallest first):"
echo "  Other (smallest):  8 countries/script | Short partition (12h)      | 100G, 40 CPUs"
echo "  Tier 3 (medium):   4 countries/script | Short partition (12h)      | 100G, 40 CPUs"
echo "  Tier 2 (large):    2 countries/script | Medium partition (48h/2d)  | 100G, 40 CPUs"  
echo "  Tier 1 (largest):  1 country/script  | Medium partition (48h/2d)  | 100G, 56 CPUs"
