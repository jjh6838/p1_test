#!/bin/bash
# Submit all parallel jobs in waves of 8 (respects per-user job limits)

# --- Conda bootstrap ---
export PATH=/soge-home/users/lina4376/miniconda3/bin:$PATH
source /soge-home/users/lina4376/miniconda3/etc/profile.d/conda.sh

conda activate p1_etl

echo "[INFO] Submitting 40 parallel jobs in waves of 8..."
echo "[INFO] This respects per-user job limits and prevents queue congestion"
echo ""

# Function to count running/pending jobs
count_jobs() {
    squeue -u $USER -h -t pending,running -r | wc -l
}

# Function to wait until job count drops below threshold
wait_for_slots() {
    local max_jobs=$1
    while [ $(count_jobs) -ge $max_jobs ]; do
        echo "[$(date +%H:%M:%S)] Waiting for job slots... ($(count_jobs)/$max_jobs running)"
        sleep 60
    done
}

echo "[INFO] Starting wave-based submission..."
echo ""

# Submit jobs in waves
for i in {01..40}; do
    # Wait if we have 8 or more jobs running/pending
    wait_for_slots 8
    
    echo "[$(date +%H:%M:%S)] Submitting job $i..."
    sbatch parallel_scripts/submit_parallel_${i}.sh
    
    # Small delay to avoid overwhelming scheduler
    sleep 2
done

echo ""
echo "[INFO] All 40 jobs submitted!"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  watch -n 60 'squeue -u \$USER | wc -l'"
echo ""
echo "Check completion:"
echo "  find outputs_per_country/parquet -name '*.parquet' | wc -l"
echo ""
echo "Resource allocation summary (tiered partition strategy):"
echo "  Tier 1 (biggest):  1 country/script  | Long partition (168h/7d)   | 896G, 56 CPUs"
echo "  Tier 2 (large):    2 countries/script | Medium partition (48h/2d)  | 896G, 56 CPUs"  
echo "  Tier 3 (medium):   4 countries/script | Short partition (12h)      | 896G, 56 CPUs"
echo "  Other (small):     8 countries/script | Short partition (12h)      | 896G, 56 CPUs"
