#!/bin/bash
# Submit all parallel jobs with tiered resource allocation

# --- Conda bootstrap ---
export PATH=/soge-home/users/lina4376/miniconda3/bin:$PATH
source /soge-home/users/lina4376/miniconda3/etc/profile.d/conda.sh

conda --version
conda activate p1_etl

echo "[INFO] Submitting 40 parallel jobs with tiered approach..."
echo ""

echo "Script 01: 1 countries (T1) - 340G, 72 CPUs"
echo "Script 02: 1 countries (T1) - 340G, 72 CPUs"
echo "Script 03: 1 countries (T1) - 340G, 72 CPUs"
echo "Script 04: 1 countries (T1) - 340G, 72 CPUs"
echo "Script 05: 1 countries (T1) - 340G, 72 CPUs"
echo "Script 06: 1 countries (T1) - 340G, 72 CPUs"
echo "Script 07: 1 countries (T1) - 340G, 72 CPUs"
echo "Script 08: 2 countries (T2) - 340G, 72 CPUs"
echo "Script 09: 2 countries (T2) - 340G, 72 CPUs"
echo "Script 10: 2 countries (T2) - 340G, 72 CPUs"
echo "Script 11: 2 countries (T2) - 340G, 72 CPUs"
echo "Script 12: 2 countries (T2) - 340G, 72 CPUs"
echo "Script 13: 1 countries (T2) - 340G, 72 CPUs"
echo "Script 14: 4 countries (T3) - 340G, 72 CPUs"
echo "Script 15: 4 countries (T3) - 340G, 72 CPUs"
echo "Script 16: 4 countries (T3) - 340G, 72 CPUs"
echo "Script 17: 4 countries (T3) - 340G, 72 CPUs"
echo "Script 18: 4 countries (T3) - 340G, 72 CPUs"
echo "Script 19: 4 countries (T3) - 340G, 72 CPUs"
echo "Script 20: 4 countries (T3) - 340G, 72 CPUs"
echo "Script 21: 4 countries (T3) - 340G, 72 CPUs"
echo "Script 22: 4 countries (T3) - 340G, 72 CPUs"
echo "Script 23: 4 countries (T3) - 340G, 72 CPUs"
echo "Script 24: 4 countries (T3) - 340G, 72 CPUs"
echo "Script 25: 4 countries (T3) - 340G, 72 CPUs"
echo "Script 26: 4 countries (T3) - 340G, 72 CPUs"
echo "Script 27: 2 countries (T3) - 340G, 72 CPUs"
echo "Script 28: 8 countries (OTHER) - 340G, 72 CPUs"
echo "Script 29: 8 countries (OTHER) - 340G, 72 CPUs"
echo "Script 30: 8 countries (OTHER) - 340G, 72 CPUs"
echo "Script 31: 8 countries (OTHER) - 340G, 72 CPUs"
echo "Script 32: 8 countries (OTHER) - 340G, 72 CPUs"
echo "Script 33: 8 countries (OTHER) - 340G, 72 CPUs"
echo "Script 34: 8 countries (OTHER) - 340G, 72 CPUs"
echo "Script 35: 8 countries (OTHER) - 340G, 72 CPUs"
echo "Script 36: 8 countries (OTHER) - 340G, 72 CPUs"
echo "Script 37: 8 countries (OTHER) - 340G, 72 CPUs"
echo "Script 38: 8 countries (OTHER) - 340G, 72 CPUs"
echo "Script 39: 8 countries (OTHER) - 340G, 72 CPUs"
echo "Script 40: 29 countries (OTHER) - 340G, 72 CPUs"

echo ""
sbatch parallel_scripts/submit_parallel_01.sh
sbatch parallel_scripts/submit_parallel_02.sh
sbatch parallel_scripts/submit_parallel_03.sh
sbatch parallel_scripts/submit_parallel_04.sh
sbatch parallel_scripts/submit_parallel_05.sh
sbatch parallel_scripts/submit_parallel_06.sh
sbatch parallel_scripts/submit_parallel_07.sh
sbatch parallel_scripts/submit_parallel_08.sh
sbatch parallel_scripts/submit_parallel_09.sh
sbatch parallel_scripts/submit_parallel_10.sh
sbatch parallel_scripts/submit_parallel_11.sh
sbatch parallel_scripts/submit_parallel_12.sh
sbatch parallel_scripts/submit_parallel_13.sh
sbatch parallel_scripts/submit_parallel_14.sh
sbatch parallel_scripts/submit_parallel_15.sh
sbatch parallel_scripts/submit_parallel_16.sh
sbatch parallel_scripts/submit_parallel_17.sh
sbatch parallel_scripts/submit_parallel_18.sh
sbatch parallel_scripts/submit_parallel_19.sh
sbatch parallel_scripts/submit_parallel_20.sh
sbatch parallel_scripts/submit_parallel_21.sh
sbatch parallel_scripts/submit_parallel_22.sh
sbatch parallel_scripts/submit_parallel_23.sh
sbatch parallel_scripts/submit_parallel_24.sh
sbatch parallel_scripts/submit_parallel_25.sh
sbatch parallel_scripts/submit_parallel_26.sh
sbatch parallel_scripts/submit_parallel_27.sh
sbatch parallel_scripts/submit_parallel_28.sh
sbatch parallel_scripts/submit_parallel_29.sh
sbatch parallel_scripts/submit_parallel_30.sh
sbatch parallel_scripts/submit_parallel_31.sh
sbatch parallel_scripts/submit_parallel_32.sh
sbatch parallel_scripts/submit_parallel_33.sh
sbatch parallel_scripts/submit_parallel_34.sh
sbatch parallel_scripts/submit_parallel_35.sh
sbatch parallel_scripts/submit_parallel_36.sh
sbatch parallel_scripts/submit_parallel_37.sh
sbatch parallel_scripts/submit_parallel_38.sh
sbatch parallel_scripts/submit_parallel_39.sh
sbatch parallel_scripts/submit_parallel_40.sh

echo "All 40 jobs submitted!"
echo ""
echo "Resource allocation summary:"
echo "  Tier 1 (biggest): 1 country/script, 340G, 72 CPUs"
echo "  Tier 2 (large):   2 countries/script, 340G, 72 CPUs"  
echo "  Tier 3 (medium):  4 countries/script, 340G, 72 CPUs"
echo "  Other (small):    8 countries/script, 340G, 72 CPUs"
echo ""
echo "Monitor with: squeue -u $USER"
