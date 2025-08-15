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


echo ""
sbatch parallel_scripts/submit_parallel_01.sh
sbatch parallel_scripts/submit_parallel_02.sh
sbatch parallel_scripts/submit_parallel_03.sh
sbatch parallel_scripts/submit_parallel_04.sh
sbatch parallel_scripts/submit_parallel_05.sh
sbatch parallel_scripts/submit_parallel_06.sh
sbatch parallel_scripts/submit_parallel_07.sh


echo "All 7 jobs submitted!"
echo ""
echo "Resource allocation summary:"
echo "  Tier 1 (biggest): 1 country/script, 340G, 72 CPUs"
echo ""
echo "Monitor with: squeue -u $USER"
