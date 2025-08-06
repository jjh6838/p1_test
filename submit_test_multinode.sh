#!/bin/bash
#SBATCH --job-name=test_multinode
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --output=outputs_global/logs/test_multinode_%j.out
#SBATCH --error=outputs_global/logs/test_multinode_%j.err

# Load required modules
module load conda

# Create output directories
mkdir -p outputs_per_country
mkdir -p outputs_global/logs

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

echo "Multi-node test started"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "Total CPUs: $SLURM_CPUS_PER_TASK per task x $SLURM_NTASKS tasks"
echo "Memory per node: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"

# Test multiple countries in parallel
echo "Testing multiple countries in parallel..."

# Country 1: Jamaica (small island)
srun --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16G \
     python process_country_supply.py JAM --output-dir outputs_per_country --threads 8 &

# Country 2: Luxembourg (small European country)  
srun --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16G \
     python process_country_supply.py LUX --output-dir outputs_per_country --threads 8 &

# Wait for both to complete
wait

echo "Parallel test completed at: $(date)"

# Test combine function
echo "Testing combine function..."
python combine_global_results.py --input-dir outputs_per_country --output-file outputs_global/global_multinode_test.parquet

echo "Multi-node test completed!"
