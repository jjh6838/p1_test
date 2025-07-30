#!/bin/bash
# submit_workflow.sh - Submit Snakemake workflow to SLURM cluster

#SBATCH --job-name=supply_analysis
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/snakemake_%j.out
#SBATCH --error=logs/snakemake_%j.err

# Create logs directory
mkdir -p logs

# Load required modules (adjust based on your cluster)
# module load python/3.9
# module load anaconda3

# Activate conda environment
# conda activate your_environment_name

# Set Snakemake parameters
MAX_JOBS=50  # Maximum number of simultaneous jobs
CLUSTER_CONFIG="cluster_config.yaml"

echo "Starting Snakemake workflow at $(date)"
echo "Working directory: $(pwd)"

# Run Snakemake with cluster submission
snakemake \
    --jobs $MAX_JOBS \
    --cluster-config $CLUSTER_CONFIG \
    --cluster "sbatch --partition={cluster.partition} --time={cluster.time} --mem={cluster.mem} --nodes={cluster.nodes} --ntasks={cluster.ntasks} --cpus-per-task={cluster.cpus-per-task} --output={cluster.output} --error={cluster.error}" \
    --use-conda \
    --conda-prefix conda_envs \
    --rerun-incomplete \
    --keep-going \
    --printshellcmds \
    --reason

echo "Snakemake workflow completed at $(date)"
