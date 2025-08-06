#!/bin/bash --login


#SBATCH --job-name=supply_analysis
#SBATCH --partition=Short
#SBATCH --time=12:00:00
#SBATCH --mem=340G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --output=outputs_global/logs/snakemake_%j.out
#SBATCH --error=outputs_global/logs/snakemake_%j.err
#SBATCH --mail-type=END,FAIL

mkdir -p outputs_per_country
mkdir -p outputs_global/logs



# Make conda available on compute nodes
export PATH=/soge-home/users/lina4376/miniconda3/bin:$PATH

cd $SLURM_SUBMIT_DIR

echo "Starting Snakemake workflow for 200 countries at $(date)"
echo "Working directory: $(pwd)"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"




snakemake \
    --cores 72 \
    --use-conda \
    --conda-prefix conda_envs \
    --rerun-incomplete \
    --keep-going \
    --printshellcmds \
    --latency-wait 60

echo "Snakemake workflow completed at $(date)"
