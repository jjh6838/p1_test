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
mkdir -p outputs_global
mkdir -p outputs_global/logs

# Reset submit_workflow.sh to use the clean conda installation
export PATH=/soge-home/users/lina4376/miniconda3/bin:$PATH
source /soge-home/users/lina4376/miniconda3/etc/profile.d/conda.sh

# Check for conda/micromamba conflicts
echo "Checking conda setup..."
which conda
which mamba 2>/dev/null || echo "mamba not found"
which micromamba 2>/dev/null || echo "micromamba not found"
conda --version

# Validate environment.yml file before proceeding
echo "Validating environment.yml..."
if [ -f "environment.yml" ]; then
    # Check if file is readable and has valid YAML syntax
    python -c "import yaml; yaml.safe_load(open('environment.yml'))" 2>/dev/null && echo "environment.yml is valid" || {
        echo "ERROR: environment.yml is corrupted or has invalid syntax"
        echo "File encoding check:"
        file environment.yml
        echo "First few bytes:"
        hexdump -C environment.yml | head -3
        exit 1
    }
else
    echo "ERROR: environment.yml not found"
    exit 1
fi

# Use absolute path for conda environments to avoid conflicts
CONDA_ENVS_PATH="/soge-home/users/lina4376/miniconda3/envs"
echo "Using conda environments path: $CONDA_ENVS_PATH"


# Remove old conda_envs to force fresh environment creation
if [ -d "conda_envs" ]; then
    echo "Removing old conda_envs directory to ensure clean environment..."
    rm -rf conda_envs
fi

# Also clean up any environments in the main conda path
if [ -d "$CONDA_ENVS_PATH/p1_snakemake" ]; then
    echo "Removing old p1_snakemake environment..."
    conda env remove -n p1_snakemake -y 2>/dev/null || true
fi

# Remove old .snakemake to force a clean workflow run
if [ -d ".snakemake" ]; then
    echo "Removing old .snakemake directory for a clean workflow run..."
    rm -rf .snakemake
fi

cd $SLURM_SUBMIT_DIR

echo "Starting Snakemake workflow for 200 countries at $(date)"
echo "Working directory: $(pwd)"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"




# Debugging: Run Snakemake with 4 cores for troubleshooting
# snakemake --cores 4 --use-conda --rerun-incomplete --printshellcmds --latency-wait 60

# First, create environments only to check for errors
echo "Creating conda environments first..."
snakemake \
    --cores 1 \
    --use-conda \
    --conda-prefix "$CONDA_ENVS_PATH" \
    --conda-create-envs-only

# Check if pandas is available in the created environment
if [ -d "$CONDA_ENVS_PATH" ]; then
    echo "Checking for pandas in created environments..."
    for env_dir in "$CONDA_ENVS_PATH"/*/; do
        if [ -d "$env_dir" ]; then
            env_name=$(basename "$env_dir")
            echo "Testing environment: $env_name"
            conda activate "$env_name" 2>/dev/null || source activate "$env_name"
            python -c "import pandas; print('pandas version:', pandas.__version__)" 2>/dev/null || echo "pandas NOT found in $env_name"
            conda deactivate 2>/dev/null || true
        fi
    done
fi

echo "Running main workflow..."
snakemake \
    --cores 72 \
    --use-conda \
    --conda-prefix "$CONDA_ENVS_PATH" \
    --rerun-incomplete \
    --keep-going \
    --printshellcmds \
    --latency-wait 60

echo "Snakemake workflow completed at $(date)"
