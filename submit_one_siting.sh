#!/bin/bash
# Run a single parallel siting script by number
# Usage: ./submit_one_siting.sh <script_number>

if [ $# -ne 1 ]; then
    echo "Usage: $0 <script_number>"
    echo "Example: $0 06"
    echo ""
    echo "Available scripts:"
    ls -1 parallel_scripts_siting/submit_parallel_siting_*.sh | sed 's/.*submit_parallel_siting_/  /' | sed 's/.sh//'
    exit 1
fi

SCRIPT_NUM=$(printf "%02d" $1)
SCRIPT_FILE="parallel_scripts_siting/submit_parallel_siting_${SCRIPT_NUM}.sh"

if [ ! -f "$SCRIPT_FILE" ]; then
    echo "Error: Script not found: $SCRIPT_FILE"
    echo ""
    echo "Available scripts:"
    ls -1 parallel_scripts_siting/submit_parallel_siting_*.sh | sed 's/.*submit_parallel_siting_/  /' | sed 's/.sh//'
    exit 1
fi

echo "[INFO] Submitting siting script ${SCRIPT_NUM}..."
sbatch "$SCRIPT_FILE"

echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  watch -n 60 'squeue -u \$USER'"
