#!/bin/bash

# Need to use this to activate conda environments
eval "$(conda shell.bash hook)"
conda deactivate && conda activate sparsehair

# Navigate to the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
export PYTHONPATH="$PYTHONPATH:$(pwd)"


# Run the Python training script with the specified arguments
python StrandVAE/gui_strand.py
