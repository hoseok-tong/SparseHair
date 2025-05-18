#!/bin/bash

# Need to use this to activate conda environments
eval "$(conda shell.bash hook)"
conda deactivate && conda activate sparsehair

########################################################################
# General Training Parameters
SEED=42
DEVICE_NUM=0
########################################################################

# Data Parameters
DATA_DIR="data/"  # Input data directory
UV_COLOR_MAP="data/head_template/head_template.png"
# MODEL_CKPT="/hdd_sda1/tonghs/workspace/SparseHair/StrandVAE/output/strand_vae/2024-10-18_15-57-46/checkpoint_epoch_1000.pth"  # baseline
MODEL_CKPT="/hdd_sda1/tonghs/workspace/SparseHair/StrandVAE/output/strand_vae/2025-01-20_20-41-44/checkpoint_epoch_700.pth"  # baseline w leakyrelu
# MODEL_CKPT="/hdd_sda1/tonghs/workspace/SparseHair/StrandVAE/output/strand_vae/2024-10-22_14-09-13/checkpoint_epoch_1400.pth"    # strandvae_3
########################################################################

# Model Parameters StrandVAE
DIM_IN=1
DIM_HIDDEN=256
DIM_OUT=6  # 6 for 6DoF rotation(strandvae_3), 3 for euler rotation(baseline)
NUM_LAYERS=5
W0_INITIAL=30.0
LATENT_DIM=64
COORD_LENGTH=100 # 99 for (strandvae_3), 100 for (baseline)
########################################################################

# Navigate to the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
export PYTHONPATH="$PYTHONPATH:$(pwd)"


# Run the Python training script with the specified arguments
python StrandVAE/decode_hair.py \
    --seed $SEED \
    --device_num $DEVICE_NUM \
    --data_dir $DATA_DIR \
    --dim_in $DIM_IN \
    --dim_hidden $DIM_HIDDEN \
    --dim_out $DIM_OUT \
    --num_layers $NUM_LAYERS \
    --w0_initial $W0_INITIAL \
    --latent_dim $LATENT_DIM \
    --coord_length $COORD_LENGTH \
    --model_ckpt $MODEL_CKPT \
    --uv_color_map $UV_COLOR_MAP
