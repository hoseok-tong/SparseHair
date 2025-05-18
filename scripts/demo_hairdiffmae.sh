#!/bin/bash

# Need to use this to activate conda environments
eval "$(conda shell.bash hook)"
conda deactivate && conda activate hairdiffmae

########################################################################
# General Training Parameters
SEED=42
DEVICE_NUM=0
#DATA_DIR="/hdd_sda1/tonghs/workspace/SparseHair/StrandVAE/output/shape_texture/2024-10-25_01-16-09/"  # baseline shape texture
DATA_DIR="/hdd_sda1/tonghs/workspace/SparseHair/StrandVAE/output/shape_texture/2024-10-28_23-47-24/"  # cVAE shape texture
########################################################################

# Training Parameters DiffMAE
BATCH_SIZE=32   # 64
MASK_RATIO=0.75
DDPM_NUM_STEPS=1000 #5000

HIDDEN_SIZE=768
NUM_HIDDEN_LAYERS=12
NUM_ATTENTION_HEADS=12
INTERMEDIATE_SIZE=3072
IMAGE_SIZE=256
PATCH_SIZE=16
NUM_CHANNELS=64
DECODER_NUM_ATTENTION_HEADS=16
DECODER_HIDDEN_SIZE=512
DECODER_NUM_HIDDEN_LAYERS=8
DECODER_INTERMEDIATE_SIZE=2048

# Model Parameters StrandVAE (Callback)
DIM_IN=1
DIM_HIDDEN=256
DIM_OUT=6  # 6 for 6DoF rotation, 3 for euler rotation
NUM_LAYERS=5
W0_INITIAL=30.0
LATENT_DIM=64
COORD_LENGTH=99

# Pre-trained StrandVAE (Callback)
UV_COLOR_MAP="data/head_template/head_template.png"
#MODEL_CKPT="/hdd_sda1/tonghs/workspace/SparseHair/StrandVAE/output/strand_vae/2024-10-18_15-57-46/checkpoint_epoch_1000.pth"
MODEL_CKPT="/hdd_sda1/tonghs/workspace/SparseHair/StrandVAE/output/strand_vae/2024-10-22_14-09-13/checkpoint_epoch_1400.pth"    # strandvae_3
########################################################################

# Navigate to the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
export PYTHONPATH="$PYTHONPATH:$(pwd)"


# Run the Python training script with the specified arguments
python HairDiffMAE/main_diffmae_hair_ths.py \
    --seed $SEED \
    --device_num $DEVICE_NUM \
    --data_dir $DATA_DIR \
    --batch_size $BATCH_SIZE \
    --mask_ratio $MASK_RATIO \
    --ddpm_num_steps $DDPM_NUM_STEPS \
    --hidden_size $HIDDEN_SIZE \
    --num_hidden_layers $NUM_HIDDEN_LAYERS \
    --num_attention_heads $NUM_ATTENTION_HEADS \
    --intermediate_size $INTERMEDIATE_SIZE \
    --image_size $IMAGE_SIZE \
    --patch_size $PATCH_SIZE \
    --num_channels $NUM_CHANNELS \
    --decoder_num_attention_heads $DECODER_NUM_ATTENTION_HEADS \
    --decoder_hidden_size $DECODER_HIDDEN_SIZE \
    --decoder_num_hidden_layers $DECODER_NUM_HIDDEN_LAYERS \
    --decoder_intermediate_size $DECODER_INTERMEDIATE_SIZE \
    --dim_in $DIM_IN \
    --dim_hidden $DIM_HIDDEN \
    --dim_out $DIM_OUT \
    --num_layers $NUM_LAYERS \
    --w0_initial $W0_INITIAL \
    --latent_dim $LATENT_DIM \
    --coord_length $COORD_LENGTH \
    --model_ckpt $MODEL_CKPT \
    --uv_color_map $UV_COLOR_MAP \
    --ddpm_num_steps $DDPM_NUM_STEPS
