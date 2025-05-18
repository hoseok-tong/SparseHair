#!/bin/bash

# Need to use this to activate conda environments
eval "$(conda shell.bash hook)"
conda deactivate && conda activate sparsehair

########################################################################
# General Training Parameters
SEED=42
DEVICE_NUM=0
MAX_EPOCHS=5001
STATS_PRINT_INTERVAL=100
VALIDATION_EPOCH_INTERVAL=20
CHECKPOINT_EPOCH_INTERVAL=100
########################################################################

# Output Directories
TENSORBOARD_LOG_DIR="tensorboard/"
CALLBACK_PATH="callback/"
########################################################################

# Data Parameters
DATA_DIR="data/"  # Input data directory
BATCH_SIZE=3000
########################################################################

# Model Parameters
DIM_IN=1
DIM_HIDDEN=256
DIM_OUT=6  # 6 for 6DoF rotation, 3 for axis angle
NUM_LAYERS=5
W0_INITIAL=30.0
LATENT_DIM=64
COORD_LENGTH=100
########################################################################

# Loss Parameters
L_MAIN_MSE=1.0
L_MAIN_COS=1.0
L_KLD=0.0001
########################################################################

# Optimizer Parameters
LEARNING_RATE=1e-4
########################################################################

# # Other Parameters (Add any additional parameters you need)
# NUM_WORKERS=4  # Number of worker processes for data loading
# SAVE_CHECKPOINTS=1  # Set to 1 to save checkpoints

# # Create output directory with timestamp
# OUTPUT_DIR="output/$(date +'%Y-%m-%d_%H-%M-%S')"
# mkdir -p $OUTPUT_DIR
########################################################################

# Navigate to the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
export PYTHONPATH="$PYTHONPATH:$(pwd)"


# Run the Python training script with the specified arguments
python StrandVAE/train_strand_vae.py \
    --seed $SEED \
    --device_num $DEVICE_NUM \
    --max_epochs $MAX_EPOCHS \
    --stats_print_interval $STATS_PRINT_INTERVAL \
    --validation_epoch_interval $VALIDATION_EPOCH_INTERVAL \
    --checkpoint_epoch_interval $CHECKPOINT_EPOCH_INTERVAL \
    --tensorboard_log_dir $TENSORBOARD_LOG_DIR \
    --callback_path $CALLBACK_PATH \
    --data_dir $DATA_DIR \
    --batch_size $BATCH_SIZE \
    --dim_in $DIM_IN \
    --dim_hidden $DIM_HIDDEN \
    --dim_out $DIM_OUT \
    --num_layers $NUM_LAYERS \
    --w0_initial $W0_INITIAL \
    --latent_dim $LATENT_DIM \
    --coord_length $COORD_LENGTH \
    --l_main_mse $L_MAIN_MSE \
    --l_main_cos $L_MAIN_COS \
    --l_kld $L_KLD \
    --learning_rate $LEARNING_RATE \
    # --num_workers $NUM_WORKERS \
    # --save_checkpoints $SAVE_CHECKPOINTS \
    # --output_dir $OUTPUT_DIR
