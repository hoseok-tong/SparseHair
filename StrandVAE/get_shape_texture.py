import os
import torch
import warnings
import numpy as np

from glob import glob
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from shutil import copyfile

from StrandVAE.util.arguments import get_shape_texture_args
from StrandVAE.util.utils import *
from StrandVAE.model.strand_vae import StrandVAE, StrandVAE_2, StrandVAE_3
from StrandVAE.model.shape_texture import ExtractShapeTexture


# Base directory where the script is located
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def main():
    args = get_shape_texture_args()

    # Create output directories with current timestamp
    OUTPUT_DIR = os.path.join(BASE_DIR, "output", "shape_texture", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Resolve paths relative to BASE_DIR
    data_dir = os.path.join(BASE_DIR, args.data_dir)

    # Set the relevant seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device on which to run
    if torch.cuda.is_available():
        device = f"cuda:{args.device_num}"
    else:
        warnings.warn(
            "Please note that although executing on CPU is supported,"
            " the training is unlikely to finish in reasonable time"
        )
        device = "cpu"

    # Initialize the model
    print("Arguments:", args)

    model_params = {
        'dim_in': args.dim_in,
        'dim_hidden': args.dim_hidden,
        'dim_out': args.dim_out,
        'num_layers': args.num_layers,
        'w0_initial': args.w0_initial,
        'latent_dim': args.latent_dim,
        'coord_length': args.coord_length,
    }
    model = StrandVAE(**model_params)   # baseline
    # model = StrandVAE_3(**model_params) # ths

    # Load the checkpoint - strand parametric model
    model.load_state_dict(torch.load(args.model_ckpt, map_location=device, weights_only=True)["model_state_dict"])

    # Move the model to the relevant device
    model.to(device)

    # Set the model to the training mode.
    model.eval()
    
    # Backup files for recording
    file_backup(OUTPUT_DIR)

    # Load origin strands
    usc_hair_data_dir = os.path.join(data_dir, 'usc_hair_resampled')
    usc_hair_data_list = sorted(glob(f"{usc_hair_data_dir}/*.npz"), reverse=True)
    usc_hair_mix_data_dir = os.path.join(data_dir, 'usc_hair_mix_resampled')
    usc_hair_mix_data_list = sorted(glob(f"{usc_hair_mix_data_dir}/*/data/*.npz"), reverse=True)
    strands_data_list = usc_hair_data_list + usc_hair_mix_data_list
    # strands_data_list = strands_data_list[:2]
    # print(strands_data_list)

    for hair_npz_path in tqdm(strands_data_list):
        print(hair_npz_path)
        # Load origin source strand
        extractor = ExtractShapeTexture(
            mesh_path=os.path.join(data_dir, 'head_template/head_template.obj'),
            hair_npz_path=hair_npz_path,
            uv_map_size=args.uv_map_size,
            interp_method=args.interp_method,
            model=model,
            output_path=OUTPUT_DIR
        )

        with torch.no_grad():
            extractor.process()


def file_backup(output_dir):
    dir_lis = ["/hdd_sda1/tonghs/workspace/SparseHair/scripts/get_shape_texture.sh", 
               "/hdd_sda1/tonghs/workspace/SparseHair/StrandVAE/get_shape_texture.py", 
               "/hdd_sda1/tonghs/workspace/SparseHair/StrandVAE/model",
               "/hdd_sda1/tonghs/workspace/SparseHair/StrandVAE/model/component",
               "/hdd_sda1/tonghs/workspace/SparseHair/StrandVAE/util"]
    os.makedirs(os.path.join(output_dir, 'recording'), exist_ok=True)
    for dir_name in dir_lis:
        if Path(dir_name).is_dir():
            cur_dir = os.path.join(output_dir, 'recording', os.path.basename(dir_name))
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name.endswith('.py'):
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))
        else:
            copyfile(dir_name, os.path.join(output_dir, 'recording', Path(dir_name).name))


if __name__ == "__main__":
    main()
