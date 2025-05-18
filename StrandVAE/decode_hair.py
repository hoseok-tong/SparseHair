import os
import PIL
import torch
import warnings
import numpy as np

from glob import glob
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from einops import rearrange
import torchvision.transforms as transform


from StrandVAE.util.arguments import decode_hair_args
from StrandVAE.util.utils import *
from StrandVAE.model.strand_vae import StrandVAE, StrandVAE_2, StrandVAE_3
from StrandVAE.util.transforms import tangent_to_model_space
from StrandVAE.model.component.modules import find_value_from_uv_mappeing, ClosestPointUV2Mesh


# Base directory where the script is located
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def main():
    args = decode_hair_args()

    # Create output directories with current timestamp
    OUTPUT_DIR = os.path.join(BASE_DIR, "output", "decode_hair", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Resolve paths relative to BASE_DIR
    data_dir = os.path.join(BASE_DIR, args.data_dir)

    # Set the relevant seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cpu"
    # # Device on which to run
    # if torch.cuda.is_available():
    #     device = f"cuda:{args.device_num}"
    # else:
    #     warnings.warn(
    #         "Please note that although executing on CPU is supported,"
    #         " the training is unlikely to finish in reasonable time"
    #     )
    #     device = "cpu"

    # Load uv color map
    uv_color_map_path = os.path.join(BASE_DIR, args.uv_color_map)
    uv_color_map = transform.functional.vflip(transform.ToTensor()(PIL.Image.open(uv_color_map_path).convert("RGB"))).to(device)
    # print(uv_color_map.shape)   # (3(C), 3500(H), 3464(W))

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

    # Load shape textures
    # usc_hair_data_dir = os.path.join(data_dir, 'usc_hair_resampled')
    # usc_hair_data_list = sorted(glob(f"{usc_hair_data_dir}/*.pt"), reverse=True)
    # strands_data_list = usc_hair_data_list
    # shape_texture_list = strands_data_list[:1]
    # print(strands_data_list)

    shape_texture_list = ["/hdd_sda1/tonghs/workspace/SparseHair/StrandVAE/output/shape_texture/2025-01-22_09-48-14/latent/strands00514.pt",
                          "/hdd_sda1/tonghs/workspace/SparseHair/StrandVAE/output/shape_texture/2025-01-22_09-48-14/latent/strands00514_mirror.pt"]

    for shape_texture_path in tqdm(shape_texture_list):
        filename = os.path.basename(shape_texture_path).split('.')[0]

        nt = torch.load(shape_texture_path, weights_only=False)
        nt = nt.to(device)
        nt = rearrange(nt, 'h w c -> c h w')

        if '_mirror' in filename:
            npzfile = np.load(f'{data_dir}/roots_tbns_roots_uv_mirror.npz')
            roots = torch.from_numpy(npzfile['roots'])  # (10000, 3)
            tbns = torch.from_numpy(npzfile['tbns'])    # (10000, 3, 3)
            roots_uv = torch.from_numpy(npzfile['roots_uv'])    # (10000, 2)
        else:
            npzfile = np.load(f'{data_dir}/roots_tbns_roots_uv.npz')
            roots = torch.from_numpy(npzfile['roots'])
            tbns = torch.from_numpy(npzfile['tbns'])
            roots_uv = torch.from_numpy(npzfile['roots_uv'])

        roots_uv_int = (roots_uv * 256).type(torch.int)
        zg = nt[:, roots_uv_int[:,0], roots_uv_int[:,1]]  # (64, root개수)
        zg = zg.permute(1,0)    # (root개수, 64)
        strands_tan = model.dec(zg)     # (root개수, 100, 3)
        strands_model = tangent_to_model_space(strands_tan, tbns, roots)    # (root개수, 100, 3)

        # Get strand color according to the root uv coordinate
        color = torch.ones_like(strands_model) * find_value_from_uv_mappeing(roots_uv[None,:,:], uv_color_map)[:, :, None]
        
        strands_model = strands_model.squeeze(0).flatten(0, 1).detach().cpu().numpy()
        color = color.squeeze(0).flatten(0, 1).detach().cpu().numpy()
        os.makedirs(os.path.join(OUTPUT_DIR, 'decoded'), exist_ok=True)
        save_hair(os.path.join(OUTPUT_DIR, 'decoded', f'{filename}_decoded.obj'), strands_model, color=color)




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
