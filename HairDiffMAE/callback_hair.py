import os
import torch
import numpy as np
import PIL
from tqdm import tqdm
import torchvision.transforms as transform
from StrandVAE.util.transforms import tangent_to_model_space
from StrandVAE.model.component.modules import find_value_from_uv_mappeing
from StrandVAE.util.utils import *
import torch.nn.functional as F


class CallbackDecodeHair:
    def __init__(self, args, base_dir="/hdd_sda1/tonghs/workspace/SparseHair/StrandVAE"):
        self.args = args
        self.base_dir = base_dir
        self.device = 'cuda:{}'.format(args.device_num) if torch.cuda.is_available() else 'cpu'

        # Set the relevant seeds for reproducibility
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        # Load UV color map once
        uv_color_map_path = os.path.join(self.base_dir, args.uv_color_map)
        self.uv_color_map = transform.functional.vflip(transform.ToTensor()(PIL.Image.open(uv_color_map_path).convert("RGB"))).to(self.device)

        # Initialize and load the model once
        self.model = self._initialize_model(args)

        # Load .npz files once
        self.npz_data = self._load_npz_data()

        # Output directory initialization
        self.output_dir = None

    def _initialize_model(self, args):
        from StrandVAE.model.strand_vae import StrandVAE_3

        model_params = {
            'dim_in': args.dim_in,
            'dim_hidden': args.dim_hidden,
            'dim_out': args.dim_out,
            'num_layers': args.num_layers,
            'w0_initial': args.w0_initial,
            'latent_dim': args.latent_dim,
            'coord_length': args.coord_length,
        }
        model = StrandVAE_3(**model_params) # cVAE 모델 사용
        model.load_state_dict(torch.load(args.model_ckpt, map_location=self.device, weights_only=False)["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model

    def _load_npz_data(self):
        """Load .npz files for roots, tbns, and roots_uv, storing both the mirrored and non-mirrored versions."""
        data_dir = os.path.join(self.base_dir, "data")

        # Non-mirrored version
        npzfile = np.load(f'{data_dir}/roots_tbns_roots_uv.npz')
        roots = torch.from_numpy(npzfile['roots']).to(self.device)
        tbns = torch.from_numpy(npzfile['tbns']).to(self.device)
        roots_uv = torch.from_numpy(npzfile['roots_uv']).to(self.device)

        # Mirrored version
        npzfile_mirror = np.load(f'{data_dir}/roots_tbns_roots_uv_mirror.npz')
        roots_mirror = torch.from_numpy(npzfile_mirror['roots']).to(self.device)
        tbns_mirror = torch.from_numpy(npzfile_mirror['tbns']).to(self.device)
        roots_uv_mirror = torch.from_numpy(npzfile_mirror['roots_uv']).to(self.device)

        return {
            'normal': (roots, tbns, roots_uv),
            'mirror': (roots_mirror, tbns_mirror, roots_uv_mirror)
        }

    def run_callback(self, img_pred_save_dir, img_clean_save_dir, mask=None):
            # Set the output directory
            self.output_dir = os.path.dirname(img_pred_save_dir)

            # Process shape textures
            filename_clean = os.path.basename(img_clean_save_dir).split('.')[0]
            filename_pred = os.path.basename(img_pred_save_dir).split('.')[0]
            filename_output = filename_pred.replace('pred', 'output')

            # Load the shape texture file
            nt_pred = torch.load(img_pred_save_dir, weights_only=False)[:4].to(self.device)
            nt_clean = torch.load(img_clean_save_dir, weights_only=False)[:4].to(self.device)

            # When dataLoader load rezied pt, make it back to 256 256 shape texture size.
            if nt_pred.shape[2] != 256:
                nt_pred = F.interpolate(nt_pred, size=(256, 256), mode='bilinear', align_corners=False)
                nt_clean = F.interpolate(nt_clean, size=(256, 256), mode='bilinear', align_corners=False)
                mask = F.interpolate(mask, size=(256, 256), mode='bilinear', align_corners=False)

            
            # Select preloaded data based on filename
            if '_mirror' in filename_clean:
                roots, tbns, roots_uv = self.npz_data['mirror']
            else:
                roots, tbns, roots_uv = self.npz_data['normal']

            roots_uv = roots_uv[:1000]  # 1000 strands save
            roots_uv_int = (roots_uv * 256).type(torch.int)

            # decode nt_clean to .obj
            for i, ntnt_clean in enumerate(nt_clean):
                zg_clean = ntnt_clean[:, roots_uv_int[:, 0], roots_uv_int[:, 1]].permute(1, 0).to(self.device)
                
                strands_tan = self.model.dec(zg_clean)  # Decode the strands
                strands_model = tangent_to_model_space(strands_tan, tbns, roots)

                # Get strand color using root UV coordinates
                color = torch.ones_like(strands_model) * find_value_from_uv_mappeing(roots_uv[None, :, :], self.uv_color_map)[:, :, None]
                strands_model, color = strands_model.squeeze(0).flatten(0, 1).detach().cpu().numpy(), color.squeeze(0).flatten(0, 1).detach().cpu().numpy()

                os.makedirs(os.path.join(self.output_dir, 'decoded'), exist_ok=True)
                save_hair(os.path.join(self.output_dir, 'decoded', f'{filename_clean}_{i}_decoded.obj'), strands_model, color=color)

            # decode nt_pred to .obj
            for i, ntnt_pred in enumerate(nt_pred):
                zg_pred = ntnt_pred[:, roots_uv_int[:, 0], roots_uv_int[:, 1]].permute(1, 0).to(self.device)
                
                strands_tan = self.model.dec(zg_pred)  # Decode the strands
                strands_model = tangent_to_model_space(strands_tan, tbns, roots)

                # Get strand color using root UV coordinates
                color = torch.ones_like(strands_model) * find_value_from_uv_mappeing(roots_uv[None, :, :], self.uv_color_map)[:, :, None]
                strands_model, color = strands_model.squeeze(0).flatten(0, 1).detach().cpu().numpy(), color.squeeze(0).flatten(0, 1).detach().cpu().numpy()

                os.makedirs(os.path.join(self.output_dir, 'decoded'), exist_ok=True)
                save_hair(os.path.join(self.output_dir, 'decoded', f'{filename_pred}_{i}_decoded.obj'), strands_model, color=color)

            # decode nt_output to .obj
            for i, ntnt_pred in enumerate(nt_pred):
                ntnt = nt_clean[i] * (1 - mask[i]) + ntnt_pred * mask[i]
                # print(nt_clean.shape)       # torch.Size([4, 64, 256, 256])
                # print(nt_clean[i].shape)    # torch.Size([64, 256, 256])
                # print(ntnt_pred.shape)      # torch.Size([64, 256, 256])
                # print(mask.shape)           # torch.Size([4, 64, 256, 256])
                # print(ntnt.shape)           # torch.Size([4, 64, 256, 256])
                zg_output = ntnt[:, roots_uv_int[:, 0], roots_uv_int[:, 1]].permute(1, 0).to(self.device)
                
                strands_tan = self.model.dec(zg_output)  # Decode the strands
                strands_model = tangent_to_model_space(strands_tan, tbns, roots)

                # Get strand color using root UV coordinates
                color = torch.ones_like(strands_model) * find_value_from_uv_mappeing(roots_uv[None, :, :], self.uv_color_map)[:, :, None]
                strands_model, color = strands_model.squeeze(0).flatten(0, 1).detach().cpu().numpy(), color.squeeze(0).flatten(0, 1).detach().cpu().numpy()

                os.makedirs(os.path.join(self.output_dir, 'decoded'), exist_ok=True)
                save_hair(os.path.join(self.output_dir, 'decoded', f'{filename_output}_{i}_decoded.obj'), strands_model, color=color)          









# import os
# import PIL
# import torch
# import warnings
# import numpy as np

# from glob import glob
# from tqdm import tqdm
# from datetime import datetime
# from pathlib import Path
# from shutil import copyfile
# from einops import rearrange
# import torchvision.transforms as transform


# from StrandVAE.util.arguments import decode_hair_args
# from StrandVAE.util.utils import *
# from StrandVAE.model.strand_vae import StrandVAE, StrandVAE_2, StrandVAE_3
# from StrandVAE.util.transforms import tangent_to_model_space
# from StrandVAE.model.component.modules import find_value_from_uv_mappeing, ClosestPointUV2Mesh


# # Base directory where the script is located
# BASE_DIR = "/ext3/tonghs/SparseHair/StrandVAE"

# def callback(args, img_pred_save_dir, img_clean_save_dir):
#     OUTPUT_DIR = os.path.dirname(img_pred_save_dir)
#     data_dir = os.path.join(BASE_DIR, "data")
#     # Set the relevant seeds for reproducibility
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)

#     device = "cpu"

#     # Load uv color map
#     uv_color_map_path = os.path.join(BASE_DIR, args.uv_color_map)
#     uv_color_map = transform.functional.vflip(transform.ToTensor()(PIL.Image.open(uv_color_map_path).convert("RGB"))).to(device)
#     # print(uv_color_map.shape)   # (3(C), 3500(H), 3464(W))

#     # Initialize the model
#     print("Arguments:", args)

#     model_params = {
#         'dim_in': args.dim_in,
#         'dim_hidden': args.dim_hidden,
#         'dim_out': args.dim_out,
#         'num_layers': args.num_layers,
#         'w0_initial': args.w0_initial,
#         'latent_dim': args.latent_dim,
#         'coord_length': args.coord_length,
#     }
#     model = StrandVAE(**model_params)

#     # Load the checkpoint - strand parametric model
#     model.load_state_dict(torch.load(args.model_ckpt, map_location=device, weights_only=True)["model_state_dict"])

#     # Move the model to the relevant device
#     model.to(device)

#     # Set the model to the training mode.
#     model.eval()
    
#     # Backup files for recording
#     file_backup(OUTPUT_DIR)

#     # List of callback shape_textures
#     shape_texture_list = [img_pred_save_dir, img_clean_save_dir]

#     for shape_texture_path in tqdm(shape_texture_list):
#         filename = os.path.basename(shape_texture_path).split('.')[0]

#         nt = torch.load(shape_texture_path, weights_only=False)[0]  # torch.Size([64(C), 256(H), 256(W)])
#         nt = nt.to(device)
#         # nt = rearrange(nt, 'h w c -> c h w')

#         if '_mirror' in filename:
#             npzfile = np.load(f'{data_dir}/roots_tbns_roots_uv_mirror.npz')
#             roots = torch.from_numpy(npzfile['roots'])  # (10000, 3)
#             tbns = torch.from_numpy(npzfile['tbns'])    # (10000, 3, 3)
#             roots_uv = torch.from_numpy(npzfile['roots_uv'])    # (10000, 2)
#         else:
#             npzfile = np.load(f'{data_dir}/roots_tbns_roots_uv.npz')
#             roots = torch.from_numpy(npzfile['roots'])
#             tbns = torch.from_numpy(npzfile['tbns'])
#             roots_uv = torch.from_numpy(npzfile['roots_uv'])

#         roots_uv_int = (roots_uv * 256).type(torch.int)
#         zg = nt[:, roots_uv_int[:,0], roots_uv_int[:,1]]  # (64, root개수)
#         zg = zg.permute(1,0)    # (root개수, 64)
#         strands_tan = model.dec(zg)     # (root개수, 100, 3)
#         strands_model = tangent_to_model_space(strands_tan, tbns, roots)    # (root개수, 100, 3)

#         # Get strand color according to the root uv coordinate
#         color = torch.ones_like(strands_model) * find_value_from_uv_mappeing(roots_uv[None,:,:], uv_color_map)[:, :, None]
        
#         strands_model = strands_model.squeeze(0).flatten(0, 1).detach().cpu().numpy()
#         color = color.squeeze(0).flatten(0, 1).detach().cpu().numpy()
#         os.makedirs(os.path.join(OUTPUT_DIR, 'decoded'), exist_ok=True)
#         save_hair(os.path.join(OUTPUT_DIR, 'decoded', f'{filename}_decoded.obj'), strands_model, color=color)


# def file_backup(output_dir):
#     dir_lis = ["/ext3/tonghs/SparseHair/scripts/get_shape_texture.sh", 
#                "/ext3/tonghs/SparseHair/StrandVAE/get_shape_texture.py", 
#                "/ext3/tonghs/SparseHair/StrandVAE/model",
#                "/ext3/tonghs/SparseHair/StrandVAE/model/component",
#                "/ext3/tonghs/SparseHair/StrandVAE/util"]
#     os.makedirs(os.path.join(output_dir, 'recording'), exist_ok=True)
#     for dir_name in dir_lis:
#         if Path(dir_name).is_dir():
#             cur_dir = os.path.join(output_dir, 'recording', os.path.basename(dir_name))
#             os.makedirs(cur_dir, exist_ok=True)
#             files = os.listdir(dir_name)
#             for f_name in files:
#                 if f_name.endswith('.py'):
#                     copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))
#         else:
#             copyfile(dir_name, os.path.join(output_dir, 'recording', Path(dir_name).name))

