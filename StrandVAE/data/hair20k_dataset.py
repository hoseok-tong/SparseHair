import os
import glob
import torch
import numpy as np

from tqdm import tqdm
from pytorch3d.io import load_obj
from torch.utils.data import Dataset

from StrandVAE.util.utils import load_hair
from StrandVAE.util.transforms import model_to_tangent_space, calculate_TBNs


class Hair20k(Dataset):
    def __init__(self, data_dir, split="train", preprocess=True):
        super().__init__()
        usc_hair_data_dir = os.path.join(data_dir, 'usc_hair_resampled')
        usc_hair_data_list = sorted(glob.glob(f"{usc_hair_data_dir}/*.data"), reverse=True)
        strands_data_list = usc_hair_data_list
        # usc_hair_mix_data_dir = os.path.join(data_dir, 'usc_hair_mix_resampled')
        # usc_hair_mix_data_list = sorted(glob.glob(f"{usc_hair_mix_data_dir}/*/data/*.data"), reverse=True)
        # strands_data_list = usc_hair_data_list + usc_hair_mix_data_list

        if split == "train":
            # strands_data_list = strands_data_list[:int(len(strands_data_list) * 0.001)]
            strands_data_list = strands_data_list[:int(len(strands_data_list) * 0.8)]
        else:
            # strands_data_list = strands_data_list[int(len(strands_data_list) * 0.999):]
            strands_data_list = strands_data_list[int(len(strands_data_list) * 0.8):]

        head_model_path = os.path.join(data_dir, 'head_template/head_template.obj')
        vertsH, facesH, auxH = load_obj(head_model_path)

        list_strand_tan, list_tbn, list_root = [], [], []   
        for data_path in tqdm(strands_data_list, desc="Process input dataloader"):
            npz_path = data_path.replace('.data', '.npz')
            
            # Check if preprocessed .npz file exists
            if preprocess and os.path.exists(npz_path):
                # Load pre-processed data
                npz_data = np.load(npz_path)
                vertsS_tan = torch.from_numpy(npz_data['vertsS_tan'])
                TBNs = torch.from_numpy(npz_data['TBNs'])
                roots = torch.from_numpy(npz_data['roots'])
            else:
                # Preprocess and save the data
                vertsS_model = torch.from_numpy(load_hair(data_path))  # (num_strands, 100, 3)
                roots = vertsS_model[:, 0]
                TBNs = calculate_TBNs(roots, vertsH, facesH, auxH)
                vertsS_tan = model_to_tangent_space(vertsS_model, TBNs, roots)

                # Save preprocessed data to .npz
                np.savez(npz_path, vertsS_tan=vertsS_tan.numpy(), TBNs=TBNs.numpy(), roots=roots.numpy())
            
            list_strand_tan.append(vertsS_tan)
            list_tbn.append(TBNs)
            list_root.append(roots)
        self.strands_tan = torch.cat(list_strand_tan, dim=0)
        self.tbns = torch.cat(list_tbn, dim=0)
        self.roots = torch.cat(list_root, dim=0)

    def __len__(self):
        return len(self.strands_tan)

    def __getitem__(self, idx):
        return self.strands_tan[idx], self.tbns[idx], self.roots[idx]