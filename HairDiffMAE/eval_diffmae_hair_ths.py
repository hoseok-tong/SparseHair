import os
import torch
import random
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm

from diffusers import DDPMScheduler
from data_loader_ths import get_dataloaders
from transformers import ViTDiffMAEForPreTraining, ViTDiffMAEConfig

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def main(args, dataloader, model, device):
    results_dir = os.path.join(args.outputs_dir, 'gen_hairstyles')
    os.makedirs(results_dir, exist_ok=True)

    noise_scheduler = DDPMScheduler(num_train_timesteps=args.ddpm_num_steps, beta_schedule=args.ddpm_beta_schedule)
    model.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))   
        img_clean = batch['images'].to(device)

        noise = torch.randn(img_clean.shape).to(device)

        data_max = torch.load(os.path.join(args.data_dir, 'optim_neural_textures/min_max/data_max.pt')).to(device)
        data_min = torch.load(os.path.join(args.data_dir, 'optim_neural_textures/min_max/data_min.pt')).to(device)
        time_schedules = torch.cat([noise_scheduler.timesteps[[0]], noise_scheduler.timesteps[[-1]], torch.tensor(0)[None]]).to(device)
        
        # control the noise seed -> set up mask patches
        # noise_seed = torch.arange(256, 0, -1).reshape(16, 16).transpose(1, 0).reshape(1, 256).to(device)          # right half keep. otherwise remove: 성민형

        mask_map = (1 - torch.load(sorted(glob(os.path.join(args.data_dir, 'hair3D/*_mask.pt')))[-1])).repeat(1, 1, 32).transpose(1, 0)         # ths
        mask_map_patch = model.patchify(mask_map.permute(2,1,0).unsqueeze(0))
        noise_seed = (mask_map_patch.sum(dim=2) > 0).float().to(device)
        non_zero_ratio = (noise_seed != 0).float().mean()
        model.config.mask_ratio = non_zero_ratio.item()

        for i in tqdm(range(len(time_schedules)-1)):
            if i == 0:
                img_noise = noise_scheduler.add_noise(img_clean, noise, time_schedules[i])
                outputs = model(img_clean, img_noise, time_schedules[i], noise=noise_seed) 
                noise_seed = outputs.noise.detach()
            else:
                # 2. compute previous image: x_t -> x_t-1
                img_noise = noise_scheduler.add_noise(img_pred, img_noise, time_schedules[i+1])
                outputs = model(img_clean, img_noise, time_schedules[i], noise=noise_seed)

            # 1. predict noise model_output
            img_pred = outputs.logits.detach()
            img_pred = model.unpatchify(img_pred)
            img_pred = img_pred.clamp(-1, 1)

            mask_noise = outputs.mask.detach().detach().cpu()
            mask_noise = mask_noise.unsqueeze(-1).repeat(1, 1, model.config.patch_size**2 * model.config.num_channels)  # (N, H*W, p*p*3)
            mask_noise = model.unpatchify(mask_noise)  # 1 is removing, 0 is keeping

            img_pred_save = img_pred * (data_max - data_min) + data_min
            img_clean_save = img_clean * (data_max - data_min) + data_min

            # tmp_mask: make 0 value to Nan
            tmp_mask = torch.ones(img_clean_save.shape)
            nan_mask = (img_clean_save == 0).all(1).unsqueeze(1).repeat(1, img_clean_save.shape[1], 1, 1)
            tmp_mask[nan_mask == 1] = torch.nan

            img_clean_save = img_clean_save.detach().cpu() * tmp_mask
            img_pred_save = img_pred_save.detach().cpu() * tmp_mask
            img_final_save = (img_clean_save.detach().cpu() * (1 - mask_noise) + img_pred_save.detach().cpu() * (mask_noise))

            torch.save(img_clean_save, os.path.join(results_dir, 'img_clean_nor_%03d.pt' % time_schedules[i]))
            torch.save(img_pred_save, os.path.join(results_dir, 'img_pred_nor_%03d.pt' % time_schedules[i]))
            torch.save(img_final_save, os.path.join(results_dir, 'img_final_nor_%03d.pt' % time_schedules[i]))
            torch.save(mask_noise.detach().cpu(), os.path.join(results_dir, 'mask_nor_%03d.pt' % time_schedules[i]))    # 1 is removing, 0 is keeping


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.5) # ratio to be removed
    parser.add_argument('--total_epoch', type=int, default=10000)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument('--outputs_dir', type=str, default='outputs')
    parser.add_argument('--warmup_epoch', type=int, default=200)

    args = parser.parse_args()

    setup_seed(args.seed)
    device = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'

    ''' Dataset '''
    dataset = get_dataloaders(args)
    dataloader = dataset['train']

    ''' Network '''
    config = ViTDiffMAEConfig(num_channels=32, image_size=args.image_size, patch_size=args.patch_size, mask_ratio=args.mask_ratio)
    model = ViTDiffMAEForPreTraining(config=config).to(device)
    model.load_state_dict(torch.load(sorted(glob(os.path.join(args.outputs, 'checkpoints/model_epoch_*.pth')))[-1], map_location=f"cuda:{args.gpu}"))

    main(args, dataloader, model, device)