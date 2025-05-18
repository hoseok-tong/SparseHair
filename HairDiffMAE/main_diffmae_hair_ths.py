import os
import math
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from HairDiffMAE.transformers.src.transformers.models.vit_diffmae import ViTDiffMAEForPreTraining, ViTDiffMAEConfig
from HairDiffMAE.diffusers.src.diffusers.schedulers import DDPMScheduler
from data_loader_ths import get_dataloaders

from HairDiffMAE.callback_hair import CallbackDecodeHair  # Import the refactored Callback class

import cv2
import matplotlib.cm as cm

# Base directory where the script is located
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    setup_seed(args.seed)
    device = 'cuda:{}'.format(args.device_num) if torch.cuda.is_available() else 'cpu'
    OUTPUT_DIR = os.path.join(BASE_DIR, "output", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    
    # Backup files for recording
    file_backup(OUTPUT_DIR)    

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'callback'), exist_ok=True)

    # Dataset
    dataset = get_dataloaders(args)
    dataloader = dataset['train']
    
    # Datast's min-max
    data_max = torch.load(os.path.join(args.data_dir, 'min_max/data_max.pt'), weights_only=False).to(device)    # tensor(19.4395)
    data_min = torch.load(os.path.join(args.data_dir, 'min_max/data_min.pt'), weights_only=False).to(device)    # tensor(-25.0610)

    # Network Configuration and Model Initialization
    config = ViTDiffMAEConfig(hidden_size=args.hidden_size,
                              num_hidden_layers=args.num_hidden_layers,
                              num_attention_heads=args.num_attention_heads,
                              intermediate_size=args.intermediate_size,
                              image_size=args.image_size,
                              patch_size=args.patch_size,
                              num_channels=args.num_channels,
                              decoder_num_attention_heads=args.decoder_num_attention_heads,
                              decoder_hidden_size=args.decoder_hidden_size,
                              decoder_num_hidden_layers=args.decoder_num_hidden_layers,
                              decoder_intermediate_size=args.decoder_intermediate_size,
                              mask_ratio=args.mask_ratio,
                              norm_pix_loss=False)  # Setting this to False generally improves performance
    model = ViTDiffMAEForPreTraining(config=config).to(device)

    # Optimizer and Learning Rate Scheduler
    lr = args.learning_rate
    optim = torch.optim.AdamW(model.parameters(), lr=lr, betas=[0.9, 0.95], 
                              weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-6), 
                                0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)
    
    writer = SummaryWriter(os.path.join(OUTPUT_DIR, 'logs'))
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.ddpm_num_steps, 
                                    beta_schedule=args.ddpm_beta_schedule)

    # Set Callback class for decoding hair .obj
    callback = CallbackDecodeHair(args)

    # Training Loop
    global_step = 0
    for epoch in range(args.total_epoch):
        model.train()
        losses = []
        progress_bar = tqdm(total=len(dataloader), desc=f"Epoch {epoch}")
        
        for step, batch in enumerate(dataloader):
            global_step += 1
            img_clean = batch['images'].to(device)
            baldness_map = batch['baldness'].to(device)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                      [len(img_clean),], device=device).long()
            noise = torch.randn(img_clean.shape, device=device)
            img_noise = noise_scheduler.add_noise(img_clean, noise, timesteps)
            
            # outputs = model(img_clean, img_noise, timesteps)
            outputs = model(img_clean, img_noise, timesteps, baldness_map)
            loss = outputs.loss
            loss.backward()
            optim.step()
            optim.zero_grad()

            losses.append(loss.detach().item())
            writer.add_scalar('mae_loss', np.mean(losses), global_step)
            progress_bar.set_postfix(loss=np.mean(losses), lr=lr_scheduler.get_last_lr()[0])
            progress_bar.update()

        lr_scheduler.step()
        progress_bar.close()

        # Saving Model and Visualization
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'checkpoints/model_epoch_{}.pth'.format(epoch)))
            model.eval()
            with torch.no_grad():
                batch = next(iter(dataloader))   
                img_clean = batch['images'].to(device)      # torch.Size([64(B), 64(C), 256(H), 256(W)])
                baldness_map = batch['baldness'].to(device) # torch.Size([64(B), 64(C), 256(H), 256(W)])
                timesteps = torch.full([len(img_clean),], noise_scheduler.config.num_train_timesteps-1, device=device).long()
                noise = torch.randn(img_clean.shape).to(device)

                time_schedules = torch.cat([noise_scheduler.timesteps[0::500], torch.tensor(0)[None]]).to(device)
                for i in tqdm(range(len(time_schedules)-1)):
                    if i == 0:
                        img_noise = noise_scheduler.add_noise(img_clean, noise, time_schedules[i])
                        # outputs = model(img_clean, img_noise, time_schedules[i])
                        outputs = model(img_clean, img_noise, time_schedules[i], baldness_map)
                        noise_seed = outputs.noise.detach()

                    else:
                        # 2. compute previous image: x_t -> x_t-1
                        img_noise = noise_scheduler.add_noise(img_pred, img_noise, time_schedules[i+1])
                        # outputs = model(img_clean, img_noise, time_schedules[i], noise=noise_seed)
                        outputs = model(img_clean, img_noise, time_schedules[i], baldness_map, noise=noise_seed)
                    
                    # 1. predict noise model_output
                    img_pred = outputs.logits.detach()
                    img_pred = model.unpatchify(img_pred)


                    img_pred = img_pred.clamp(-1, 1)    # torch.Size([64(B), 64(C), 256(H), 256(W)])
                    img_pred_save = (img_pred + 1) / 2 * (data_max - data_min) + data_min     # range [data_min, data_max]
                    img_clean_save = (img_clean + 1) / 2 * (data_max - data_min) + data_min   # range [data_min, data_max]
                    # img_pred = img_pred.clamp(0, 1)    # torch.Size([64(B), 64(C), 256(H), 256(W)])
                    # img_pred_save = img_pred * (data_max - data_min) + data_min     # range [data_min, data_max]
                    # img_clean_save = img_clean * (data_max - data_min) + data_min   # range [data_min, data_max]
                    
                    if epoch % 100 == 0:
                        img_pred_save_dir = os.path.join(OUTPUT_DIR, 'callback/epoch_%04d_img_pred_nor_%04d.pt' % (epoch, time_schedules[i].item()))
                        img_clean_save_dir = os.path.join(OUTPUT_DIR, 'callback/epoch_%04d_img_clean_nor_%04d.pt' % (epoch, time_schedules[i].item()))
                        torch.save(img_pred_save.detach().cpu(), img_pred_save_dir)
                        torch.save(img_clean_save.detach().cpu(), img_clean_save_dir)
                        
                        
                        mask = outputs.mask.detach()
                        mask = mask.unsqueeze(-1).repeat(1, 1, model.config.patch_size**2 * model.config.num_channels)  # (N, H*W, p*p*3)
                        mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping


                        # decode and save(.obj) the callback shape texture
                        callback.run_callback(img_pred_save_dir, img_clean_save_dir, mask)

                mask = outputs.mask.detach()
                mask = mask.unsqueeze(-1).repeat(1, 1, model.config.patch_size**2 * model.config.num_channels)  # (N, H*W, p*p*3)
                mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping

                mask = mask.detach().cpu()[:,:3]
                noise = noise.detach().cpu()[:,:3]

                img_pred = img_pred_save.detach().cpu()[:, :3]
                img_clean = img_clean_save.detach().cpu()[:, :3]
                img_noise = img_noise.detach().cpu()[:, :3]
                # baldness_map = baldness_map.detach().cpu()[:, :3]
                writer.add_images('gt_image', (img_clean + 1) / 2, epoch)
                writer.add_images('mae_image', (img_pred + 1) / 2, epoch)
                writer.add_images('input_image', ((img_clean * (1 - mask) + noise * (mask)) + 1) / 2, epoch)
                writer.add_images('output_image', ((img_clean * (1 - mask) + img_pred * (mask)) + 1) / 2, epoch)
                # writer.add_images('baldnessmap', baldness_map, epoch)

from pathlib import Path
from shutil import copyfile
def file_backup(output_dir):
    dir_lis = ["/hdd_sda1/tonghs/workspace/SparseHair/scripts/train_hairdiffmae.sh", 
               "/hdd_sda1/tonghs/workspace/SparseHair/HairDiffMAE/main_diffmae_hair_ths.py", 
               "/hdd_sda1/tonghs/workspace/SparseHair/HairDiffMAE/data_loader_ths.py",
               "/hdd_sda1/tonghs/workspace/SparseHair/HairDiffMAE/transformers/src/transformers/models/vit_diffmae/modeling_vit_diffmae.py",
               "/hdd_sda1/tonghs/workspace/SparseHair/HairDiffMAE/transformers/src/transformers/models/vit_diffmae/configuration_vit_diffmae.py"]
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # General Training Parameters
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device_num', type=int, default=0)
    parser.add_argument("--data_dir", type=str, required=True)
    # Training Parameters DiffMAE
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--num_hidden_layers', type=int, default=12)
    parser.add_argument('--num_attention_heads', type=int, default=12)
    parser.add_argument('--intermediate_size', type=int, default=3072)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=16)   # 16x16 = 256 patches    
    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--decoder_num_attention_heads', type=int, default=16)
    parser.add_argument('--decoder_hidden_size', type=int, default=512)
    parser.add_argument('--decoder_num_hidden_layers', type=int, default=8)
    parser.add_argument('--decoder_intermediate_size', type=int, default=2048)

    parser.add_argument('--mask_ratio', type=float, default=0.75) # ratio to be removed = mask patches / all patches
    parser.add_argument('--learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--total_epoch', type=int, default=100001)
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--warmup_epoch', type=int, default=200)
    # Model parameters StrandVAE
    parser.add_argument('--dim_in', type=int, default=1, help='Input dimension for the model')
    parser.add_argument('--dim_hidden', type=int, default=256, help='Hidden dimension for the model')
    parser.add_argument('--dim_out', type=int, default=6, help='Output dimension for the model (6 for 6DoF, 3 for Euclidean)')
    parser.add_argument('--num_layers', type=int, default=5, help='Number of layers in the model')
    parser.add_argument('--w0_initial', type=float, default=30.0, help='Initial value for w0 in SIREN')
    parser.add_argument('--latent_dim', type=int, default=64, help='Latent dimension for VAE')
    parser.add_argument('--coord_length', type=int, default=100, help='Coordinate length')   
    # Pre-trained StrandVAE
    parser.add_argument('--model_ckpt', type=str, required=True, help='Path to the pre-trained strandvae model ckpt')
    parser.add_argument('--uv_color_map', type=str, default='data/head_template/head_template.png', help='UV color map path')
    args = parser.parse_args()

    main(args)
