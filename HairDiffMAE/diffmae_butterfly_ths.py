import os
import math
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from HairDiffMAE.transformers.src.transformers.models.vit_diffmae import ViTDiffMAEForPreTraining, ViTDiffMAEConfig
from HairDiffMAE.diffusers.src.diffusers.schedulers import DDPMScheduler
from datetime import datetime


# Base directory where the script is located
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def transform(examples):
    preprocess = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}

def main(args):
    setup_seed(args.seed)
    device = 'cuda:{}'.format(args.device_num) if torch.cuda.is_available() else 'cpu'
    OUTPUT_DIR = os.path.join(BASE_DIR, "output", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    print(OUTPUT_DIR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'callback'), exist_ok=True)

    # Dataset
    from datasets import load_dataset
    dataset = load_dataset("huggan/smithsonian_butterflies_subset")['train']
    dataset.set_transform(transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)


    # Network Configuration and Model Initialization
    config = ViTDiffMAEConfig(num_channels=3, image_size=args.image_size, 
                              patch_size=args.patch_size, mask_ratio=args.mask_ratio)
    model = ViTDiffMAEForPreTraining(config=config).to(device)

    # Optimizer and Learning Rate Scheduler
    lr = args.learning_rate
    optim = torch.optim.AdamW(model.parameters(), lr=lr, betas=[0.9, 0.95], 
                              weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 
                                0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)
    
    writer = SummaryWriter(os.path.join(OUTPUT_DIR, 'logs'))
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.ddpm_num_steps, 
                                    beta_schedule=args.ddpm_beta_schedule)

    # Training Loop
    global_step = 0
    for epoch in range(args.total_epoch):
        model.train()
        losses = []
        progress_bar = tqdm(total=len(dataloader), desc=f"Epoch {epoch}")
        
        for step, batch in enumerate(dataloader):
            global_step += 1
            img_clean = batch['images'].to(device)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                      [len(img_clean),], device=device).long()
            noise = torch.randn(img_clean.shape, device=device)
            img_noise = noise_scheduler.add_noise(img_clean, noise, timesteps)
            
            outputs = model(img_clean, img_noise, timesteps)
            loss = outputs.loss
            loss.backward()
            optim.step()
            optim.zero_grad()

            losses.append(loss.detach().item())
            writer.add_scalar('mae_loss', np.mean(losses), global_step)
            progress_bar.set_postfix(loss=np.mean(losses))
            progress_bar.update()

        lr_scheduler.step()
        progress_bar.close()

        # Saving Model and Visualization
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'checkpoints/model_epoch_{}.pth'.format(epoch)))
            model.eval()
            with torch.no_grad():
                batch = next(iter(dataloader))   
                img_clean = batch['images'].to(device)  # torch.Size([64(B), 64(C), 256(H), 256(W)])
                timesteps = torch.full([len(img_clean),], noise_scheduler.config.num_train_timesteps-1, device=device).long()
                noise = torch.randn(img_clean.shape).to(device)

                time_schedules = torch.cat([noise_scheduler.timesteps[0::100], torch.tensor(0)[None]]).to(device)
                for i in tqdm(range(len(time_schedules)-1)):
                    if i == 0:
                        img_noise = noise_scheduler.add_noise(img_clean, noise, time_schedules[i])

                        outputs = model(img_clean, img_noise, time_schedules[i])
                        noise_seed = outputs.noise.detach()
                    else:
                        # 2. compute previous image: x_t -> x_t-1
                        img_noise = noise_scheduler.add_noise(img_pred, img_noise, time_schedules[i+1])

                        outputs = model(img_clean, img_noise, time_schedules[i], noise=noise_seed)

                    # 1. predict noise model_output
                    img_pred = outputs.logits.detach()
                    img_pred = model.unpatchify(img_pred)   # torch.Size([64(B), 64(C), 256(H), 256(W)])

                    mask = outputs.mask.detach()
                    mask = mask.unsqueeze(-1).repeat(1, 1, model.config.patch_size**2 * model.config.num_channels)  # (N, H*W, p*p*3)
                    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping

                # img_pred = img_pred[:, :3]
                # img_clean = img_clean[:, :3]
                # noise = noise[:, :3]
                # mask = mask[:, :3]

                writer.add_images('gt_image', (img_clean.detach().cpu() + 1) / 2, epoch)
                writer.add_images('mae_image', (img_pred.detach().cpu() + 1) / 2, epoch)
                writer.add_images('input_image', ((img_clean * (1 - mask) + noise * (mask)).detach().cpu() + 1) / 2, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # General Training Parameters
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device_num', type=int, default=2)
    # Training Parameters DiffMAE
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=16)   # 16x16 = 256 patches
    parser.add_argument('--mask_ratio', type=float, default=0.75) # ratio to be removed = mask patches / all patches
    parser.add_argument('--learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--total_epoch', type=int, default=10000)
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--warmup_epoch', type=int, default=200)
    args = parser.parse_args()

    main(args)
