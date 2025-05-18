import os
import torch
import warnings
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path
from shutil import copyfile

from StrandVAE.model.strand_vae import StrandVAE, StrandVAE_1, StrandVAE_2, StrandVAE_3
from StrandVAE.util.arguments import train_strand_vae_args
from StrandVAE.util.state import Stats
from StrandVAE.util.utils import *
from StrandVAE.data.hair20k_dataset import Hair20k
from StrandVAE.model.component.compute_loss import ComputeLossStrandVAE

# Base directory where the script is located
BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def main():
    args = train_strand_vae_args()

    # Create output directories with current timestamp
    OUTPUT_DIR = os.path.join(BASE_DIR, "output", "strand_vae", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Resolve paths relative to BASE_DIR
    data_dir = os.path.join(BASE_DIR, args.data_dir)
    tensorboard_log_dir = os.path.join(OUTPUT_DIR, args.tensorboard_log_dir)
    callback_path = os.path.join(OUTPUT_DIR, args.callback_path)
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    os.makedirs(callback_path, exist_ok=True)

    # Set tensorboard writer
    writer = SummaryWriter(tensorboard_log_dir)

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
    model = StrandVAE(**model_params)

    # Move the model to the relevant device
    model.to(device)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate
    )

    # Set up the dataset and dataloaders
    train_dataset = Hair20k(data_dir, split='train')
    valid_dataset = Hair20k(data_dir, split='valid')

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8  # Adjust based on your system
    )

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8  # Adjust based on your system
    )
    train_max_it = len(train_dataset) // args.batch_size
    valid_max_it = len(valid_dataset) // args.batch_size

    # Define the loss function
    loss_params = {
        'l_main_mse': args.l_main_mse,
        'l_main_cos': args.l_main_cos,
        'l_kld': args.l_kld
    }
    compute_loss = ComputeLossStrandVAE(loss_params)

    # Initialize stats object
    stats = Stats(
        ["loss", "sec/it", *loss_params.keys()]
    )

    # Backup files for recording
    file_backup(OUTPUT_DIR)

    # Training loop
    global_step = 0
    for epoch in range(train_max_it):

        # Adjust the learning rate, loss weight after epoch 500, 1000, 1500
        if epoch == 500:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 5e-5
            print("Learning rate adjusted to 5e-5 after epoch 500")
        if epoch == 1000:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-5
            print("Learning rate adjusted to 1e-5 after epoch 1000")
        if epoch == 1500:
            compute_loss.loss_dict['l_main_mse'] = 1.0
            compute_loss.loss_dict['l_main_cos'] = 0.1
            compute_loss.loss_dict['l_kld'] = 0.001


        model.train()
        stats.new_epoch()
        for iteration, batch in enumerate(train_dataloader):
            # Unpack your batch data
            # For example:
            strands_tan, _, _ = batch
            strands_tan = strands_tan.to(device)

            optimizer.zero_grad()

            # Run the forward pass
            loss, log_dict = compute_loss(model, strands_tan)
            loss.backward()
            optimizer.step()

            # Update stats and tensorboard
            stats.update(log_dict, stat_set="train")
            if iteration % args.stats_print_interval == 0:
                stats.print(max_it=len(train_dataloader), stat_set="train")

            writer.add_scalar('Train/Loss', loss.item(), global_step)
            for key, value in log_dict.items():
                writer.add_scalar(f'train/{key}', value, global_step)            
                
            global_step += 1

        # Validation
        if epoch % args.validation_epoch_interval == 0:
            global_step_valid = 0
            model.eval()
            with torch.no_grad():
                for iteration, batch in enumerate(valid_dataloader):
                    strands_tan, _, _ = batch
                    strands_tan = strands_tan.to(device)

                    _, log_dict = compute_loss(model, strands_tan)

                    # Update stats with the validation metrics.
                    stats.update(log_dict, stat_set="val")
                    stats.print(max_it=valid_max_it, stat_set="val")

                    if args.tensorboard_log_dir:
                        for key, value in log_dict.items():
                            writer.add_scalar(f'valid/{key}', value, global_step_valid)
                    global_step_valid += 1

                # Callback
                if args.callback_path:
                    source_strand = torch.stack([valid_dataset[idx][0].to(device) for idx in range(10)], dim=0)
                    recon, _, _, _ = model(source_strand)

                    source_color = torch.tensor([1, 0, 0])  # RED   : GT strand
                    target_color = torch.tensor([0, 0, 1])  # BLUE  : Generated strand

                    for idx, (s, r) in enumerate(zip(source_strand, recon)):
                        s_c = source_color.expand_as(s)
                        r_c = target_color.expand_as(r)

                        save_obj = torch.cat([s, r]).cpu().numpy()
                        save_color = torch.cat([s_c, r_c]).cpu().numpy()
                        save_obj_path = os.path.join(callback_path, f'recon_epoch{epoch:>04}_num{idx:>03}.obj')
                        save_hair(save_obj_path, save_obj, color=save_color)
            model.train()

        # Checkpointing
        if epoch % args.checkpoint_epoch_interval == 0:
            ckpt_path = os.path.join(OUTPUT_DIR, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
            print(f"Checkpoint saved at {ckpt_path}")

    writer.close()

def file_backup(output_dir):
    dir_lis = ["/hdd_sda1/tonghs/workspace/SparseHair/scripts/train_strand_vae.sh", 
               "/hdd_sda1/tonghs/workspace/SparseHair/StrandVAE/train_strand_vae.py", 
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



if __name__ == '__main__':
    main()
