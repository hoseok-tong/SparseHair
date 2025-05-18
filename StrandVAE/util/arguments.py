import argparse

def train_strand_vae_args():
    parser = argparse.ArgumentParser(description='Train StrandVAE model')

    # General parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device_num', type=int, default=0, help='GPU device number to use')
    parser.add_argument('--max_epochs', type=int, default=5001, help='Maximum number of epochs to train')
    parser.add_argument('--stats_print_interval', type=int, default=100, help='Interval of printing stats')
    parser.add_argument('--validation_epoch_interval', type=int, default=100, help='Validation epoch interval')
    parser.add_argument('--checkpoint_epoch_interval', type=int, default=100, help='Checkpoint saving epoch interval')

    # TensorBoard and callback paths
    parser.add_argument('--tensorboard_log_dir', type=str, default='tensorboard/', help='TensorBoard log directory')
    parser.add_argument('--callback_path', type=str, default='callback/', help='Callback directory')

    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data/', help='Data directory')
    parser.add_argument('--batch_size', type=int, default=2000, help='Batch size for training')

    # Model parameters
    parser.add_argument('--dim_in', type=int, default=1, help='Input dimension for the model')
    parser.add_argument('--dim_hidden', type=int, default=128, help='Hidden dimension for the model')
    parser.add_argument('--dim_out', type=int, default=6, help='Output dimension for the model (6 for 6DoF, 3 for Euclidean)')
    parser.add_argument('--num_layers', type=int, default=5, help='Number of layers in the model')
    parser.add_argument('--w0_initial', type=float, default=30.0, help='Initial value for w0 in SIREN')
    parser.add_argument('--latent_dim', type=int, default=64, help='Latent dimension for VAE')
    parser.add_argument('--coord_length', type=int, default=100, help='Coordinate length')

    # Loss parameters
    parser.add_argument('--l_main_mse', type=float, default=1.0, help='Weight for main MSE loss')
    parser.add_argument('--l_main_cos', type=float, default=1.0, help='Weight for main cosine loss')
    parser.add_argument('--l_kld', type=float, default=0.0001, help='Weight for KLD loss')

    # Optimizer parameters
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')

    # Add other arguments as needed
    # For example, if you decide to use a learning rate scheduler:
    # parser.add_argument('--lr_scheduler_gamma', type=float, default=0.95, help='LR scheduler gamma')
    # parser.add_argument('--lr_scheduler_step_size', type=int, default=10, help='LR scheduler step size')

    args = parser.parse_args()
    return args



def get_shape_texture_args():
    parser = argparse.ArgumentParser(description='Train StrandVAE model')

    # General parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device_num', type=int, default=0, help='GPU device number to use')
    parser.add_argument('--max_epochs', type=int, default=5001, help='Maximum number of epochs to train')
    parser.add_argument('--stats_print_interval', type=int, default=100, help='Interval of printing stats')
    parser.add_argument('--validation_epoch_interval', type=int, default=100, help='Validation epoch interval')
    parser.add_argument('--checkpoint_epoch_interval', type=int, default=100, help='Checkpoint saving epoch interval')

    # TensorBoard and callback paths
    parser.add_argument('--tensorboard_log_dir', type=str, default='tensorboard/', help='TensorBoard log directory')
    parser.add_argument('--callback_path', type=str, default='callback/', help='Callback directory')

    # Data parameters
    parser.add_argument('--model_ckpt', type=str, required=True, help='Path to the pre-trained strandvae model ckpt')
    parser.add_argument('--data_dir', type=str, default='data/', help='Data directory')
    parser.add_argument('--batch_size', type=int, default=2000, help='Batch size for training')
    parser.add_argument('--uv_map_size', type=int, default=256, help='UV map size for shape texture')
    parser.add_argument('--interp_method', type=str, default='linear', help='Interpolation method')

    # Model parameters
    parser.add_argument('--dim_in', type=int, default=1, help='Input dimension for the model')
    parser.add_argument('--dim_hidden', type=int, default=128, help='Hidden dimension for the model')
    parser.add_argument('--dim_out', type=int, default=6, help='Output dimension for the model (6 for 6DoF, 3 for Euclidean)')
    parser.add_argument('--num_layers', type=int, default=5, help='Number of layers in the model')
    parser.add_argument('--w0_initial', type=float, default=30.0, help='Initial value for w0 in SIREN')
    parser.add_argument('--latent_dim', type=int, default=64, help='Latent dimension for VAE')
    parser.add_argument('--coord_length', type=int, default=100, help='Coordinate length')

    # Loss parameters
    parser.add_argument('--l_main_mse', type=float, default=1.0, help='Weight for main MSE loss')
    parser.add_argument('--l_main_cos', type=float, default=1.0, help='Weight for main cosine loss')
    parser.add_argument('--l_kld', type=float, default=0.0001, help='Weight for KLD loss')

    # Optimizer parameters
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')


    args = parser.parse_args()
    return args




def decode_hair_args():
    parser = argparse.ArgumentParser(description='Train StrandVAE model')

    # General parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device_num', type=int, default=0, help='GPU device number to use')

    # Data parameters
    parser.add_argument('--model_ckpt', type=str, required=True, help='Path to the pre-trained strandvae model ckpt')
    parser.add_argument('--data_dir', type=str, default='data/', help='Data directory')
    parser.add_argument('--uv_color_map', type=str, default='data/head_template/head_template.png', help='UV color map path')

    # Model parameters
    parser.add_argument('--dim_in', type=int, default=1, help='Input dimension for the model')
    parser.add_argument('--dim_hidden', type=int, default=128, help='Hidden dimension for the model')
    parser.add_argument('--dim_out', type=int, default=6, help='Output dimension for the model (6 for 6DoF, 3 for Euclidean)')
    parser.add_argument('--num_layers', type=int, default=5, help='Number of layers in the model')
    parser.add_argument('--w0_initial', type=float, default=30.0, help='Initial value for w0 in SIREN')
    parser.add_argument('--latent_dim', type=int, default=64, help='Latent dimension for VAE')
    parser.add_argument('--coord_length', type=int, default=100, help='Coordinate length')


    args = parser.parse_args()
    return args