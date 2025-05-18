import os
import argparse
import torch
from torch.utils import data
from glob import glob
import numpy as np
from tqdm import tqdm

class HairDiffMAEDataset(data.Dataset):
    """A HairDiffMAE Dataset class compatible with PyTorch DataLoader."""
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        return self.data[index]

    def __len__(self):
        """Returns the total number of data samples."""
        return len(self.data)


def read_data(args):
    """Load and preprocess data."""
    print("Loading data...")
    data_list = np.array(glob(os.path.join(args.data_dir, 'latent/*.pt')))[:10000]  # ~ 21054
    
    # Load normalization statistics
    data_max = torch.load(os.path.join(args.data_dir, 'min_max/data_max.pt'), weights_only=False)
    data_min = torch.load(os.path.join(args.data_dir, 'min_max/data_min.pt'), weights_only=False)

    train_ratio = 0.8
    n_train = int(len(data_list) * train_ratio)
    print(f'n_train: {n_train}, n_test: {len(data_list) - n_train}')

    train_data, test_data = [], []
    for data_name in tqdm(data_list):
        # Load pt tensor and handle NaNs
        pt = torch.load(data_name, weights_only=False).permute(2, 0, 1).float() 

        # Create baldnessmap based on NaNs in pt -> 1 if pt is not NaN, otherwise 0
        baldnessmap = (~torch.isnan(pt)).float().clamp(0, 1)

        # Replace NaNs in pt with 0s
        pt = pt.nan_to_num(0)        

        # pt = (pt - data_min) / (data_max - data_min + 1.0e-6)  # Normalize data, avoid nan # make pt in [0, 1]
        pt = 2 * (pt - data_min) / (data_max - data_min + 1.0e-6) - 1 # Normalize data, avoid nan # make pt in [-1, 1]

        data_item = {"key": data_name, "images": pt, "baldness": baldnessmap}
        train_data.append(data_item) if len(train_data) < n_train else test_data.append(data_item)

    print(f'\nData Partition: {len(train_data)} for training, {len(test_data)} for testing')
    return train_data, test_data


def get_dataloaders(args):
    """Create DataLoader for training and testing."""
    train_data, test_data = read_data(args)
    
    # Initialize datasets
    train_dataset = HairDiffMAEDataset(train_data)
    test_dataset = HairDiffMAEDataset(test_data)
    
    # Create DataLoaders
    dataset = {
        "train": data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16),
        "test": data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)
    }
    return dataset







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Define command line arguments
    parser.add_argument("--data_dir", type=str, default="/mnt/mnt/Database1/USC-hairsalon/neural_texture/optim_neural_textures")
    parser.add_argument("--batch_size", type=int, default=16)
    # Add additional arguments as needed
    args = parser.parse_args()

    # Set CUDA device
    torch.multiprocessing.set_start_method('spawn')

    # Load datasets
    dataset = get_dataloaders(args)
    
    # Example: Iterate through the training dataset
    for i, batch in enumerate(dataset['train']):
        print(batch['images'].shape)
