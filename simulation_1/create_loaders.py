from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import random_split, DataLoader
import torch

# Returns transformed train and test data from torchvision.datasets
def get_train_test(data_path : str = "./") :

    # Arguments for Normalize are mean and variance for single scalar data in this case (MNIST is grayscale)
    trans_fn = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    # 'transform' transforms the calling function's dataset to a network-transferrable format
    train_set = MNIST(data_path, train=True, download=True, transform=trans_fn)
    test_set = MNIST(data_path, train=False, download=True, transform=trans_fn)
    
    return train_set, test_set


# Partioning the data for FL -> returns dataloaders, not the data itself
# WHAT IS BATCH_SIZE HERE?
# 3 main ways of partitioning : IID, Non-IID, Pathological (N labels, each client at most M labels)

def prepare_data(num_partitions : int, batch_size : int, val_ratio : float) :
    # Getting big train_set and big test_set
    train_set, test_set = get_train_test()

    # Partitioning big train_set among all clients
    fraction_images_per_partition = num_partitions * [len(train_set) // num_partitions]

    # train_sets = list of datasets for all partitions
    train_sets = random_split(train_set, fraction_images_per_partition, torch.Generator().manual_seed(42))

    # Creating list of dataloaders for train and validation sets
    train_loaders = []
    val_loaders = []

    # For each client's dataset -> splitting into train and validation
    for part_train_set in train_sets :
        total_imgs = len(part_train_set)
        val_imgs = int(val_ratio * total_imgs)
        train_imgs = total_imgs - val_imgs

        final_train_set, final_validation_set = random_split(part_train_set, [train_imgs, val_imgs])

        # batch_size here refers to the number of samples passed to the model after which an update is done
        train_loaders.append(DataLoader(final_train_set, batch_size = batch_size, shuffle = True, num_workers = 2))
        val_loaders.append(DataLoader(final_validation_set, batch_size = batch_size, num_workers = 2))
    

    # batch_size here refers to the number of parallel predictions made
    test_loader = DataLoader(test_set, batch_size = 128)

    return train_loaders, val_loaders, test_loader


