from collections import Counter

from utils.config import cfg  # your config file
import os
from torch.utils.data import DataLoader
from torchvision import datasets


def build_dataloaders(model, data_root, preprocess, debug=False):
    """
    Crea i DataLoader di Training e Validazione usando il Dataset di Tiny ImageNet (or ImageFolder datasets).

    Args:
        model: timm model (efficientNET) (needed for preprocessing config)
        data_root (str): path to dataset root (expects 'train' and 'val' subfolders)
        debug (bool): true to show debug prints, false otherwise
    Returns:
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
    """

    train_set = datasets.ImageFolder(os.path.join(data_root, "train"), transform=preprocess)
    val_set   = datasets.ImageFolder(os.path.join(data_root, "val"), transform=preprocess)

    if(debug): print_dataloaders_values(train_set, val_set)
    print("DEBUG: Using validation transform:", preprocess)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def print_dataloaders_values(train_set, val_set):
    """
    Dati i set di Training e Validazione nè stampa il numero di immagini per classe.
    :param train_set:
    :param val_set:
    :return:
    """
    print(f"📊 Dataset sizes:")
    print(f"   Train: {len(train_set)} images")
    print(f"   Val:   {len(val_set)} images")
    train_counts = Counter([label for _, label in train_set.samples])
    val_counts = Counter([label for _, label in val_set.samples])

    print("\n🔹 Per-class sample counts (Train, first 5 classes):")
    for idx, count in list(train_counts.items())[:5]:
        print(f"   Class {idx}: {count} images")

    print("\n🔹 Per-class sample counts (Val, first 5 classes):")
    for idx, count in list(val_counts.items())[:5]:
        print(f"   Class {idx}: {count} images")
