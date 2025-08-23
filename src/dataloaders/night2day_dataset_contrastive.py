"""
Night2Day Dataset Loader for Contrastive Learning

This module provides a custom dataset loader for the Night2Day dataset that supports
contrastive learning by providing both joint (paired) and independent (shuffled) 
image pairs with corresponding labels.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import random
from pathlib import Path
from typing import Tuple, Optional, Union


class Night2DayDataset(Dataset):
    """
    Custom dataset for Night2Day images supporting contrastive learning.
    
    Args:
        data_path: Path to the dataset directory
        resolution: Target resolution for images (default: 64)
        contrastive: If True, returns mixed joint/independent pairs with labels
                    If False, returns only joint pairs without labels
        train: Whether this is training data (affects data augmentation)
        split_ratio: Ratio of joint vs independent pairs when contrastive=True (default: 0.5)
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        resolution: int = 64,
        contrastive: bool = False,
        train: bool = True,
        split_ratio: float = 0.5
    ):
        self.data_path = Path(data_path)
        self.resolution = resolution
        self.contrastive = contrastive
        self.train = train
        self.split_ratio = split_ratio
        
        # Load all image paths
        self.image_paths = list(self.data_path.rglob("*.jpg"))
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.data_path}")
        
        # Set up transforms
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])
        
        # For data augmentation during training
        if train:
            self.augment_transform = transforms.Compose([
                transforms.Resize((resolution + 8, resolution + 8)),
                transforms.RandomCrop((resolution, resolution)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.augment_transform = self.transform
            
        print(f"Loaded {len(self.image_paths)} images from {self.data_path}")
        print(f"Resolution: {resolution}x{resolution}")
        print(f"Contrastive mode: {contrastive}")
        if contrastive:
            print(f"Joint/Independent split ratio: {split_ratio}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def _load_and_split_image(self, img_path: Path) -> Tuple[Image.Image, Image.Image]:
        """Load an image and split it into two parts."""
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        
        if width == 512 and height == 256:
            # Split horizontally: left and right
            img_a = img.crop((0, 0, 256, 256))
            img_b = img.crop((256, 0, 512, 256))
        elif width == 256 and height == 512:
            # Split vertically: top and bottom
            img_a = img.crop((0, 0, 256, 256))
            img_b = img.crop((0, 256, 256, 512))
        else:
            raise ValueError(f"Unexpected image size {width}x{height} for {img_path}")
        
        return img_a, img_b
    
    def _get_joint_pair(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a joint (paired) image pair."""
        img_path = self.image_paths[idx]
        img_a, img_b = self._load_and_split_image(img_path)
        
        # Apply transforms
        if self.train:
            # Apply same random seed for both images to ensure consistent augmentation
            seed = random.randint(0, 2**32 - 1)
            
            random.seed(seed)
            torch.manual_seed(seed)
            img_a_tensor = self.augment_transform(img_a)
            
            random.seed(seed)
            torch.manual_seed(seed)
            img_b_tensor = self.augment_transform(img_b)
        else:
            img_a_tensor = self.transform(img_a)
            img_b_tensor = self.transform(img_b)
        
        return img_a_tensor, img_b_tensor
    
    def _get_independent_pair(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get an independent (shuffled) image pair."""
        # Get first image from current index
        img_path_a = self.image_paths[idx]
        img_a, _ = self._load_and_split_image(img_path_a)
        
        # Get second image from random index
        random_idx = random.randint(0, len(self.image_paths) - 1)
        img_path_b = self.image_paths[random_idx]
        _, img_b = self._load_and_split_image(img_path_b)
        
        # Apply transforms independently
        img_a_tensor = self.augment_transform(img_a) if self.train else self.transform(img_a)
        img_b_tensor = self.augment_transform(img_b) if self.train else self.transform(img_b)
        
        return img_a_tensor, img_b_tensor
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], 
                                           Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Get an item from the dataset.
        
        Returns:
            If contrastive=False: (img_a, img_b)
            If contrastive=True: (img_a, img_b, label) where label=1 for joint, 0 for independent
        """
        if not self.contrastive:
            # Always return joint pairs
            img_a, img_b = self._get_joint_pair(idx)
            return img_a, img_b
        else:
            # Return mixed joint/independent pairs with labels
            if random.random() < self.split_ratio:
                # Return joint pair
                img_a, img_b = self._get_joint_pair(idx)
                label = torch.tensor(1.0, dtype=torch.float32)
            else:
                # Return independent pair
                img_a, img_b = self._get_independent_pair(idx)
                label = torch.tensor(0.0, dtype=torch.float32)
            
            return img_a, img_b, label


def create_dataloaders(
    train_path: Union[str, Path],
    val_path: Optional[Union[str, Path]] = None,
    test_path: Optional[Union[str, Path]] = None,
    resolution: int = 64,
    contrastive: bool = False,
    batch_size: int = 32,
    num_workers: int = 4,
    split_ratio: float = 0.5
) -> dict:
    """
    Create DataLoaders for Night2Day dataset.
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data (optional)
        test_path: Path to test data (optional)
        resolution: Target image resolution
        contrastive: Whether to use contrastive learning mode
        batch_size: Batch size for DataLoaders
        num_workers: Number of workers for data loading
        split_ratio: Ratio of joint vs independent pairs for contrastive learning
    
    Returns:
        Dictionary containing DataLoaders for available splits
    """
    dataloaders = {}
    
    # Training dataset
    train_dataset = Night2DayDataset(
        train_path, 
        resolution=resolution,
        contrastive=contrastive,
        train=True,
        split_ratio=split_ratio
    )
    dataloaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Validation dataset
    if val_path:
        val_dataset = Night2DayDataset(
            val_path,
            resolution=resolution,
            contrastive=contrastive,
            train=False,
            split_ratio=split_ratio
        )
        dataloaders['val'] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    # Test dataset
    if test_path:
        test_dataset = Night2DayDataset(
            test_path,
            resolution=resolution,
            contrastive=contrastive,
            train=False,
            split_ratio=split_ratio
        )
        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return dataloaders


if __name__ == "__main__":
    # Example usage
    train_path = "../../data/night2day/train"
    
    # Test regular mode
    print("=== Testing Regular Mode ===")
    dataloaders_regular = create_dataloaders(
        train_path=train_path,
        resolution=64,
        contrastive=False,
        batch_size=4
    )
    
    batch = next(iter(dataloaders_regular['train']))
    print(f"Regular mode batch shapes: {batch[0].shape}, {batch[1].shape}")
    
    # Test contrastive mode
    print("\n=== Testing Contrastive Mode ===")
    dataloaders_contrastive = create_dataloaders(
        train_path=train_path,
        resolution=64,
        contrastive=True,
        batch_size=4
    )
    
    batch = next(iter(dataloaders_contrastive['train']))
    print(f"Contrastive mode batch shapes: {batch[0].shape}, {batch[1].shape}, {batch[2].shape}")
    print(f"Labels: {batch[2]}")
