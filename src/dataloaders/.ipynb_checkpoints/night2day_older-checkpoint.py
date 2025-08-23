"""
night2day_mi.py
Night2Day dataset for MI training - exactly like AVE but for image pairs
Real pairs: night-day from same location
Fake pairs: shuffled night-day from different locations
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import random
from pathlib import Path
from typing import Tuple, Dict, Optional, Union


class Night2DayMIDataset(Dataset):
    """
    MI training dataset with real and fake night-day pairs.
    Exactly like AVE dataset but for image pairs.
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        resolution: int = 64,
        real_fake_ratio: float = 0.5,
        augment: bool = False,
        debug: bool = False
    ):
        self.data_path = Path(data_path)
        self.resolution = resolution
        self.real_fake_ratio = real_fake_ratio
        self.augment = augment
        
        # Load all paired images
        self.image_paths = sorted(list(self.data_path.glob("*.jpg")) + 
                                 list(self.data_path.glob("*.png")))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.data_path}")
        
        if debug:
            print(f"Real samples: {len(self.image_paths)}")
            print(f"Resolution: {resolution}x{resolution}")
        
        # Transforms - normalize to [-1, 1] for diffusion models
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((resolution + 8, resolution + 8)),
                transforms.RandomCrop((resolution, resolution)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((resolution, resolution)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
    
    def _split_paired_image(self, img_path: Path) -> Tuple[Image.Image, Image.Image]:
        """Split paired image into two parts."""
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        
        # Handle different layouts
        if width > height:  # Horizontal split
            mid = width // 2
            img1 = img.crop((0, 0, mid, height))
            img2 = img.crop((mid, 0, width, height))
        else:  # Vertical split
            mid = height // 2
            img1 = img.crop((0, 0, width, mid))
            img2 = img.crop((0, mid, width, height))
        
        return img1, img2
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns dict matching AVE structure:
            - audio: First image (night) [3, H, W]
            - image: Second image (day) [3, H, W]  
            - is_real: 1.0 for real pairs, 0.0 for fake
        """
        # Decide real or fake
        is_real = random.random() < self.real_fake_ratio
        
        if is_real:
            # Real pair from same image
            img_path = self.image_paths[idx]
            img1, img2 = self._split_paired_image(img_path)
            
            night_tensor = self.transform(img1)
            day_tensor = self.transform(img2)
            is_real_label = 1.0
        else:
            # Fake pair from different images
            idx1 = idx
            idx2 = random.randint(0, len(self.image_paths) - 1)
            while idx2 == idx1 and len(self.image_paths) > 1:
                idx2 = random.randint(0, len(self.image_paths) - 1)
            
            img1, _ = self._split_paired_image(self.image_paths[idx1])
            _, img2 = self._split_paired_image(self.image_paths[idx2])
            
            night_tensor = self.transform(img1)
            day_tensor = self.transform(img2)
            is_real_label = 0.0
        
        return {
            'audio': night_tensor,  # First modality (using same key as AVE)
            'image': day_tensor,    # Second modality (using same key as AVE)
            'is_real': torch.tensor(is_real_label)
        }


class Night2DayMIDataModule:
    """Data module matching AVE structure."""
    
    def __init__(
        self,
        train_path: str,
        val_path: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        resolution: int = 64,
        real_fake_ratio: float = 0.5,
        pin_memory: bool = True
    ):
        self.train_path = Path(train_path)
        self.val_path = Path(val_path) if val_path else None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resolution = resolution
        self.real_fake_ratio = real_fake_ratio
        self.pin_memory = pin_memory
        
    def setup(self):
        """Setup datasets."""
        self.train_dataset = Night2DayMIDataset(
            data_path=self.train_path,
            resolution=self.resolution,
            real_fake_ratio=self.real_fake_ratio,
            augment=True,
            debug=True
        )
        
        if self.val_path:
            self.val_dataset = Night2DayMIDataset(
                data_path=self.val_path,
                resolution=self.resolution,
                real_fake_ratio=self.real_fake_ratio,
                augment=False,
                debug=False
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
    
    def val_dataloader(self):
        if not hasattr(self, 'val_dataset'):
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )