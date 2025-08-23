# dataloaders/night2day.py - Corrected for actual directory structure
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import random
from pathlib import Path
from typing import Tuple, Dict, Optional, Union
import glob


class Night2DayMIDataset(Dataset):
    """
    MI training dataset with real and fake night-day pairs.
    Real pairs: Corresponding night-day from separate folders with matching indices
    Fake pairs: Random samples from generated DDPM outputs
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],  # Base path to separated_night_day
        split: str = 'train',  # 'train' or 'val'
        fake_night_dir: str = "../data/samples/night/1_2_4/tr_stp_70000_stp1000/2025_07_21_06_55",
        fake_day_dir: str = "../data/samples/day/1_2_4/tr_stp_70000_stp1000/2025_07_20_22_40",
        resolution: int = 64,
        real_fake_ratio: float = 0.5,
        augment: bool = False,
        debug: bool = False
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.resolution = resolution
        self.real_fake_ratio = real_fake_ratio
        self.augment = augment
        
        # Load real image paths from separate folders with train/val split
        real_night_dir = self.data_path / "night" / split
        real_day_dir = self.data_path / "day" / split
        
        # Get all night images and sort them
        self.real_night_paths = sorted(list(real_night_dir.glob("*.jpg")) + 
                                       list(real_night_dir.glob("*.png")))
        self.real_day_paths = sorted(list(real_day_dir.glob("*.jpg")) + 
                                     list(real_day_dir.glob("*.png")))
        
        # Verify we have matching pairs
        if len(self.real_night_paths) != len(self.real_day_paths):
            print(f"WARNING: Mismatch in real pairs - {len(self.real_night_paths)} night, {len(self.real_day_paths)} day")
            # Use the minimum
            min_len = min(len(self.real_night_paths), len(self.real_day_paths))
            self.real_night_paths = self.real_night_paths[:min_len]
            self.real_day_paths = self.real_day_paths[:min_len]
        
        if len(self.real_night_paths) == 0:
            raise ValueError(f"No real images found in {real_night_dir} and {real_day_dir}")
        
        # Load fake generated samples
        self.fake_night_paths = sorted(glob.glob(str(Path(fake_night_dir) / "*.png")))
        self.fake_day_paths = sorted(glob.glob(str(Path(fake_day_dir) / "*.png")))
        
        if debug:
            print(f"Split: {split}")
            print(f"Real samples: {len(self.real_night_paths)} pairs")
            print(f"Fake night samples: {len(self.fake_night_paths)}")
            print(f"Fake day samples: {len(self.fake_day_paths)}")
            print(f"Resolution: {resolution}x{resolution}")
        
        # Check if we have fake data
        if len(self.fake_night_paths) == 0 or len(self.fake_day_paths) == 0:
            print("WARNING: No fake data found. Will only use real pairs.")
            self.has_fake_data = False
        else:
            self.has_fake_data = True
        
        # Transforms - normalize to [-1, 1] for DDPM (same as training)
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((resolution + 8, resolution + 8)),
                transforms.RandomCrop((resolution, resolution)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [0,1] -> [-1,1]
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((resolution, resolution)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [0,1] -> [-1,1]
            ])
    
    def get_real_pair(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a real corresponding night-day pair from matching indices."""
        idx = idx % len(self.real_night_paths)
        
        # Load corresponding night and day images
        night_img = Image.open(self.real_night_paths[idx]).convert('RGB')
        day_img = Image.open(self.real_day_paths[idx]).convert('RGB')
        
        return {
            'audio': self.transform(night_img),  # Night image [3, H, W]
            'image': self.transform(day_img),    # Day image [3, H, W]
            'is_real': torch.tensor(1.0)
        }
    
    def get_fake_pair(self) -> Dict[str, torch.Tensor]:
        """Get a fake non-corresponding night-day pair from generated samples."""
        # Randomly select from generated samples
        night_idx = random.randint(0, len(self.fake_night_paths) - 1)
        day_idx = random.randint(0, len(self.fake_day_paths) - 1)
        
        night_path = self.fake_night_paths[night_idx]
        day_path = self.fake_day_paths[day_idx]
        
        # Load images
        night_img = Image.open(night_path).convert('RGB')
        day_img = Image.open(day_path).convert('RGB')
        
        return {
            'audio': self.transform(night_img),  # Night image [3, H, W]
            'image': self.transform(day_img),    # Day image [3, H, W]
            'is_real': torch.tensor(0.0)
        }
    
    def __len__(self):
        return len(self.real_night_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns dict matching AVE structure:
            - audio: Night image [3, H, W] (normalized to [-1, 1])
            - image: Day image [3, H, W] (normalized to [-1, 1])
            - is_real: 1.0 for real pairs, 0.0 for fake
        """
        # Decide whether to return real or fake pair
        if self.has_fake_data and random.random() > self.real_fake_ratio:
            return self.get_fake_pair()
        else:
            return self.get_real_pair(idx)


class Night2DayMIDataModule:
    """Data module matching AVE structure."""
    
    def __init__(
        self,
        data_path: str = "../data/separated_night_day",
        fake_night_dir: str = "../data/samples/night/1_2_4/tr_stp_70000_stp1000/2025_07_21_06_55",
        fake_day_dir: str = "../data/samples/day/1_2_4/tr_stp_70000_stp1000/2025_07_20_22_40",
        batch_size: int = 32,
        num_workers: int = 4,
        resolution: int = 64,
        real_fake_ratio: float = 0.5,
        pin_memory: bool = True
    ):
        self.data_path = Path(data_path)
        self.fake_night_dir = fake_night_dir
        self.fake_day_dir = fake_day_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resolution = resolution
        self.real_fake_ratio = real_fake_ratio
        self.pin_memory = pin_memory
        
    def setup(self):
        """Setup datasets."""
        self.train_dataset = Night2DayMIDataset(
            data_path=self.data_path,
            split='train',
            fake_night_dir=self.fake_night_dir,
            fake_day_dir=self.fake_day_dir,
            resolution=self.resolution,
            real_fake_ratio=self.real_fake_ratio,
            augment=True,
            debug=True
        )
        
        # Check if val split exists
        val_night_dir = self.data_path / "night" / "val"
        if val_night_dir.exists():
            self.val_dataset = Night2DayMIDataset(
                data_path=self.data_path,
                split='val',
                fake_night_dir=self.fake_night_dir,
                fake_day_dir=self.fake_day_dir,
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