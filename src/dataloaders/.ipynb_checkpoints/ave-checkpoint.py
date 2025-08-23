import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
from pathlib import Path
import random
from typing import Optional, Tuple, Dict, List
import glob

class AVEContrastiveDataset(Dataset):
    """
    Dataset that combines real AVE pairs with fake generated pairs.
    Real pairs: corresponding audio-visual from AVE
    Fake pairs: randomly paired generated audio and images
    """
    
    def __init__(
        self,
        ave_mapping_file: str,
        ave_base_dir: str,
        fake_data_dir: str,
        audio_duration: float = 10.0,  # Target duration (will handle 10.24s fake audio)
        audio_sr: int = 16000,
        image_size: Tuple[int, int] = (512, 512),
        audio_transform: str = 'mel_spectrogram',
        n_mels: int = 64,
        hop_length: int = 512,
        real_fake_ratio: float = 0.5,  # Probability of sampling real vs fake
        augment: bool = False,
        debug: bool = False
    ):
        """
        Args:
            ave_mapping_file: Path to AVE mapping JSON
            ave_base_dir: Base directory for AVE data
            fake_data_dir: Directory containing fake audio/images
            audio_duration: Target audio duration in seconds
            audio_sr: Sample rate
            image_size: Target image size
            audio_transform: 'mel_spectrogram' or 'raw'
            n_mels: Number of mel bands
            hop_length: Hop length for spectrogram
            real_fake_ratio: Probability of sampling real pair (0.5 = 50/50)
            augment: Whether to apply augmentation
            debug: Print debug info
        """
        
        # Load real AVE mappings
        with open(ave_mapping_file, 'r') as f:
            ave_mappings = json.load(f)
        
        # Filter for complete samples only
        self.real_samples = [
            m for m in ave_mappings 
            if 'audio_path' in m and 'frame_path' in m
        ]
        
        self.ave_base_dir = Path(ave_base_dir)
        self.fake_data_dir = Path(fake_data_dir)
        
        # Load fake data paths
        self.fake_audio_paths = sorted(glob.glob(str(self.fake_data_dir / "audio" / "*.wav")))
        self.fake_image_paths = sorted(glob.glob(str(self.fake_data_dir / "images" / "*.png")))
        
        if debug:
            print(f"Real samples: {len(self.real_samples)}")
            print(f"Fake audio files: {len(self.fake_audio_paths)}")
            print(f"Fake image files: {len(self.fake_image_paths)}")
        
        # Check if we have fake data
        if len(self.fake_audio_paths) == 0 or len(self.fake_image_paths) == 0:
            print("WARNING: No fake data found. Will only use real pairs.")
            self.has_fake_data = False
        else:
            self.has_fake_data = True
            
            # Check fake audio duration
            if len(self.fake_audio_paths) > 0:
                test_audio, sr = torchaudio.load(self.fake_audio_paths[0])
                fake_duration = test_audio.shape[1] / sr
                if debug:
                    print(f"Fake audio duration: {fake_duration:.2f}s @ {sr}Hz")
        
        # Audio parameters
        self.audio_duration = audio_duration
        self.audio_sr = audio_sr
        self.target_length = int(audio_duration * audio_sr)
        self.audio_transform_type = audio_transform
        self.n_mels = n_mels
        self.hop_length = hop_length
        
        # Image parameters
        self.image_size = image_size
        self.augment = augment
        self.real_fake_ratio = real_fake_ratio
        
        # Create transforms
        self.image_transform = self._create_image_transform()
        
        # Audio transforms
        if audio_transform == 'mel_spectrogram':
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=audio_sr,
                n_fft=1024,
                hop_length=hop_length,
                n_mels=n_mels
            )
            self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
        # Category mapping for real data
        self.categories = sorted(list(set(
            m.get('event_category', 'unknown') for m in self.real_samples
        )))
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        
        # Use length of real data for dataset length
        self.length = len(self.real_samples)
    
    def _create_image_transform(self):
        """Create image preprocessing pipeline."""
        if self.augment:
            transform = transforms.Compose([
                transforms.Resize(self.image_size[0] + 32),
                transforms.RandomCrop(self.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        return transform
    
    def process_audio(self, audio_path: str, is_fake: bool = False) -> torch.Tensor:
        """Load and process audio to ensure consistent dimensions."""
        try:
            waveform, sr = torchaudio.load(audio_path)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if necessary
            if sr != self.audio_sr:
                resampler = torchaudio.transforms.Resample(sr, self.audio_sr)
                waveform = resampler(waveform)
            
            # Pad or truncate to exact target length
            current_length = waveform.shape[1]
            
            if current_length < self.target_length:
                # Pad
                padding = self.target_length - current_length
                pad_left = padding // 2
                pad_right = padding - pad_left
                waveform = torch.nn.functional.pad(waveform, (pad_left, pad_right))
            elif current_length > self.target_length:
                # Truncate (handle 10.24s fake audio -> 10s)
                if self.augment and not is_fake:  # Random crop for real data
                    max_start = current_length - self.target_length
                    start = random.randint(0, max_start)
                else:  # Center crop
                    start = (current_length - self.target_length) // 2
                waveform = waveform[:, start:start + self.target_length]
            
            # Transform to mel spectrogram if specified
            if self.audio_transform_type == 'mel_spectrogram':
                mel_spec = self.mel_transform(waveform)
                mel_spec_db = self.amplitude_to_db(mel_spec)
                mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
                return mel_spec_db.squeeze(0)
            else:
                # Normalize waveform
                waveform = waveform / (torch.abs(waveform).max() + 1e-8)
                return waveform.squeeze(0)
                
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            # Return zeros with correct shape
            if self.audio_transform_type == 'mel_spectrogram':
                n_frames = self.target_length // self.hop_length + 1
                return torch.zeros(self.n_mels, n_frames)
            else:
                return torch.zeros(self.target_length)
    
    def process_image(self, image_path: str) -> torch.Tensor:
        """Load and process image to ensure consistent dimensions."""
        try:
            image = Image.open(image_path).convert('RGB')
            return self.image_transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return torch.zeros(3, *self.image_size)
    
    def get_real_pair(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a real corresponding audio-visual pair."""
        sample = self.real_samples[idx % len(self.real_samples)]
        
        audio_path = str(self.ave_base_dir / sample['audio_path'])
        image_path = str(self.ave_base_dir / sample['frame_path'])
        
        return {
            'audio': self.process_audio(audio_path, is_fake=False),
            'image': self.process_image(image_path),
            'is_real': torch.tensor(1.0),  # Real pair
            'category': torch.tensor(
                self.category_to_idx.get(sample.get('event_category', 'unknown'), 0),
                dtype=torch.long
            ),
            'category_name': sample.get('event_category', 'unknown')
        }
    
    def get_fake_pair(self) -> Dict[str, torch.Tensor]:
        """Get a fake non-corresponding audio-visual pair."""
        # Randomly select fake audio and image (they don't need to match)
        fake_audio_idx = random.randint(0, len(self.fake_audio_paths) - 1)
        fake_image_idx = random.randint(0, len(self.fake_image_paths) - 1)
        
        audio_path = self.fake_audio_paths[fake_audio_idx]
        image_path = self.fake_image_paths[fake_image_idx]
        
        return {
            'audio': self.process_audio(audio_path, is_fake=True),
            'image': self.process_image(image_path),
            'is_real': torch.tensor(0.0),  # Fake pair
            'category': torch.tensor(-1, dtype=torch.long),  # No category for fake
            'category_name': 'fake'
        }
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample (either real or fake based on probability).
        
        Returns:
            Dictionary with:
                - audio: Audio tensor [n_mels, time] or [time]
                - image: Image tensor [3, H, W]
                - is_real: 1.0 for real pairs, 0.0 for fake pairs
                - category: Category index (-1 for fake)
                - category_name: Category string ('fake' for fake pairs)
        """
        # Decide whether to return real or fake pair
        if self.has_fake_data and random.random() > self.real_fake_ratio:
            return self.get_fake_pair()
        else:
            return self.get_real_pair(idx)


class AVEContrastiveDataModule:
    """Data module for contrastive learning with real and fake pairs."""
    
    def __init__(
        self,
        ave_data_dir: str,
        fake_data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        audio_duration: float = 10.0,
        audio_sr: int = 16000,
        image_size: Tuple[int, int] = (512, 512),
        audio_transform: str = 'mel_spectrogram',
        real_fake_ratio: float = 0.5,
        pin_memory: bool = True
    ):
        self.ave_data_dir = Path(ave_data_dir)
        self.fake_data_dir = Path(fake_data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        self.dataset_params = {
            'audio_duration': audio_duration,
            'audio_sr': audio_sr,
            'image_size': image_size,
            'audio_transform': audio_transform,
            'real_fake_ratio': real_fake_ratio
        }
    
    def setup(self):
        """Setup datasets."""
        # Training with augmentation and 50/50 real/fake
        self.train_dataset = AVEContrastiveDataset(
            ave_mapping_file=str(self.ave_data_dir / 'mappings' / 'train_mappings.json'),
            ave_base_dir=str(self.ave_data_dir),
            fake_data_dir=str(self.fake_data_dir),
            augment=True,
            debug=True,
            **self.dataset_params
        )
        
        # Validation without augmentation
        self.val_dataset = AVEContrastiveDataset(
            ave_mapping_file=str(self.ave_data_dir / 'mappings' / 'val_mappings.json'),
            ave_base_dir=str(self.ave_data_dir),
            fake_data_dir=str(self.fake_data_dir),
            augment=False,
            **self.dataset_params
        )
        
        # Test without augmentation
        self.test_dataset = AVEContrastiveDataset(
            ave_mapping_file=str(self.ave_data_dir / 'mappings' / 'test_mappings.json'),
            ave_base_dir=str(self.ave_data_dir),
            fake_data_dir=str(self.fake_data_dir),
            augment=False,
            **self.dataset_params
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
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )