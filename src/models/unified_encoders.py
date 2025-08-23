"""
unified_encoders.py
Unified encoder classes for both latent and pixel space
Save this in src/models/
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class LatentEncoder(nn.Module):
    """Linear encoder for latent representations (SD/AudioLDM)."""
    
    def __init__(self, latent_shape: Tuple[int, ...], hidden_dim: int = 512, feature_dim: int = 256):
        super().__init__()
        latent_size = np.prod(latent_shape)
        
        self.encoder = nn.Sequential(
            nn.Linear(latent_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, feature_dim)
        )
    
    def forward(self, x):
        # Flatten latent: [B, C, H, W] -> [B, C*H*W]
        x_flat = x.view(x.size(0), -1)
        return self.encoder(x_flat)


class PixelEncoder(nn.Module):
    """CNN encoder for pixel space images."""
    
    def __init__(self, image_size: int, in_channels: int = 3, feature_dim: int = 256):
        super().__init__()
        
        if image_size == 64:
            # For 64x64 images (DDPM)
            self.encoder = nn.Sequential(
                # 64x64 -> 32x32
                nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                # 32x32 -> 16x16
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                # 16x16 -> 8x8
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                # 8x8 -> 4x4
                nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
        elif image_size == 512:
            # For 512x512 images (if needed for pixel space SD)
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=4, stride=4, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=4, stride=4, padding=0),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=4, stride=4, padding=0),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 512, kernel_size=4, stride=4, padding=0),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
        else:
            raise ValueError(f"Unsupported image size: {image_size}")
        
        self.projection = nn.Linear(512, feature_dim)
        
    def forward(self, x):
        h = self.encoder(x).view(x.size(0), -1)
        return self.projection(h)


class MelSpecEncoder(nn.Module):
    """CNN encoder for mel spectrograms (audio in pixel space)."""
    
    def __init__(self, n_mels: int = 64, feature_dim: int = 256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(n_mels, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.projection = nn.Linear(512, feature_dim)
        
    def forward(self, x):
        # x: [B, n_mels, time_frames]
        h = self.encoder(x).squeeze(-1)
        return self.projection(h)


class UnifiedEncoder(nn.Module):
    """
    Unified encoder that automatically selects the right encoder type.
    """
    
    def __init__(
        self,
        input_type: str,  # 'latent', 'pixel', 'mel_spec'
        input_shape: Optional[Tuple[int, ...]] = None,
        feature_dim: int = 256,
        hidden_dim: int = 512
    ):
        super().__init__()
        self.input_type = input_type
        
        if input_type == 'latent':
            if input_shape is None:
                raise ValueError("latent_shape required for latent encoder")
            self.encoder = LatentEncoder(input_shape, hidden_dim, feature_dim)
            
        elif input_type == 'pixel':
            if input_shape is None or len(input_shape) < 2:
                raise ValueError("image shape required for pixel encoder")
            # Assume shape is (C, H, W) or (H, W)
            if len(input_shape) == 2:
                h, w = input_shape
                c = 3
            else:
                c, h, w = input_shape
            if h != w:
                raise ValueError("Only square images supported")
            self.encoder = PixelEncoder(h, c, feature_dim)
            
        elif input_type == 'mel_spec':
            if input_shape is None:
                n_mels = 64  # default
            else:
                n_mels = input_shape[0] if len(input_shape) > 0 else 64
            self.encoder = MelSpecEncoder(n_mels, feature_dim)
            
        else:
            raise ValueError(f"Unknown input type: {input_type}")
    
    def forward(self, x):
        return self.encoder(x)
    
    @classmethod
    def create_sd_image_encoder(cls, feature_dim=256):
        """Create encoder for SD image latents."""
        return cls(
            input_type='latent',
            input_shape=(4, 64, 64),
            feature_dim=feature_dim
        )
    
    @classmethod
    def create_audioldm_encoder(cls, feature_dim=256):
        """Create encoder for AudioLDM latents."""
        return cls(
            input_type='latent',
            input_shape=(8, 16, 78),
            feature_dim=feature_dim
        )
    
    @classmethod
    def create_ddpm_image_encoder(cls, feature_dim=256):
        """Create encoder for 64x64 DDPM images."""
        return cls(
            input_type='pixel',
            input_shape=(3, 64, 64),
            feature_dim=feature_dim
        )
    
    @classmethod
    def create_mel_spec_encoder(cls, n_mels=64, feature_dim=256):
        """Create encoder for mel spectrograms."""
        return cls(
            input_type='mel_spec',
            input_shape=(n_mels,),
            feature_dim=feature_dim
        )


# Test in notebook:
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test latent encoders
    sd_encoder = UnifiedEncoder.create_sd_image_encoder()
    audio_encoder = UnifiedEncoder.create_audioldm_encoder()
    
    # Test pixel encoders
    ddpm_encoder = UnifiedEncoder.create_ddpm_image_encoder()
    
    # Test with dummy data
    sd_latent = torch.randn(4, 4, 64, 64).to(device)
    audio_latent = torch.randn(4, 8, 16, 78).to(device)
    ddpm_image = torch.randn(4, 3, 64, 64).to(device)
    
    sd_encoder = sd_encoder.to(device)
    audio_encoder = audio_encoder.to(device)
    ddpm_encoder = ddpm_encoder.to(device)
    
    # Forward passes
    sd_features = sd_encoder(sd_latent)
    audio_features = audio_encoder(audio_latent)
    ddpm_features = ddpm_encoder(ddpm_image)
    
    print(f"SD features: {sd_features.shape}")
    print(f"Audio features: {audio_features.shape}")
    print(f"DDPM features: {ddpm_features.shape}")