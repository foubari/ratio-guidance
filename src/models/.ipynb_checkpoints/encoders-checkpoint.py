"""
encoders.py
Encoder networks for audio and image modalities
"""

import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    """CNN encoder for mel spectrograms (pixel space)."""
    
    def __init__(self, n_mels: int = 64, hidden_dim: int = 256):
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
        self.projection = nn.Linear(512, hidden_dim)
        
    def forward(self, x):
        """
        Args:
            x: [B, n_mels, time_frames]
        Returns:
            features: [B, hidden_dim]
        """
        h = self.encoder(x).squeeze(-1)
        return self.projection(h)


class ImageEncoder(nn.Module):
    """CNN encoder for images (pixel space)."""
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=4, padding=0),
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
        self.projection = nn.Linear(512, hidden_dim)
        
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            features: [B, hidden_dim]
        """
        h = self.encoder(x).view(x.size(0), -1)
        return self.projection(h)


class LatentAudioEncoder(nn.Module):
    """Linear encoder for audio latents."""
    
    def __init__(self, latent_size: int, hidden_dim: int = 512, feature_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(latent_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, feature_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, latent_size] flattened latent
        Returns:
            features: [B, feature_dim]
        """
        return self.encoder(x)


class LatentImageEncoder(nn.Module):
    """Linear encoder for image latents."""
    
    def __init__(self, latent_size: int, hidden_dim: int = 512, feature_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(latent_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, feature_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, latent_size] flattened latent
        Returns:
            features: [B, feature_dim]
        """
        return self.encoder(x)