"""
image_sound_guide.py
MI estimator models with optional timestep conditioning
Place this in src/models/
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import sys
from pathlib import Path

# Add parent (src/) to path to import from utils
sys.path.append(str(Path(__file__).parent.parent))

# Import from same models folder
from .encoders import (
    AudioEncoder, ImageEncoder, 
    LatentAudioEncoder, LatentImageEncoder
)

# Import from utils folder
from utils.utils import SinusoidalPositionEmbeddings


class MIEstimator(nn.Module):
    """
    Basic MI estimator without timestep conditioning.
    """
    
    def __init__(self, feature_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, audio_features: torch.Tensor, image_features: torch.Tensor):
        """
        Args:
            audio_features: [B, feature_dim]
            image_features: [B, feature_dim]
        Returns:
            scores: [B]
        """
        combined = torch.cat([audio_features, image_features], dim=-1)
        return self.network(combined).squeeze(-1)


class TimestepMIEstimator(nn.Module):
    """
    MI estimator with timestep conditioning for diffusion models.
    """
    
    def __init__(self, feature_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        
        # Timestep embedding
        self.time_embed_dim = 128
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(self.time_embed_dim),
            nn.Linear(self.time_embed_dim, self.time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim * 2, self.time_embed_dim),
        )
        
        # Main network with timestep conditioning
        self.network = nn.Sequential(
            nn.Linear(feature_dim * 2 + self.time_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, audio_features: torch.Tensor, 
                image_features: torch.Tensor, 
                t: torch.Tensor):
        """
        Args:
            audio_features: [B, feature_dim]
            image_features: [B, feature_dim]
            t: [B] timesteps
        Returns:
            scores: [B]
        """
        # Embed timesteps
        t_emb = self.time_embed(t.float())
        
        # Concatenate all inputs
        combined = torch.cat([audio_features, image_features, t_emb], dim=-1)
        
        return self.network(combined).squeeze(-1)


class MIModel(nn.Module):
    """
    Complete MI model for clean data (no timestep).
    """
    
    def __init__(self, n_mels: int = 64, feature_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.audio_encoder = AudioEncoder(n_mels=n_mels, hidden_dim=feature_dim)
        self.image_encoder = ImageEncoder(hidden_dim=feature_dim)
        self.mi_estimator = MIEstimator(feature_dim=feature_dim, hidden_dim=hidden_dim)
        
    def forward(self, audio: torch.Tensor, image: torch.Tensor):
        """
        Args:
            audio: [B, n_mels, time]
            image: [B, 3, H, W]
        Returns:
            scores: [B]
        """
        audio_features = self.audio_encoder(audio)
        image_features = self.image_encoder(image)
        return self.mi_estimator(audio_features, image_features)
    
    def get_features(self, audio: torch.Tensor, image: torch.Tensor):
        """Get intermediate features for analysis."""
        audio_features = self.audio_encoder(audio)
        image_features = self.image_encoder(image)
        return audio_features, image_features


class FlexibleMIModel(nn.Module):
    """
    MI model that can work in either latent or pixel/audio space,
    with optional timestep conditioning.
    """
    
    def __init__(self, 
                 # For pixel/audio space
                 n_mels: int = 64,
                 image_size: Tuple[int, int] = (512, 512),
                 # For latent space
                 audio_latent_shape: Optional[Tuple[int, ...]] = None,
                 image_latent_shape: Optional[Tuple[int, ...]] = None,
                 # Common parameters
                 hidden_dim: int = 512,
                 feature_dim: int = 256,
                 use_latent_space: bool = False,
                 use_timestep: bool = False):
        super().__init__()
        
        self.use_latent_space = use_latent_space
        self.use_timestep = use_timestep
        
        # Setup encoders based on mode
        if use_latent_space:
            assert audio_latent_shape is not None and image_latent_shape is not None
            audio_flat_size = np.prod(audio_latent_shape)
            image_flat_size = np.prod(image_latent_shape)
            
            self.audio_encoder = LatentAudioEncoder(
                audio_flat_size, hidden_dim, feature_dim
            )
            self.image_encoder = LatentImageEncoder(
                image_flat_size, hidden_dim, feature_dim
            )
        else:
            self.audio_encoder = AudioEncoder(n_mels=n_mels, hidden_dim=feature_dim)
            self.image_encoder = ImageEncoder(hidden_dim=feature_dim)
        
        # Setup MI estimator based on timestep usage
        if use_timestep:
            self.mi_estimator = TimestepMIEstimator(
                feature_dim=feature_dim, 
                hidden_dim=hidden_dim
            )
        else:
            self.mi_estimator = MIEstimator(
                feature_dim=feature_dim, 
                hidden_dim=hidden_dim
            )
    
    def forward(self, audio: torch.Tensor, image: torch.Tensor, 
                t: Optional[torch.Tensor] = None):
        """
        Args:
            audio: [B, ...] audio in latent or pixel space
            image: [B, ...] image in latent or pixel space
            t: [B] timesteps (optional, required if use_timestep=True)
        Returns:
            scores: [B]
        """
        if self.use_latent_space:
            # Flatten latents
            audio_flat = audio.view(audio.size(0), -1)
            image_flat = image.view(image.size(0), -1)
            audio_features = self.audio_encoder(audio_flat)
            image_features = self.image_encoder(image_flat)
        else:
            # Use ConvNet encoders
            audio_features = self.audio_encoder(audio)
            image_features = self.image_encoder(image)
        
        # Forward through MI estimator
        if self.use_timestep:
            assert t is not None, "Timestep required when use_timestep=True"
            return self.mi_estimator(audio_features, image_features, t)
        else:
            return self.mi_estimator(audio_features, image_features)