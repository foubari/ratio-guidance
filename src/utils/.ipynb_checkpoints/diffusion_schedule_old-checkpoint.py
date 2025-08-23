"""
diffusion_schedule.py
Handles diffusion noise scheduling for different modalities
"""

import torch
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class DiffusionSchedule:
    """Stores diffusion schedule parameters for a modality."""
    alpha_cumprod: torch.Tensor  # ᾱ_t values
    num_timesteps: int
    
    @classmethod
    def from_stable_diffusion(cls, num_timesteps: int = 1000, 
                             beta_start: float = 0.00085, 
                             beta_end: float = 0.012):
        """Create schedule matching Stable Diffusion's default."""
        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        return cls(alpha_cumprod=alpha_cumprod, num_timesteps=num_timesteps)
    
    @classmethod
    def from_audio_ldm(cls, num_timesteps: int = 1000, 
                       beta_start: float = 0.0015 , 
                       beta_end: float = 0.0195):
        """Create schedule for audio (AudioLDM typically uses different betas)."""
        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        return cls(alpha_cumprod=alpha_cumprod, num_timesteps=num_timesteps)
    
    def add_noise(self, x: torch.Tensor, t: torch.Tensor, 
                  noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to samples at timestep t.
        x_t = √(ᾱ_t) * x + √(1 - ᾱ_t) * ε
        
        Args:
            x: Clean samples [B, ...]
            t: Timesteps [B] (integers from 0 to num_timesteps-1)
            noise: Optional pre-generated noise
        
        Returns:
            x_t: Noisy samples
            noise: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x)
        
        # Get alpha_cumprod values for the batch
        alpha_cumprod_t = self.alpha_cumprod[t].to(x.device)
        
        # Reshape for broadcasting
        while len(alpha_cumprod_t.shape) < len(x.shape):
            alpha_cumprod_t = alpha_cumprod_t.unsqueeze(-1)
        
        # Apply forward process
        x_t = torch.sqrt(alpha_cumprod_t) * x + torch.sqrt(1 - alpha_cumprod_t) * noise
        
        return x_t, noise