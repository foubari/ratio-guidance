"""
diffusion_schedules.py
Unified diffusion schedule class for both latent and pixel space models
Save this in src/schedules/
"""

import torch
from typing import Optional, Tuple


class DiffusionSchedule:
    """
    Unified diffusion schedule that works for any model type.
    Supports different beta schedules for different modalities.
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        schedule_type: str = "linear",
        device: str = 'cuda'
    ):
        """
        Args:
            num_timesteps: Number of diffusion steps
            beta_start: Starting beta value
            beta_end: Ending beta value
            schedule_type: Type of schedule ('linear', 'cosine')
            device: Device to place tensors on
        """
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type
        
        if schedule_type == "linear":
            betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps)**2
        elif schedule_type == "linear_simple":
            # DDPM style
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        alphas = 1.0 - betas
        self.alpha_cumprod = torch.cumprod(alphas, dim=0).to(device)
    
    def add_noise(self, x: torch.Tensor, t: torch.Tensor, 
                  noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to samples at timestep t.
        
        Args:
            x: Clean samples [B, ...]
            t: Timesteps [B]
            noise: Optional pre-generated noise
            
        Returns:
            x_noisy: Noisy samples
            noise: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x)
        
        alpha_cumprod_t = self.alpha_cumprod[t]
        
        # Reshape for broadcasting
        while len(alpha_cumprod_t.shape) < len(x.shape):
            alpha_cumprod_t = alpha_cumprod_t.unsqueeze(-1)
        
        x_noisy = torch.sqrt(alpha_cumprod_t) * x + torch.sqrt(1 - alpha_cumprod_t) * noise
        
        return x_noisy, noise
    
    @classmethod
    def stable_diffusion_schedule(cls, device='cuda'):
        """Preset for Stable Diffusion."""
        return cls(
            num_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            schedule_type="linear",
            device=device
        )
    
    @classmethod
    def audioldm_schedule(cls, device='cuda'):
        """Preset for AudioLDM."""
        return cls(
            num_timesteps=1000,
            beta_start=0.0015,
            beta_end=0.0195,
            schedule_type="linear",
            device=device
        )
    
    @classmethod
    def ddpm_schedule(cls, device='cuda'):
        """Preset for DDPM."""
        return cls(
            num_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            schedule_type="linear_simple",
            device=device
        )


# Test in notebook:
if __name__ == "__main__":
    # Test different schedules
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # For SD/AudioLDM
    sd_schedule = DiffusionSchedule.stable_diffusion_schedule(device)
    audio_schedule = DiffusionSchedule.audioldm_schedule(device)
    
    # For DDPM
    ddpm_schedule = DiffusionSchedule.ddpm_schedule(device)
    
    # Test adding noise
    x = torch.randn(4, 3, 64, 64).to(device)
    t = torch.randint(0, 1000, (4,)).to(device)
    
    x_noisy, noise = sd_schedule.add_noise(x, t)
    print(f"Input shape: {x.shape}, Noisy shape: {x_noisy.shape}")