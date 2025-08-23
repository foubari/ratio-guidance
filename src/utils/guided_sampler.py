# sampling/guided_sampler.py - Fixed version with actual DDPM integration
import sys
from pathlib import Path

# add ../ddpm to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent / "ddpm"))

import torch
import numpy as np
from tqdm import tqdm
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

class GuidedJointSampler:
    """Joint guided sampling from two diffusion models."""
    
    def __init__(
        self,
        diffusion_1,  # Actual diffusion model, not just schedule
        diffusion_2,
        score_computer,
        guidance_scale=1.0,
        num_sampling_steps=50,
        device='cuda'
    ):
        self.diffusion_1 = diffusion_1
        self.diffusion_2 = diffusion_2
        self.score_computer = score_computer
        self.guidance_scale = guidance_scale
        self.device = device
        self.num_sampling_steps = num_sampling_steps
        
    def sample_ddpm(self, num_samples=1):
        """
        Guided sampling for DDPM using the actual diffusion models.
        """
        # Use DDPM's built-in sampling with modification for guidance
        shape = (num_samples, 3, 64, 64)
        
        # Start from noise
        img_1 = torch.randn(shape, device=self.device)
        img_2 = torch.randn(shape, device=self.device)
        
        # Get timesteps from the diffusion model
        timesteps = self.diffusion_1.num_timesteps
        
        # If using DDIM sampling with fewer steps
        if self.num_sampling_steps < timesteps:
            indices = torch.linspace(0, timesteps - 1, self.num_sampling_steps).long()
            timesteps_to_use = indices.flip(0)  # Reverse for sampling
        else:
            timesteps_to_use = torch.arange(timesteps - 1, -1, -1)
        
        for t_idx in tqdm(timesteps_to_use, desc=f"Guided DDPM sampling ({self.score_computer.loss_type})"):
            t = torch.full((num_samples,), t_idx, device=self.device, dtype=torch.long)
            
            # Get unconditional noise predictions from DDPM models
            with torch.no_grad():
                # Get noise predictions
                noise_pred_1 = self.diffusion_1.model(img_1, t)
                noise_pred_2 = self.diffusion_2.model(img_2, t)
                
                # Get guidance gradients
                grad_1, grad_2 = self.score_computer.get_log_ratio_grad(
                    img_1, img_2, t, target='both'
                )
                
                # Apply guidance by modifying the predicted noise
                # The score is -noise_pred in DDPM, so we subtract gradient
                noise_pred_1 = noise_pred_1 - self.guidance_scale * grad_1
                noise_pred_2 = noise_pred_2 - self.guidance_scale * grad_2
                
                # Use DDPM's p_sample to take a reverse step
                # Get alpha values
                alpha = self.diffusion_1.alphas_cumprod[t_idx]
                alpha_prev = self.diffusion_1.alphas_cumprod[t_idx - 1] if t_idx > 0 else torch.tensor(1.0)
                
                # Compute x_{t-1} from x_t using the modified noise prediction
                img_1 = self.p_sample(img_1, noise_pred_1, t_idx, alpha, alpha_prev)
                img_2 = self.p_sample(img_2, noise_pred_2, t_idx, alpha_prev, alpha_prev)
        
        # Clamp to valid range
        img_1 = torch.clamp(img_1, -1, 1)
        img_2 = torch.clamp(img_2, -1, 1)
        
        return img_1, img_2
    
    def p_sample(self, x, noise_pred, t, alpha, alpha_prev):
        """Single DDPM reverse step."""
        # Ensure beta is always a tensor
        if t > 0:
            beta = 1 - alpha / alpha_prev
        else:
            beta = torch.tensor(0.0, device=x.device)
        
        # Predict x_0
        x_start = (x - torch.sqrt(1 - alpha) * noise_pred) / torch.sqrt(alpha)
        x_start = torch.clamp(x_start, -1, 1)
        
        # Compute mean
        mean = torch.sqrt(alpha_prev) * beta / (1 - alpha) * x_start + \
               torch.sqrt(1 - beta) * (1 - alpha_prev) / (1 - alpha) * x
        
        # Add noise if not last step
        if t > 0:
            noise = torch.randn_like(x)
            std = torch.sqrt(beta * (1 - alpha_prev) / (1 - alpha))
            return mean + std * noise
        else:
            return mean