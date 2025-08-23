"""
utils.py
Utility functions and modules for MI estimation
"""

import torch
import torch.nn as nn


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal timestep embeddings for diffusion models."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, timesteps: torch.Tensor):
        """
        Create sinusoidal embeddings for timesteps.
        
        Args:
            timesteps: [B] tensor of timestep indices
            
        Returns:
            embeddings: [B, dim] tensor of embeddings
        """
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=timesteps.device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


def sample_timesteps(batch_size: int, 
                    min_timestep: int, 
                    max_timestep: int,
                    sampling_strategy: str = "uniform",
                    device: str = "cuda") -> torch.Tensor:
    """
    Sample timesteps according to various strategies.
    
    Args:
        batch_size: Number of timesteps to sample
        min_timestep: Minimum timestep value
        max_timestep: Maximum timestep value
        sampling_strategy: One of "uniform", "importance", "antithetic"
        device: Device to place tensors on
        
    Returns:
        Tensor of sampled timesteps
    """
    if sampling_strategy == "uniform":
        # Uniform sampling
        t = torch.randint(
            min_timestep, 
            max_timestep + 1, 
            (batch_size,), 
            device=device
        )
    elif sampling_strategy == "importance":
        # Sample more from middle timesteps (higher noise variance)
        beta_samples = torch.distributions.Beta(2.0, 2.0).sample((batch_size,))
        t = (beta_samples * (max_timestep - min_timestep) + min_timestep).long().to(device)
    elif sampling_strategy == "antithetic":
        # Antithetic sampling for variance reduction
        half_batch = batch_size // 2
        t1 = torch.randint(min_timestep, max_timestep + 1, (half_batch,), device=device)
        t2 = max_timestep - t1  # Antithetic pairs
        t = torch.cat([t1, t2])
        if batch_size % 2 == 1:
            # Add one more random sample if odd batch size
            t = torch.cat([t, torch.randint(min_timestep, max_timestep + 1, (1,), device=device)])
    else:
        raise ValueError(f"Unknown timestep sampling strategy: {sampling_strategy}")
    
    return t


def get_mi_estimate(model, data_loader, loss_type, device='cuda'):
    """
    Estimate MI using a trained density ratio model.
    
    For DV: MI ≈ E_q[T] - log(E_r[exp(T)])
    For disc: MI ≈ log(4) + E_q[log σ(T)] + E_r[log(1-σ(T))]
    For others: Convert estimated ratio to MI
    """
    import torch.nn.functional as F
    
    model.eval()
    T_real_all = []
    T_fake_all = []
    
    with torch.no_grad():
        for batch in data_loader:
            audio = batch['audio'].to(device)
            image = batch['image'].to(device)
            is_real = batch['is_real'].to(device)
            
            real_mask = is_real > 0.5
            fake_mask = ~real_mask
            
            if real_mask.sum() > 0:
                T_real = model(audio[real_mask], image[real_mask])
                T_real_all.append(T_real)
            
            if fake_mask.sum() > 0:
                T_fake = model(audio[fake_mask], image[fake_mask])
                T_fake_all.append(T_fake)
    
    if not T_real_all or not T_fake_all:
        return None
    
    T_real_all = torch.cat(T_real_all)
    T_fake_all = torch.cat(T_fake_all)
    
    if loss_type == "dv":
        # DV bound
        mi_estimate = T_real_all.mean() - torch.logsumexp(T_fake_all, dim=0) + \
                      torch.log(torch.tensor(float(len(T_fake_all))))
    elif loss_type == "disc":
        # From discriminator using MI = JS * 2 + log(4)
        js_divergence = 0.5 * (F.softplus(-T_real_all).mean() + F.softplus(T_fake_all).mean())
        mi_estimate = torch.log(torch.tensor(4.0)) - 2 * js_divergence
    else:
        # For ratio-based methods, approximate MI from average log ratio
        mi_estimate = T_real_all.mean() - T_fake_all.mean()
    
    return mi_estimate.item()