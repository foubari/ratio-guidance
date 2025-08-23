# utils/guided_sampler_native.py - Better step scaling
import torch
from tqdm import tqdm

class NativeGuidedSampler:
    """Guided sampler using DDPM's native p_sample."""
    
    def __init__(self, diffusion_1, diffusion_2, score_computer, guidance_scale=1.0, 
                 sampling_steps=None, device='cuda'):
        self.diffusion_1 = diffusion_1
        self.diffusion_2 = diffusion_2
        self.score_computer = score_computer
        self.guidance_scale = guidance_scale
        self.device = device
        
        # Override sampling timesteps if specified
        if sampling_steps is not None:
            self.diffusion_1.sampling_timesteps = sampling_steps
            self.diffusion_2.sampling_timesteps = sampling_steps
        
        print(f"Using {self.diffusion_1.sampling_timesteps} sampling steps")
        
    def sample(self, batch_size=1):
        """Sample with guidance using native DDPM methods."""
        shape = (batch_size, 3, 64, 64)
        
        # Start from noise
        img_1 = torch.randn(shape, device=self.device)
        img_2 = torch.randn(shape, device=self.device)
        
        # Use the model's sampling_timesteps
        timesteps = self.diffusion_1.sampling_timesteps
        
        # Reverse through timesteps
        for t in tqdm(reversed(range(0, timesteps)), 
                     desc=f'Sampling (g={self.guidance_scale})'):
            
            # Map to actual timestep
            actual_t = t * (self.diffusion_1.num_timesteps // timesteps)
            t_batch = torch.full((batch_size,), actual_t, device=self.device, dtype=torch.long)
            
            if self.guidance_scale > 0 and t > 0:
                with torch.inference_mode(False):
                    with torch.enable_grad():
                        img_1_grad = img_1.detach().clone().requires_grad_(True)
                        img_2_grad = img_2.detach().clone().requires_grad_(True)
                        
                        grad_1, grad_2 = self.score_computer.get_log_ratio_grad(
                            img_1_grad, img_2_grad, t_batch, target='both'
                        )
                
                # Scale by noise level at this timestep
                # Use sqrt(1 - alpha_cumprod) as noise scale
                alpha_cumprod = self.diffusion_1.alphas_cumprod[actual_t]
                noise_scale = torch.sqrt(1 - alpha_cumprod).view(-1, 1, 1, 1)
                
                # Apply guidance scaled by noise level
                img_1 = img_1 + self.guidance_scale * grad_1 * noise_scale * 0.1
                img_2 = img_2 + self.guidance_scale * grad_2 * noise_scale * 0.1
            
            # Use the diffusion model's p_sample directly
            result_1 = self.diffusion_1.p_sample(img_1, actual_t)
            result_2 = self.diffusion_2.p_sample(img_2, actual_t)
            
            # Handle both tuple and tensor returns
            img_1 = result_1[0] if isinstance(result_1, tuple) else result_1
            img_2 = result_2[0] if isinstance(result_2, tuple) else result_2
        
        return img_1, img_2