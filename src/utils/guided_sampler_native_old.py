# utils/guided_sampler_native.py - Fixed to handle inference mode
import torch
from tqdm import tqdm

class NativeGuidedSampler:
    """Guided sampler using DDPM's native p_sample."""
    
    def __init__(self, diffusion_1, diffusion_2, score_computer, guidance_scale=1.0, device='cuda'):
        self.diffusion_1 = diffusion_1
        self.diffusion_2 = diffusion_2
        self.score_computer = score_computer
        self.guidance_scale = guidance_scale
        self.device = device
        
    def sample(self, batch_size=1):
        """Sample with guidance using native DDPM methods."""
        shape = (batch_size, 3, 64, 64)
        
        # Start from noise
        img_1 = torch.randn(shape, device=self.device)
        img_2 = torch.randn(shape, device=self.device)
        
        # Reverse through timesteps
        for t in tqdm(reversed(range(0, self.diffusion_1.num_timesteps)), 
                     desc=f'Sampling (g={self.guidance_scale})'):
            
            # For guidance gradient computation, we need a tensor
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            
            if self.guidance_scale > 0 and t > 0:
                # Temporarily exit inference mode to compute gradients
                with torch.inference_mode(False):
                    with torch.enable_grad():
                        img_1_grad = img_1.detach().clone().requires_grad_(True)
                        img_2_grad = img_2.detach().clone().requires_grad_(True)
                        
                        grad_1, grad_2 = self.score_computer.get_log_ratio_grad(
                            img_1_grad, img_2_grad, t_batch, target='both'
                        )
                
                # Apply guidance as a correction to the images before denoising
                img_1 = img_1 + self.guidance_scale * grad_1# * 0.01
                img_2 = img_2 + self.guidance_scale * grad_2# * 0.01
            
            # Use the diffusion model's p_sample directly
            # p_sample returns a tuple (x_prev, x_0_pred), we only need x_prev
            result_1 = self.diffusion_1.p_sample(img_1, t)
            result_2 = self.diffusion_2.p_sample(img_2, t)
            
            # Handle both tuple and tensor returns
            img_1 = result_1[0] if isinstance(result_1, tuple) else result_1
            img_2 = result_2[0] if isinstance(result_2, tuple) else result_2
        
        return img_1, img_2