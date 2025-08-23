# test_ddpm_models.py - Fixed version
import torch
import sys
from pathlib import Path
import torchvision.utils as vutils
import math

sys.path.append('./ddpm')
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

def test_ddpm_sampling(checkpoint_path, num_samples=4, save_prefix='test'):
    """Test DDPM model sampling using its native methods."""
    
    # Model config
    dim = 64
    dim_mults = (1, 2, 4)
    timesteps = 1000
    device = 'cuda'
    
    # Create model
    model = Unet(dim=dim, dim_mults=dim_mults, flash_attn=True)
    diffusion = GaussianDiffusion(
        model,
        image_size=64,
        timesteps=timesteps,
        sampling_timesteps=250
    )
    
    # Load using trainer
    folder = str(Path(checkpoint_path).parent if Path(checkpoint_path).suffix == '.pt' else Path(checkpoint_path))
    trainer = Trainer(
        diffusion,
        '',
        results_folder=folder,
        train_batch_size=16,
        train_lr=1e-4,
        train_num_steps=1,
        calculate_fid=False,
    )
    
    milestone = 10
    trainer.load(milestone=milestone)
    
    # Get the EMA model
    ema_model = trainer.ema.ema_model.to(device).eval()
    
    # Sample using the model's own sample method
    print(f"Sampling {num_samples} images from {checkpoint_path}")
    with torch.no_grad():
        samples = ema_model.sample(batch_size=num_samples)
    
    # Don't normalize - save as is (the model likely outputs in correct range)
    save_path = f'test_outputs/{save_prefix}_samples.png'
    Path('test_outputs').mkdir(exist_ok=True)
    vutils.save_image(samples, save_path, nrow=int(math.sqrt(num_samples)))
    
    # Also test with clamping to be safe
    samples_clamped = torch.clamp(samples, -1, 1)
    vutils.save_image(samples_clamped, f'test_outputs/{save_prefix}_samples_clamped.png', 
                      nrow=int(math.sqrt(num_samples)), normalize=True, value_range=(-1, 1))
    
    print(f"Saved to {save_path}")
    return samples

if __name__ == "__main__":
    # Test day model
    day_samples = test_ddpm_sampling(
        "./ddpm/results/day/1_2_4/tr_stp_70000_stp1000/2025_07_20_22_40",
        num_samples=4,
        save_prefix='day'
    )
    
    # Test night model
    night_samples = test_ddpm_sampling(
        "./ddpm/results/night/1_2_4/tr_stp_70000_stp1000/2025_07_21_06_55",
        num_samples=4,
        save_prefix='night'
    )
    
    print("Test complete - check test_outputs/ folder")