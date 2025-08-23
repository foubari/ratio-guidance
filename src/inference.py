# inference.py - Modified to save paired visualizations
import argparse
import torch
from pathlib import Path
from utils.score_computation import ScoreComputer
from utils.diffusion_model_loader import UnifiedDiffusionLoader
from utils.guided_sampler_native import NativeGuidedSampler
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import math

# DDPM checkpoint paths
DDPM_CHECKPOINTS = {
    'day': "./ddpm/results/day/1_2_4/tr_stp_70000_stp1000/2025_07_20_22_40",
    'night': "./ddpm/results/night/1_2_4/tr_stp_70000_stp1000/2025_07_21_06_55"
}

def save_paired_visualization(night_imgs, day_imgs, save_path, num_per_fig=5):
    """Save paired visualizations with day on top, night on bottom."""
    num_samples = night_imgs.shape[0]
    num_figures = (num_samples + num_per_fig - 1) // num_per_fig
    
    # Convert from [-1, 1] to [0, 1] for visualization
    night_imgs_vis = (night_imgs + 1) / 2
    day_imgs_vis = (day_imgs + 1) / 2
    
    for fig_idx in range(num_figures):
        start_idx = fig_idx * num_per_fig
        end_idx = min(start_idx + num_per_fig, num_samples)
        num_in_this_fig = end_idx - start_idx
        
        # Create figure with 2 rows x num_per_fig columns
        fig, axes = plt.subplots(2, num_in_this_fig, figsize=(3*num_in_this_fig, 6))
        
        # Handle case where we have only 1 image
        if num_in_this_fig == 1:
            axes = axes.reshape(2, 1)
        
        for i in range(num_in_this_fig):
            idx = start_idx + i
            
            # Day on top
            day_img = day_imgs_vis[idx].permute(1, 2, 0).cpu().numpy()
            axes[0, i].imshow(day_img)
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_ylabel('Day', fontsize=12, fontweight='bold')
            axes[0, i].set_title(f'Pair {idx}', fontsize=10)
            
            # Night on bottom
            night_img = night_imgs_vis[idx].permute(1, 2, 0).cpu().numpy()
            axes[1, i].imshow(night_img)
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_ylabel('Night', fontsize=12, fontweight='bold')
        
        plt.suptitle(f'Generated Pairs (Batch {fig_idx+1}/{num_figures})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        fig_save_path = save_path / f'paired_batch_{fig_idx+1}.png'
        fig.savefig(fig_save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved paired visualization to {fig_save_path}")

def guided_generation(
    loss_type='disc',
    model_type='sd_audioldm',
    num_samples=4,
    guidance_scale=1.0,
    num_sampling_steps=50,
    save_dir='outputs'
):
    """Generate guided samples using trained MI model."""
    device = 'cuda'
    save_dir = Path(save_dir) / f'{model_type}_{loss_type}_g{guidance_scale}_stp{num_sampling_steps}'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load score computer
    score_computer = ScoreComputer(
        loss_type=loss_type,
        model_type=model_type,
        device=device
    )
    
    # Load diffusion models
    loader = UnifiedDiffusionLoader(model_type, device)
    
    if model_type == 'ddpm_night2day':
        # Load DDPM models
        models = loader.load_models(
            day_checkpoint_path=DDPM_CHECKPOINTS['day'],
            night_checkpoint_path=DDPM_CHECKPOINTS['night'],
            num_sampling_steps=num_sampling_steps,
        )
        
        # Use native guided sampler for DDPM
        sampler = NativeGuidedSampler(
            models['diffusion_1'],  # night
            models['diffusion_2'],  # day
            score_computer,
            guidance_scale=guidance_scale,
            device=device
        )
        
        # Generate samples using native sampling
        night_imgs, day_imgs = sampler.sample(batch_size=num_samples)
        
        # The images are in [-1, 1] range which is correct for the night2day dataset
        print(f"Output range: [{night_imgs.min():.2f}, {night_imgs.max():.2f}]")
        
        # Save individual images (keeping original functionality)
        for i in range(num_samples):
            vutils.save_image(
                night_imgs[i:i+1], 
                save_dir / f'night_{i}.png',
                normalize=True,
                value_range=(-1, 1)
            )
            vutils.save_image(
                day_imgs[i:i+1], 
                save_dir / f'day_{i}.png',
                normalize=True,
                value_range=(-1, 1)
            )
        
        # Save grids
        vutils.save_image(
            night_imgs, 
            save_dir / 'night_grid.png', 
            nrow=int(math.sqrt(num_samples)),
            normalize=True,
            value_range=(-1, 1)
        )
        vutils.save_image(
            day_imgs, 
            save_dir / 'day_grid.png', 
            nrow=int(math.sqrt(num_samples)),
            normalize=True,
            value_range=(-1, 1)
        )
        
        # Save paired visualizations (5 pairs per figure)
        save_paired_visualization(night_imgs, day_imgs, save_dir, num_per_fig=5)
        
    else:  # sd_audioldm
        # SD/AudioLDM needs different handling - not implemented yet
        raise NotImplementedError("SD/AudioLDM guided sampling not yet implemented with native methods")
    
    print(f"Generated {num_samples} samples saved to {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type', choices=['sd_audioldm', 'ddpm_night2day'])
    parser.add_argument('--loss_type', choices=['disc', 'dv', 'ulsif', 'rulsif', 'kliep'], default='disc')
    parser.add_argument('--num_samples', type=int, default=10)  # Changed default to 10 for better visualization
    parser.add_argument('--guidance_scale', type=float, default=1.0)
    parser.add_argument('--sampling_steps', type=int, default=50)
    
    args = parser.parse_args()
    guided_generation(
        loss_type=args.loss_type,
        model_type=args.model_type,
        num_samples=args.num_samples,
        guidance_scale=args.guidance_scale,
        num_sampling_steps=args.sampling_steps
    )