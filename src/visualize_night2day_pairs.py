# visualize_night2day_pairs.py
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import random
sys.path.append('.')

from dataloaders.night2day import Night2DayMIDataset

def denormalize(tensor):
    """Convert from [-1, 1] to [0, 1] for visualization."""
    return (tensor + 1) / 2

def plot_night2day_pairs(dataset, num_samples=5):
    """Plot real and fake night-day pairs."""
    
    # Collect real and fake samples
    real_samples = []
    fake_samples = []
    
    # Get shuffled indices for real pairs while maintaining correspondence
    total_real_pairs = len(dataset.real_night_paths)
    shuffled_indices = list(range(total_real_pairs))
    random.shuffle(shuffled_indices)
    
    # Get real samples using shuffled indices
    dataset.real_fake_ratio = 1.0  # Force real
    for i in range(min(num_samples, total_real_pairs)):
        idx = shuffled_indices[i]
        sample = dataset.get_real_pair(idx)
        real_samples.append(sample)
    
    if dataset.has_fake_data:
        for i in range(num_samples):
            sample = dataset.get_fake_pair()
            fake_samples.append(sample)
    
    # Create figure
    num_rows = 2 if dataset.has_fake_data else 1
    fig, axes = plt.subplots(num_rows * 2, num_samples, figsize=(15, 6 * num_rows))
    
    if num_rows == 1:
        axes = axes.reshape(1, 2, num_samples)
    else:
        axes = axes.reshape(num_rows, 2, num_samples)
    
    # Plot real samples
    for i, sample in enumerate(real_samples):
        night = denormalize(sample['audio']).permute(1, 2, 0).cpu().numpy()
        day = denormalize(sample['image']).permute(1, 2, 0).cpu().numpy()
        
        # Day on top
        axes[0, 0, i].imshow(day)
        axes[0, 0, i].axis('off')
        if i == 0:
            axes[0, 0, i].set_ylabel('Day', fontsize=12)
        
        # Night on bottom
        axes[0, 1, i].imshow(night)
        axes[0, 1, i].axis('off')
        if i == 0:
            axes[0, 1, i].set_ylabel('Night', fontsize=12)
    
    # Add title for real samples
    fig.text(0.5, 0.95 if num_rows == 2 else 0.9, 'REAL PAIRS (Corresponding)', 
             ha='center', fontsize=14, fontweight='bold', color='green')
    
    # Plot fake samples if available
    if dataset.has_fake_data:
        for i, sample in enumerate(fake_samples):
            night = denormalize(sample['audio']).permute(1, 2, 0).cpu().numpy()
            day = denormalize(sample['image']).permute(1, 2, 0).cpu().numpy()
            
            # Day on top
            axes[1, 0, i].imshow(day)
            axes[1, 0, i].axis('off')
            if i == 0:
                axes[1, 0, i].set_ylabel('Day', fontsize=12)
            
            # Night on bottom
            axes[1, 1, i].imshow(night)
            axes[1, 1, i].axis('off')
            if i == 0:
                axes[1, 1, i].set_ylabel('Night', fontsize=12)
        
        # Add title for fake samples
        fig.text(0.5, 0.48, 'FAKE PAIRS (Random DDPM Samples)', 
                ha='center', fontsize=14, fontweight='bold', color='red')
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    return fig

def main():
    # Set seed for reproducible shuffling (optional - remove for different shuffles each run)
    # random.seed(42)
    
    # Setup dataset with corrected paths
    dataset = Night2DayMIDataset(
        data_path="../data/separated_night_day",  # Base path
        split='train',  # Specify split
        fake_night_dir="../data/samples/night/1_2_4/tr_stp_70000_stp1000/2025_07_21_06_55",
        fake_day_dir="../data/samples/day/1_2_4/tr_stp_70000_stp1000/2025_07_20_22_40",
        resolution=64,
        real_fake_ratio=0.5,
        augment=False,
        debug=True
    )
    
    # Plot and save
    fig = plot_night2day_pairs(dataset, num_samples=5)
    output_path = "night2day_pairs_visualization.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    plt.show()

if __name__ == "__main__":
    main()