# utils/diffusion_model_loader.py - Fixed version

import sys
from pathlib import Path

# add ../ddpm to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent / "ddpm"))

import torch
from typing import Tuple
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

class DDPMLoader:
    """Loader for DDPM models."""
    
    @staticmethod
    def load_ddpm_models(
        day_checkpoint_path: str,
        night_checkpoint_path: str,
        device: str = 'cuda',
        num_sampling_steps:int=1000
    ) -> Tuple[GaussianDiffusion, GaussianDiffusion]:
        """
        Load DDPM models for night and day.
        
        Args:
            day_checkpoint_path: Path to checkpoint (can be directory or .pt file)
            night_checkpoint_path: Path to checkpoint (can be directory or .pt file)
            device: Device to load models on
            
        Returns:
            Tuple of (day_diffusion, night_diffusion)
        """
        # Model architecture (must match training)
        dim = 64
        dim_mults = (1, 2, 4)
        sampling_timesteps = num_sampling_steps
        
        # Parse paths and milestones
        def parse_checkpoint_path(path):
            path = Path(path)
            if path.suffix == '.pt':
                # Direct path to .pt file
                folder = str(path.parent)
                milestone = int(path.stem.split('-')[1])
            elif path.name.startswith('model-'):
                # Path like "results/day/.../model-70"
                folder = str(path.parent)
                milestone = int(path.name.split('-')[1])
            else:
                # Path is just the folder, need to find latest checkpoint
                folder = str(path)
                # Find latest model-*.pt file
                model_files = list(path.glob('model-*.pt'))
                if not model_files:
                    raise FileNotFoundError(f"No model-*.pt files found in {path}")
                latest = max(model_files, key=lambda p: int(p.stem.split('-')[1]))
                milestone = int(latest.stem.split('-')[1])
            return folder, milestone
        
        day_folder, day_milestone = parse_checkpoint_path(day_checkpoint_path)
        night_folder, night_milestone = parse_checkpoint_path(night_checkpoint_path)
        
        print(f"Loading day model from {day_folder}/model-{day_milestone}.pt")
        print(f"Loading night model from {night_folder}/model-{night_milestone}.pt")
        
        # Create models
        day_model = Unet(dim=dim, dim_mults=dim_mults, flash_attn=True)
        night_model = Unet(dim=dim, dim_mults=dim_mults, flash_attn=True)
        
        # Create diffusion wrappers
        day_diffusion = GaussianDiffusion(
            day_model,
            image_size=64,
            timesteps=1000,
            sampling_timesteps=sampling_timesteps
        )
        
        night_diffusion = GaussianDiffusion(
            night_model,
            image_size=64,
            timesteps=1000,
            sampling_timesteps=sampling_timesteps
        )
        
        # Load using trainers
        day_trainer = Trainer(
            day_diffusion,
            '',  # Dummy folder
            results_folder=day_folder,
            train_batch_size=16,
            train_lr=1e-4,
            train_num_steps=1000,
            calculate_fid=False,
        )
        day_trainer.load(milestone=day_milestone)
        
        night_trainer = Trainer(
            night_diffusion,
            '',  # Dummy folder
            results_folder=night_folder,
            train_batch_size=16,
            train_lr=1e-4,
            train_num_steps=1000,
            calculate_fid=False,
        )
        night_trainer.load(milestone=night_milestone)
        
        # Get the EMA models and move to device
        day_diffusion = day_trainer.ema.ema_model.to(device).eval()
        night_diffusion = night_trainer.ema.ema_model.to(device).eval()
        
        print(f"Loaded DDPM models from milestones {day_milestone} and {night_milestone}")
        return day_diffusion, night_diffusion


class UnifiedDiffusionLoader:
    """Unified loader for all diffusion models."""
    
    def __init__(self, model_type: str, device: str = 'cuda'):
        """
        Args:
            model_type: 'sd_audioldm' or 'ddpm_night2day'
        """
        self.model_type = model_type
        self.device = device
        
    def load_models(self, **kwargs):
        """
        Load diffusion models based on model type.
        
        For DDPM: kwargs should contain day_checkpoint_path and night_checkpoint_path
        For SD/AudioLDM: uses DiffusionModelLoader
        """
        if self.model_type == 'ddpm_night2day':
            return self._load_ddpm(**kwargs)
        else:  # sd_audioldm
            return self._load_sd_audioldm(**kwargs)
    
    def _load_ddpm(self, day_checkpoint_path: str, night_checkpoint_path: str, num_sampling_steps:int,  **kwargs):
        """Load DDPM models."""
        loader = DDPMLoader()
        day_diffusion, night_diffusion = loader.load_ddpm_models(
            day_checkpoint_path,
            night_checkpoint_path,
            self.device,
            num_sampling_steps,
        )
        return {
            'diffusion_1': night_diffusion,  # night
            'diffusion_2': day_diffusion,    # day
            'vae_1': None,
            'vae_2': None
        }
    
    def _load_sd_audioldm(self, **kwargs):
        """Load SD and AudioLDM models with full pipelines."""
        from utils.diffusion_loaders import DiffusionModelLoader
        
        loader = DiffusionModelLoader()
        
        # Load full pipelines, not just VAEs
        audio_pipe = loader.load_audioldm2(vae_only=False)
        sd_pipe = loader.load_stable_diffusion(vae_only=False)
        
        return {
            'diffusion_1': audio_pipe,  # AudioLDM pipeline
            'diffusion_2': sd_pipe,      # SD pipeline
            'vae_1': audio_pipe.vae,
            'vae_2': sd_pipe.vae
        }