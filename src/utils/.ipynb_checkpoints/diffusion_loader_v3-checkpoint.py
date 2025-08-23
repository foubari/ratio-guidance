"""
diffusion_loader_v3.py
Updated loader that supports your DDPM models
Place this in src/utils/
"""

import torch
from pathlib import Path
from typing import Optional, Tuple, Any, Literal
from dataclasses import dataclass

# Your existing SD/AudioLDM loader
from .diffusion_loaders import DiffusionModelLoader, DiffusionConfig
from .diffusion_schedule import DiffusionSchedule


@dataclass
class GeneralDiffusionConfig:
    """General config that can specify model types."""
    # Model types
    image_model_type: Literal["stable_diffusion", "ddpm_denoising_diffusion_pytorch"] = "stable_diffusion"
    audio_model_type: Literal["audioldm2", "ddpm", "none"] = "audioldm2"
    
    # Paths for SD/AudioLDM (existing)
    audioldm2_path: Optional[str] = None
    sd_root: Optional[str] = None
    
    # Paths for DDPM models (denoising_diffusion_pytorch)
    image_ddpm_checkpoint: Optional[str] = None  # Path to model-1.pt or specific milestone
    image_ddpm_results_folder: Optional[str] = None  # Path to results folder
    image_ddpm_milestone: Optional[int] = None  # Which milestone to load
    
    # Model parameters
    image_size: int = 64  # For DDPM
    ddpm_timesteps: int = 1000  # Diffusion timesteps for DDPM
    ddpm_dim: int = 64  # DDPM UNet dim
    ddpm_dim_mults: Tuple[int, ...] = (1, 2, 4)  # DDPM UNet multipliers
    
    # Device settings
    device: str = "cuda"
    dtype: str = "float16"  # or "float32" for DDPM
    enable_cpu_offload: bool = True


class DDPMWrapper:
    """
    Wrapper for denoising_diffusion_pytorch models to match our interface.
    """
    def __init__(self, config: GeneralDiffusionConfig):
        self.config = config
        self.device = config.device
        self.model = None
        self.diffusion = None
        self.trainer = None
        self.noise_schedule = None
        
    def load_model(self):
        """Load DDPM from denoising_diffusion_pytorch."""
        from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
        
        # Create model
        model = Unet(
            dim=self.config.ddpm_dim,
            dim_mults=self.config.ddpm_dim_mults,
            flash_attn=True
        )
        
        # Create diffusion
        diffusion = GaussianDiffusion(
            model,
            image_size=self.config.image_size,
            timesteps=self.config.ddpm_timesteps,
            sampling_timesteps=250  # For DDIM sampling
        )
        
        # Create trainer for loading
        if self.config.image_ddpm_results_folder:
            # Load from results folder
            trainer = Trainer(
                diffusion,
                folder='.',  # Dummy folder
                results_folder=self.config.image_ddpm_results_folder,
                train_batch_size=1,  # Dummy values for loading
                train_lr=1e-4,
                train_num_steps=1
            )
            
            # Load specific milestone
            if self.config.image_ddpm_milestone:
                trainer.load(self.config.image_ddpm_milestone)
            else:
                # Load latest
                trainer.load()
                
        elif self.config.image_ddpm_checkpoint:
            # Load directly from checkpoint file
            checkpoint = torch.load(self.config.image_ddpm_checkpoint, map_location=self.device)
            model.load_state_dict(checkpoint['model'])
            if 'ema' in checkpoint:
                # Load EMA weights if available
                ema_model = model  # You might need to handle EMA differently
                ema_model.load_state_dict(checkpoint['ema'])
                model = ema_model
        else:
            raise ValueError("Need either results_folder or checkpoint path for DDPM")
        
        self.model = model.to(self.device)
        self.diffusion = diffusion
        self.trainer = trainer if self.config.image_ddpm_results_folder else None
        
        # Create noise schedule matching DDPM
        # DDPM uses linear beta schedule
        betas = torch.linspace(0.0001, 0.02, self.config.ddpm_timesteps)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        self.noise_schedule = DiffusionSchedule(
            alpha_cumprod=alpha_cumprod,
            num_timesteps=self.config.ddpm_timesteps
        )
        
        print(f"✓ Loaded DDPM from {self.config.image_ddpm_results_folder or self.config.image_ddpm_checkpoint}")
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """For DDPM, encoding is identity (no VAE)."""
        return x
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """For DDPM, decoding is identity (no VAE)."""
        return x
    
    def add_noise(self, x: torch.Tensor, t: torch.Tensor, 
                  noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise using DDPM schedule."""
        return self.noise_schedule.add_noise(x, t, noise)
    
    @property
    def is_latent(self) -> bool:
        """DDPM works in pixel space."""
        return False


class GeneralDiffusionLoader:
    """
    General loader that handles SD, AudioLDM2, and DDPM models.
    """
    
    def __init__(self, config: GeneralDiffusionConfig, base_dir: Optional[Path] = None):
        self.config = config
        self.base_dir = base_dir or Path.cwd()
        
        # For SD/AudioLDM, we'll use existing loader
        self._existing_loader = None
        
        # For DDPM models
        self._image_model = None
        self._audio_model = None
        
        # Track what type of models we have
        self.image_is_latent = (config.image_model_type == "stable_diffusion")
        self.audio_is_latent = (config.audio_model_type == "audioldm2")
    
    def load_models(self) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Load models based on config.
        Returns (image_model_or_vae, audio_model_or_vae)
        """
        image_model = None
        audio_model = None
        
        # Load image model
        if self.config.image_model_type == "stable_diffusion":
            # Use existing loader for SD
            if self._existing_loader is None:
                old_config = DiffusionConfig(
                    sd_root=self.config.sd_root or "../pretrained_models/stable_diffusion",
                    device=self.config.device,
                    dtype=getattr(torch, self.config.dtype),
                    enable_cpu_offload=self.config.enable_cpu_offload
                )
                self._existing_loader = DiffusionModelLoader(old_config, self.base_dir)
            
            image_model = self._existing_loader.load_stable_diffusion(vae_only=True)
            print("   ✓ Loaded SD VAE")
            
        elif self.config.image_model_type == "ddpm_denoising_diffusion_pytorch":
            # Load DDPM for images
            ddpm_wrapper = DDPMWrapper(self.config)
            ddpm_wrapper.load_model()
            image_model = ddpm_wrapper
            print("   ✓ Loaded image DDPM")
        
        # Load audio model
        if self.config.audio_model_type == "audioldm2":
            # Use existing loader for AudioLDM2
            if self._existing_loader is None:
                old_config = DiffusionConfig(
                    audioldm2_path=self.config.audioldm2_path or "../pretrained_models/audioldm2",
                    device=self.config.device,
                    dtype=getattr(torch, self.config.dtype),
                    enable_cpu_offload=self.config.enable_cpu_offload
                )
                self._existing_loader = DiffusionModelLoader(old_config, self.base_dir)
            
            audio_model = self._existing_loader.load_audioldm2(vae_only=True)
            print("   ✓ Loaded AudioLDM2 VAE")
            
        elif self.config.audio_model_type == "none":
            # No audio model (for image-only experiments)
            audio_model = None
            print("   ℹ No audio model (image-only)")
        
        self._image_model = image_model
        self._audio_model = audio_model
        
        return image_model, audio_model
    
    def get_models_for_training(self) -> Tuple[Optional[Any], Optional[Any], bool, bool]:
        """
        Get models/VAEs for training.
        Returns (image_model, audio_model, image_is_latent, audio_is_latent)
        """
        if not hasattr(self, '_image_model') or self._image_model is None:
            self.load_models()
        
        # For latent models (SD/AudioLDM), return VAE
        # For DDPM, return the wrapper (which has encode/decode/add_noise methods)
        return self._image_model, self._audio_model, self.image_is_latent, self.audio_is_latent
    
    def cleanup(self):
        """Free memory."""
        if self._existing_loader:
            self._existing_loader.cleanup()
        
        if self._image_model is not None:
            if hasattr(self._image_model, 'model'):
                del self._image_model.model
            del self._image_model
            self._image_model = None
            
        if self._audio_model is not None:
            del self._audio_model
            self._audio_model = None
            
        torch.cuda.empty_cache()
        print("✓ Memory freed")


# ============================================
# Test in notebook
# ============================================
"""
# Test 1: DDPM + AudioLDM2
config = GeneralDiffusionConfig(
    image_model_type="ddpm_denoising_diffusion_pytorch",
    audio_model_type="audioldm2",
    
    # DDPM settings
    image_ddpm_results_folder="path/to/results/day/1_2_4/tr_stp_70000_stp1000/2024_12_18_14_30",
    image_ddpm_milestone=7,  # Load milestone 7 (70% through training)
    image_size=64,
    ddpm_timesteps=1000,
    
    # AudioLDM settings
    audioldm2_path="../pretrained_models/audioldm2",
    
    device='cuda',
    dtype='float32'  # DDPM typically uses float32
)

loader = GeneralDiffusionLoader(config)
image_model, audio_model, img_latent, aud_latent = loader.get_models_for_training()

print(f"Image model (DDPM): {image_model is not None}, latent: {img_latent}")
print(f"Audio model (AudioLDM2): {audio_model is not None}, latent: {aud_latent}")

# Test 2: SD + No Audio (image-only)
config2 = GeneralDiffusionConfig(
    image_model_type="stable_diffusion",
    audio_model_type="none",
    sd_root="../pretrained_models/stable_diffusion",
    device='cuda'
)

loader2 = GeneralDiffusionLoader(config2)
image_model2, audio_model2, _, _ = loader2.get_models_for_training()
"""