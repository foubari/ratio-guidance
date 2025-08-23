"""
diffusion_loader_v2.py
Generalized diffusion loader that can handle different model types
Place this in src/utils/
This is a minimal change - it wraps your existing loader and adds support for other models
"""

import torch
from pathlib import Path
from typing import Optional, Tuple, Any, Dict, Literal
from dataclasses import dataclass

# Import your existing loader
from .diffusion_loaders import DiffusionModelLoader, DiffusionConfig


@dataclass 
class GeneralDiffusionConfig:
    """General config that can specify model types."""
    # Model types
    image_model_type: Literal["stable_diffusion", "ddpm", "custom"] = "stable_diffusion"
    audio_model_type: Literal["audioldm2", "ddpm", "custom"] = "audioldm2"
    
    # Paths for SD/AudioLDM (existing)
    audioldm2_path: Optional[str] = None
    sd_root: Optional[str] = None
    
    # Paths for DDPM models (new)
    image_ddpm_path: Optional[str] = None
    audio_ddpm_path: Optional[str] = None
    
    # Custom model classes (for DDPM)
    image_model_class: Optional[str] = None  # e.g., "my_models.ImageDDPM"
    audio_model_class: Optional[str] = None  # e.g., "my_models.AudioDDPM"
    
    # Device settings
    device: str = "cuda"
    dtype: str = "float16"  # or "float32" for DDPM
    enable_cpu_offload: bool = True


class GeneralDiffusionLoader:
    """
    General loader that delegates to specific loaders based on model type.
    This is a minimal wrapper around your existing loader.
    """
    
    def __init__(self, config: GeneralDiffusionConfig, base_dir: Optional[Path] = None):
        self.config = config
        self.base_dir = base_dir or Path.cwd()
        
        # For SD/AudioLDM, we'll use your existing loader
        self._existing_loader = None
        
        # For DDPM models, we'll store them here
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
            
        elif self.config.image_model_type == "ddpm":
            # Load DDPM for images
            image_model = self._load_ddpm_model(
                self.config.image_ddpm_path,
                self.config.image_model_class
            )
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
            
        elif self.config.audio_model_type == "ddpm":
            # Load DDPM for audio
            audio_model = self._load_ddpm_model(
                self.config.audio_ddpm_path,
                self.config.audio_model_class
            )
            print("   ✓ Loaded audio DDPM")
        
        self._image_model = image_model
        self._audio_model = audio_model
        
        return image_model, audio_model
    
    def _load_ddpm_model(self, model_path: str, model_class: Optional[str] = None):
        """
        Load a DDPM model from checkpoint.
        For now, this is a placeholder - you'll need to implement based on your DDPM structure.
        """
        if model_path is None:
            raise ValueError("Model path required for DDPM")
        
        checkpoint = torch.load(model_path, map_location=self.config.device)
        
        if model_class:
            # Import and instantiate the model class
            module_name, class_name = model_class.rsplit('.', 1)
            import importlib
            module = importlib.import_module(module_name)
            model_cls = getattr(module, class_name)
            model = model_cls()
            
            # Load state dict
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
        else:
            # Assume checkpoint contains the model directly
            if 'model' in checkpoint:
                model = checkpoint['model']
            else:
                raise ValueError(f"Cannot extract model from checkpoint {model_path}")
        
        return model.to(self.config.device)
    
    def get_vaes(self) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Get VAEs for latent space models.
        For DDPM, returns None since they work in pixel/audio space.
        """
        if not hasattr(self, '_image_model') or self._image_model is None:
            self.load_models()
        
        image_vae = self._image_model if self.image_is_latent else None
        audio_vae = self._audio_model if self.audio_is_latent else None
        
        return image_vae, audio_vae
    
    def cleanup(self):
        """Free memory."""
        if self._existing_loader:
            self._existing_loader.cleanup()
        
        if self._image_model is not None:
            del self._image_model
            self._image_model = None
            
        if self._audio_model is not None:
            del self._audio_model
            self._audio_model = None
            
        torch.cuda.empty_cache()
        print("✓ Memory freed")


# Updated training script snippet - minimal change to use new loader
def load_diffusion_models_v2(args):
    """
    Example of how to use the new loader in your training script.
    This is a drop-in replacement for the VAE loading section.
    """
    from utils.diffusion_loader_v2 import GeneralDiffusionLoader, GeneralDiffusionConfig
    
    # Determine model types from args (you can add args for this)
    use_ddpm_image = getattr(args, 'use_ddpm_image', False)
    use_ddpm_audio = getattr(args, 'use_ddpm_audio', False)
    
    # Create config
    config = GeneralDiffusionConfig(
        # Model types
        image_model_type="ddpm" if use_ddpm_image else "stable_diffusion",
        audio_model_type="ddpm" if use_ddpm_audio else "audioldm2",
        
        # Existing paths (for SD/AudioLDM)
        sd_root=args.sd_root if not use_ddpm_image else None,
        audioldm2_path=args.audioldm2_path if not use_ddpm_audio else None,
        
        # DDPM paths (if using DDPM)
        image_ddpm_path=getattr(args, 'image_ddpm_path', None),
        audio_ddpm_path=getattr(args, 'audio_ddpm_path', None),
        
        # Device settings
        device='cuda' if torch.cuda.is_available() else 'cpu',
        dtype='float32' if (use_ddpm_image or use_ddpm_audio) else 'float16'
    )
    
    # Create loader
    loader = GeneralDiffusionLoader(config)
    
    # Get models/VAEs
    if args.use_latent_space:
        # For latent space, we need VAEs
        image_vae, audio_vae = loader.get_vaes()
        
        if image_vae is None and config.image_model_type == "stable_diffusion":
            raise ValueError("SD VAE required for latent space but not loaded")
        if audio_vae is None and config.audio_model_type == "audioldm2":
            raise ValueError("AudioLDM2 VAE required for latent space but not loaded")
            
        return image_vae, audio_vae, loader
    else:
        # For pixel/audio space with DDPM
        # We don't need the models for training, just return None
        return None, None, loader


# Test function
def test_general_loader():
    """Test the new general loader."""
    
    # Test 1: Load SD + AudioLDM (existing functionality)
    print("Test 1: SD + AudioLDM2")
    config1 = GeneralDiffusionConfig(
        image_model_type="stable_diffusion",
        audio_model_type="audioldm2",
        sd_root="../pretrained_models/stable_diffusion",
        audioldm2_path="../pretrained_models/audioldm2"
    )
    loader1 = GeneralDiffusionLoader(config1)
    image_vae, audio_vae = loader1.get_vaes()
    print(f"  Image VAE (SD): {image_vae is not None}")
    print(f"  Audio VAE (AudioLDM2): {audio_vae is not None}")
    loader1.cleanup()
    
    # Test 2: Mixed - SD + DDPM audio
    print("\nTest 2: SD + DDPM Audio")
    config2 = GeneralDiffusionConfig(
        image_model_type="stable_diffusion", 
        audio_model_type="ddpm",
        sd_root="../pretrained_models/stable_diffusion",
        audio_ddpm_path="../pretrained_models/audio_ddpm/checkpoint.pt"
    )
    loader2 = GeneralDiffusionLoader(config2)
    image_vae, audio_model = loader2.get_vaes()
    print(f"  Image VAE (SD): {image_vae is not None}")
    print(f"  Audio Model (DDPM): {audio_model is None}")  # DDPM returns None for VAE
    loader2.cleanup()
    
    print("\n✓ Tests complete!")


if __name__ == "__main__":
    test_general_loader()