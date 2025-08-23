"""
diffusion_loaders.py
Handles loading of diffusion models (SD and AudioLDM2) for VAE encoding
Place this in src/utils/
"""

import os
import torch
from pathlib import Path
from typing import Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class DiffusionConfig:
    """Configuration for diffusion model paths."""
    # AudioLDM2 paths
    audioldm2_path: str = "pretrained_models/audioldm2"  # Changed from ../pretrained_models
    
    # Stable Diffusion paths
    sd_root: str = "pretrained_models/stable_diffusion"  # Changed from ../pretrained_models
    sd_ckpt_path: str = "models/sd15/v1-5-pruned-emaonly.safetensors"
    sd_yaml_path: str = "configs/stable-diffusion/v1-inference.yaml"
    sd_tokenizer_path: str = "models/clip-vit-large-patch14"
    sd_diffusers_path: str = "models/sd15_diffusers"
    
    # Cache directory
    hf_cache_dir: str = "models/hub"
    
    # Device settings
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    enable_cpu_offload: bool = True
    
    def resolve_paths(self, base_dir: Optional[Path] = None):
        """Resolve paths relative to a base directory."""
        if base_dir is None:
            base_dir = Path.cwd()
        
        # Update paths to be absolute
        self.audioldm2_path = str(base_dir / self.audioldm2_path)
        
        sd_root = base_dir / self.sd_root
        self.sd_root = str(sd_root)
        self.sd_ckpt_path = str(sd_root / self.sd_ckpt_path)
        self.sd_yaml_path = str(sd_root / self.sd_yaml_path)
        self.sd_tokenizer_path = str(sd_root / self.sd_tokenizer_path)
        self.sd_diffusers_path = str(sd_root / self.sd_diffusers_path)
        
        self.hf_cache_dir = str(base_dir / self.hf_cache_dir)


class DiffusionModelLoader:
    """Manages loading and caching of diffusion models."""
    
    def __init__(self, config: Optional[DiffusionConfig] = None, base_dir: Optional[Path] = None):
        """
        Args:
            config: Configuration for model paths
            base_dir: Base directory for resolving relative paths
        """
        self.config = config or DiffusionConfig()
        
        # Resolve paths based on where we're running from
        if base_dir is None:
            # Auto-detect based on current file location
            current_file = Path(__file__)
            if "src" in current_file.parts:
                # We're in src/utils/, go up to src/
                base_dir = current_file.parent.parent
            else:
                base_dir = Path.cwd()
        
        self.config.resolve_paths(base_dir)
        
        # Set HF cache directory
        os.environ["HF_HOME"] = self.config.hf_cache_dir
        
        # Cache loaded models
        self._sd_pipe = None
        self._audio_pipe = None
        self._sd_vae = None
        self._audio_vae = None
    
    def check_sd_files(self) -> Tuple[bool, str]:
        """Check if SD files exist and return status."""
        # Check if already converted to diffusers format
        diffusers_path = Path(self.config.sd_diffusers_path)
        if (diffusers_path / "model_index.json").exists():
            return True, "SD already in diffusers format - ready to use"
        
        # If no diffusers format, check if we can convert from checkpoint
        ckpt_path = Path(self.config.sd_ckpt_path)
        yaml_path = Path(self.config.sd_yaml_path)
        
        missing_files = []
        if not ckpt_path.exists():
            missing_files.append(f"Checkpoint: {ckpt_path}")
        if not yaml_path.exists():
            missing_files.append(f"Config: {yaml_path}")
        
        if missing_files:
            error_msg = f"No diffusers format at {diffusers_path} and cannot convert.\n"
            error_msg += "Missing checkpoint files:\n" + "\n".join(missing_files)
            error_msg += f"\n\nEither:\n"
            error_msg += f"1. Place already-converted model in: {diffusers_path}\n"
            error_msg += f"2. Or download SD v1.5 checkpoint to: {ckpt_path}"
            return False, error_msg
        
        return True, "SD checkpoint files ready for conversion"
    
    def check_audioldm2_files(self) -> Tuple[bool, str]:
        """Check if AudioLDM2 files exist."""
        audio_path = Path(self.config.audioldm2_path)
        if not audio_path.exists():
            error_msg = f"AudioLDM2 directory not found: {audio_path}"
            error_msg += "\n\nPlease download AudioLDM2 to this directory."
            error_msg += "\nYou can use: git clone https://huggingface.co/cvssp/audioldm2-large"
            return False, error_msg
        
        # Check for key files
        if not (audio_path / "model_index.json").exists():
            error_msg = f"AudioLDM2 appears incomplete. Missing model_index.json in {audio_path}"
            return False, error_msg
        
        return True, "AudioLDM2 ready"
    
    def _ensure_sd_diffusers(self):
        """Convert SD checkpoint to diffusers format if needed."""
        out_dir = Path(self.config.sd_diffusers_path)
        
        # Check if already converted to diffusers format
        if (out_dir / "model_index.json").exists():
            print(f"✅ SD snapshot already exists at {self.config.sd_diffusers_path}")
            return
        
        print("SD snapshot not found → converting checkpoint...")
        
        # Only check for checkpoint files if we need to convert
        ckpt_path = Path(self.config.sd_ckpt_path)
        yaml_path = Path(self.config.sd_yaml_path)
        
        if not ckpt_path.exists() or not yaml_path.exists():
            print(f"⚠️  Cannot convert: checkpoint files not found")
            print(f"   Expected checkpoint: {ckpt_path}")
            print(f"   Expected config: {yaml_path}")
            raise FileNotFoundError(f"SD checkpoint files not found and no diffusers format available at {out_dir}")
        
        from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
            download_from_original_stable_diffusion_ckpt,
        )
        
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # The function signature might vary between diffusers versions
        # Try the newer signature first, fall back to older
        try:
            pipe = download_from_original_stable_diffusion_ckpt(
                checkpoint_path=self.config.sd_ckpt_path,
                original_config_file=self.config.sd_yaml_path,
                from_safetensors=True,
                extract_ema=True,
                device="cpu",
                load_safety_checker=False,
            )
        except TypeError:
            # Older diffusers version without named arguments
            print("Using older diffusers API...")
            pipe = download_from_original_stable_diffusion_ckpt(
                self.config.sd_ckpt_path,
                self.config.sd_yaml_path,
                from_safetensors=True,
                extract_ema=True,
                device="cpu",
                load_safety_checker=False,
            )
        
        pipe.save_pretrained(out_dir, safe_serialization=True)
        
        # Create tokenizer symlink if needed (for older diffusers)
        tok_link = out_dir / "tokenizer"
        if not tok_link.exists():
            tok_path = Path(self.config.sd_tokenizer_path)
            if tok_path.exists():
                tok_link.symlink_to(tok_path.resolve())
        
        print(f"✅ SD snapshot saved to {out_dir}")
    
    def load_stable_diffusion(self, vae_only: bool = False) -> Any:
        """
        Load Stable Diffusion pipeline or just VAE.
        
        Args:
            vae_only: If True, only load and return the VAE
            
        Returns:
            Full pipeline or just VAE depending on vae_only
        """
        if vae_only and self._sd_vae is not None:
            return self._sd_vae
        
        if self._sd_pipe is None:
            # Ensure diffusers format exists
            self._ensure_sd_diffusers()
            
            from diffusers import StableDiffusionPipeline
            
            print(f"Loading Stable Diffusion from {self.config.sd_diffusers_path}")
            
            self._sd_pipe = StableDiffusionPipeline.from_pretrained(
                self.config.sd_diffusers_path,
                torch_dtype=self.config.dtype,
                safety_checker=None,
                local_files_only=True
            ).to(self.config.device)
            
            if self.config.enable_cpu_offload:
                self._sd_pipe.enable_model_cpu_offload()
            
            # Cache VAE
            self._sd_vae = self._sd_pipe.vae
            
            print("✅ Stable Diffusion loaded")
        
        return self._sd_vae if vae_only else self._sd_pipe
    
    def load_audioldm2(self, vae_only: bool = False) -> Any:
        """
        Load AudioLDM2 pipeline or just VAE.
        
        Args:
            vae_only: If True, only load and return the VAE
            
        Returns:
            Full pipeline or just VAE depending on vae_only
        """
        if vae_only and self._audio_vae is not None:
            return self._audio_vae
        
        if self._audio_pipe is None:
            # Check files first
            files_ok, msg = self.check_audioldm2_files()
            if not files_ok:
                raise FileNotFoundError(msg)
            
            from diffusers import AudioLDM2Pipeline
            
            print(f"Loading AudioLDM2 from {self.config.audioldm2_path}")
            
            self._audio_pipe = AudioLDM2Pipeline.from_pretrained(
                self.config.audioldm2_path,
                torch_dtype=self.config.dtype,
                local_files_only=True
            ).to(self.config.device)
            
            if self.config.enable_cpu_offload:
                self._audio_pipe.enable_model_cpu_offload()
            
            # Cache VAE
            self._audio_vae = self._audio_pipe.vae
            
            print("✅ AudioLDM2 loaded")
        
        return self._audio_vae if vae_only else self._audio_pipe
    
    def load_vaes(self, skip_missing: bool = False) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Load both VAEs for latent space training.
        
        Args:
            skip_missing: If True, return None for missing models instead of raising error
            
        Returns:
            Tuple of (sd_vae, audio_vae) - may be None if skip_missing=True
        """
        sd_vae = None
        audio_vae = None
        
        # Try loading SD VAE
        try:
            sd_vae = self.load_stable_diffusion(vae_only=True)
        except FileNotFoundError as e:
            if skip_missing:
                print(f"⚠️  Skipping SD VAE: {str(e).split('Please')[0]}")
            else:
                raise
        
        # Try loading AudioLDM2 VAE
        try:
            audio_vae = self.load_audioldm2(vae_only=True)
        except FileNotFoundError as e:
            if skip_missing:
                print(f"⚠️  Skipping AudioLDM2 VAE: {str(e).split('Please')[0]}")
            else:
                raise
        
        if not skip_missing and (sd_vae is None or audio_vae is None):
            raise RuntimeError("Failed to load VAEs. Use skip_missing=True to continue without them.")
        
        return sd_vae, audio_vae
    
    def cleanup(self):
        """Free memory by deleting loaded models."""
        if self._sd_pipe is not None:
            del self._sd_pipe
            self._sd_pipe = None
        
        if self._audio_pipe is not None:
            del self._audio_pipe
            self._audio_pipe = None
        
        # Keep VAE references if they were loaded separately
        if self._sd_vae is not None:
            del self._sd_vae
            self._sd_vae = None
        
        if self._audio_vae is not None:
            del self._audio_vae
            self._audio_vae = None
        
        torch.cuda.empty_cache()
        print("✅ Model memory freed")


# Convenience function for quick loading
def load_diffusion_vaes(
    audioldm2_path: Optional[str] = None,
    sd_path: Optional[str] = None,
    base_dir: Optional[Path] = None,
    skip_missing: bool = False
) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Quick function to load both VAEs.
    
    Args:
        audioldm2_path: Path to AudioLDM2 model
        sd_path: Path to SD diffusers model
        base_dir: Base directory for paths
        skip_missing: If True, return None for missing models
        
    Returns:
        Tuple of (sd_vae, audio_vae)
    """
    config = DiffusionConfig()
    
    if audioldm2_path:
        config.audioldm2_path = audioldm2_path
    if sd_path:
        config.sd_diffusers_path = sd_path
    
    loader = DiffusionModelLoader(config, base_dir)
    return loader.load_vaes(skip_missing=skip_missing)


# For notebook testing
if __name__ == "__main__":
    # Test loading with file checking
    print("Testing diffusion model loading with file checking...")
    
    loader = DiffusionModelLoader()
    
    # Check files first
    print("\n1. Checking file availability...")
    sd_ok, sd_msg = loader.check_sd_files()
    audio_ok, audio_msg = loader.check_audioldm2_files()
    
    print(f"   SD: {sd_msg}")
    print(f"   AudioLDM2: {audio_msg}")
    
    if sd_ok and audio_ok:
        print("\n2. Loading VAEs...")
        sd_vae, audio_vae = loader.load_vaes()
        print(f"   Both VAEs loaded successfully!")
    else:
        print("\n2. Trying to load with skip_missing=True...")
        sd_vae, audio_vae = loader.load_vaes(skip_missing=True)
        print(f"   SD VAE: {'Loaded' if sd_vae else 'Skipped'}")
        print(f"   Audio VAE: {'Loaded' if audio_vae else 'Skipped'}")
    
    print("\n✅ Test complete!")