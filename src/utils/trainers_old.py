"""
trainers.py
Training classes for MI estimation with and without diffusion noise
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional, Any

from .diffusion_schedule import DiffusionSchedule
from .utils import sample_timesteps


class MITrainer:
    """Base trainer for MI estimation without diffusion noise."""
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        learning_rate: float = 1e-4,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        self.train_metrics = []
        self.val_metrics = []
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.loss_fn.train()
        
        # Move data to device
        audio = batch['audio'].to(self.device)
        image = batch['image'].to(self.device)
        is_real = batch['is_real'].to(self.device)
        
        # Separate real and fake pairs
        real_mask = is_real > 0.5
        fake_mask = ~real_mask
        
        if real_mask.sum() == 0 or fake_mask.sum() == 0:
            return {'loss': 0, 'skipped': True}
        
        # Get real and fake pairs
        audio_real = audio[real_mask]
        image_real = image[real_mask]
        audio_fake = audio[fake_mask]
        image_fake = image[fake_mask]
        
        # Forward pass
        scores_real = self.model(audio_real, image_real)
        scores_fake = self.model(audio_fake, image_fake)
        
        # Compute loss
        loss, metrics = self.loss_fn(scores_real, scores_fake)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()
        
        metrics['loss'] = loss.item()
        return metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validation loop."""
        self.model.eval()
        self.loss_fn.eval()
        all_metrics = []
        
        with torch.no_grad():
            for batch in val_loader:
                audio = batch['audio'].to(self.device)
                image = batch['image'].to(self.device)
                is_real = batch['is_real'].to(self.device)
                
                real_mask = is_real > 0.5
                fake_mask = ~real_mask
                
                if real_mask.sum() == 0 or fake_mask.sum() == 0:
                    continue
                
                audio_real = audio[real_mask]
                image_real = image[real_mask]
                audio_fake = audio[fake_mask]
                image_fake = image[fake_mask]
                
                scores_real = self.model(audio_real, image_real)
                scores_fake = self.model(audio_fake, image_fake)
                
                loss, metrics = self.loss_fn(scores_real, scores_fake)
                metrics['loss'] = loss.item()
                all_metrics.append(metrics)
        
        # Average metrics
        avg_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        return avg_metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 50,
        log_interval: int = 10,
        save_path: Optional[str] = None
    ):
        """Full training loop."""
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            epoch_metrics = []
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            for i, batch in enumerate(pbar):
                metrics = self.train_step(batch)
                
                if 'skipped' not in metrics:
                    epoch_metrics.append(metrics)
                    
                    if i % log_interval == 0 and epoch_metrics:
                        avg_loss = np.mean([m['loss'] for m in epoch_metrics[-log_interval:]])
                        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                print(f"Epoch {epoch+1} - Val Loss: {val_metrics.get('loss', 0):.4f}")
                self.val_metrics.append(val_metrics)
                
                # Save best model
                if save_path and val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    self.save_checkpoint(save_path, epoch, best_val_loss, val_metrics)
                    print(f"  -> Saved best model (val_loss: {best_val_loss:.4f})")
            
            # Store training metrics
            if epoch_metrics:
                avg_train_metrics = {
                    key: np.mean([m[key] for m in epoch_metrics])
                    for key in epoch_metrics[0].keys()
                }
                self.train_metrics.append(avg_train_metrics)
    
    def save_checkpoint(self, path: str, epoch: int, val_loss: float, val_metrics: Dict):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_fn_state_dict': self.loss_fn.state_dict(),
            'val_loss': val_loss,
            'val_metrics': val_metrics
        }, path)


class NoisyMITrainer(MITrainer):
    """Trainer for MI estimation with diffusion noise and timestep conditioning."""
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        learning_rate: float = 1e-4,
        device: str = 'cuda',
        # Diffusion schedules
        audio_schedule: Optional[DiffusionSchedule] = None,
        image_schedule: Optional[DiffusionSchedule] = None,
        num_timesteps: int = 1000,
        # VAE encoders (optional)
        sd_vae: Optional[Any] = None,
        audio_vae: Optional[Any] = None,
        sd_scaling_factor: float = 0.18215,
        # Timestep sampling
        timestep_sampling: str = "uniform",
        min_timestep: int = 0,
        max_timestep: int = 999
    ):
        super().__init__(model, loss_fn, learning_rate, device)
        
        # Check if we're in latent space mode
        self.use_latent_space = (sd_vae is not None and audio_vae is not None)
        
        # Store VAEs if provided
        self.sd_vae = sd_vae
        self.audio_vae = audio_vae
        self.sd_scaling_factor = sd_scaling_factor
        
        # Initialize diffusion schedules
        if audio_schedule is None:
            audio_schedule = DiffusionSchedule.from_audio_ldm(num_timesteps)
        if image_schedule is None:
            image_schedule = DiffusionSchedule.from_stable_diffusion(num_timesteps)
            
        self.audio_schedule = audio_schedule
        self.image_schedule = image_schedule
        self.num_timesteps = num_timesteps
        
        # Timestep sampling settings
        self.timestep_sampling = timestep_sampling
        self.min_timestep = min_timestep
        self.max_timestep = max_timestep
    
    @torch.no_grad()
    def encode_to_latent(self, audio: torch.Tensor, image: torch.Tensor):
        """Encode audio and image to latent space using VAEs."""
        if not self.use_latent_space:
            return audio, image
        
        # Encode image to latent
        if image.max() > 1.0:
            image = (image - 0.5) * 2.0  # [0,1] -> [-1,1]
        image = image.half()
        image_latent_dist = self.sd_vae.encode(image)
        image_latent = image_latent_dist.latent_dist.sample() * self.sd_scaling_factor
        
        # Encode audio to latent
        if audio.dim() == 3:
            audio = audio.unsqueeze(1)  # Add this line - AudioLDM2 needs 4D
        audio = audio.half()
        audio_latent_dist = self.audio_vae.encode(audio)
        audio_latent = audio_latent_dist.latent_dist.sample()
        
        return audio_latent.float(), image_latent.float()
    
    def add_noise_to_batch(self, audio: torch.Tensor, image: torch.Tensor, t: torch.Tensor):
        """Add noise to audio and image batches at timestep t."""
        self.audio_schedule.alpha_cumprod = self.audio_schedule.alpha_cumprod.to(self.device)
        self.image_schedule.alpha_cumprod = self.image_schedule.alpha_cumprod.to(self.device)
        
        noisy_audio, audio_noise = self.audio_schedule.add_noise(audio, t)
        noisy_image, image_noise = self.image_schedule.add_noise(image, t)
        
        return noisy_audio, noisy_image, audio_noise, image_noise
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with noise."""
        self.model.train()
        self.loss_fn.train()
        
        # Move data to device
        audio = batch['audio'].to(self.device)
        image = batch['image'].to(self.device)
        is_real = batch['is_real'].to(self.device)
        
        # Separate real and fake pairs
        real_mask = is_real > 0.5
        fake_mask = ~real_mask
        
        if real_mask.sum() == 0 or fake_mask.sum() == 0:
            return {'loss': 0, 'skipped': True}
        
        audio_real = audio[real_mask]
        image_real = image[real_mask]
        audio_fake = audio[fake_mask]
        image_fake = image[fake_mask]
        
        # Encode to latent space if using VAEs
        if self.use_latent_space:
            audio_real, image_real = self.encode_to_latent(audio_real, image_real)
            audio_fake, image_fake = self.encode_to_latent(audio_fake, image_fake)
        
        # Sample timesteps
        t_real = sample_timesteps(
            len(audio_real), self.min_timestep, self.max_timestep, 
            self.timestep_sampling, self.device
        )
        t_fake = sample_timesteps(
            len(audio_fake), self.min_timestep, self.max_timestep,
            self.timestep_sampling, self.device
        )
        
        # Add noise
        noisy_audio_real, noisy_image_real, _, _ = self.add_noise_to_batch(
            audio_real, image_real, t_real
        )
        noisy_audio_fake, noisy_image_fake, _, _ = self.add_noise_to_batch(
            audio_fake, image_fake, t_fake
        )
        
        # Forward pass with timesteps
        scores_real = self.model(noisy_audio_real, noisy_image_real, t_real)
        scores_fake = self.model(noisy_audio_fake, noisy_image_fake, t_fake)
        
        # Compute loss
        loss, metrics = self.loss_fn(scores_real, scores_fake)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()
        
        metrics['loss'] = loss.item()
        metrics['t_real_mean'] = t_real.float().mean().item()
        metrics['t_fake_mean'] = t_fake.float().mean().item()
        metrics['latent_space'] = self.use_latent_space
        
        return metrics
    
    def save_checkpoint(self, path: str, epoch: int, val_loss: float, val_metrics: Dict):
        """Save model checkpoint with additional diffusion info."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_fn_state_dict': self.loss_fn.state_dict(),
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'use_latent_space': self.use_latent_space,
            'audio_schedule': self.audio_schedule,
            'image_schedule': self.image_schedule,
            'num_timesteps': self.num_timesteps
        }, path)