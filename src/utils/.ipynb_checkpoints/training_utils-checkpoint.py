# utils/training_utils.py
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
from tqdm import tqdm



class MITrainingStep:
    """Handles training steps for MI models."""
    
    @staticmethod
    def latent_space_step(
        model, batch, 
        audio_vae, image_vae,  # More explicit naming
        audio_schedule, image_schedule,
        loss_fn, optimizer, 
        device='cuda',
        vae_scale_factor=0.18215
    ):
        """
        Training step for latent space models (SD + AudioLDM).
        
        Args:
            model: MI model
            batch: Data batch with 'audio', 'image', 'is_real' keys
            audio_vae: AudioLDM VAE
            image_vae: Stable Diffusion VAE
            audio_schedule: Diffusion schedule for audio
            image_schedule: Diffusion schedule for images
            loss_fn: Loss function
            optimizer: Optimizer
            device: Device to run on
            vae_scale_factor: Scaling factor for SD VAE latents
        """
        # Get data
        audio = batch['audio'].to(device)  # [B, n_mels, time]
        image = batch['image'].to(device)  # [B, 3, H, W]
        is_real = batch['is_real'].to(device)
        
        # Separate real/fake
        real_mask = is_real > 0.5
        fake_mask = ~real_mask
        
        if real_mask.sum() == 0 or fake_mask.sum() == 0:
            return {'loss': 0, 'skipped': True}
        
        # Encode to latent space
        with torch.no_grad():
            # Process images with image VAE
            image_real = image[real_mask]
            image_fake = image[fake_mask]
            
            # Normalize images for SD
            image_real = (image_real - 0.5) * 2.0  # [0,1] -> [-1,1]
            image_fake = (image_fake - 0.5) * 2.0
            
            image_latent_real = image_vae.encode(image_real.half()).latent_dist.sample() * vae_scale_factor
            image_latent_fake = image_vae.encode(image_fake.half()).latent_dist.sample() * vae_scale_factor
            
            # Process audio with audio VAE - need 4D input for AudioLDM2
            audio_real = audio[real_mask].unsqueeze(1)  # Add channel dim
            audio_fake = audio[fake_mask].unsqueeze(1)
            
            audio_latent_real = audio_vae.encode(audio_real.half()).latent_dist.sample()
            audio_latent_fake = audio_vae.encode(audio_fake.half()).latent_dist.sample()
            
            # Convert back to float32
            audio_latent_real = audio_latent_real.float()
            audio_latent_fake = audio_latent_fake.float()
            image_latent_real = image_latent_real.float()
            image_latent_fake = image_latent_fake.float()
        
        # Sample timesteps
        batch_size_real = len(audio_latent_real)
        batch_size_fake = len(audio_latent_fake)
        
        t_real = torch.randint(0, 1000, (batch_size_real,), device=device)
        t_fake = torch.randint(0, 1000, (batch_size_fake,), device=device)
        
        # Add noise
        audio_noisy_real, _ = audio_schedule.add_noise(audio_latent_real, t_real)
        audio_noisy_fake, _ = audio_schedule.add_noise(audio_latent_fake, t_fake)
        image_noisy_real, _ = image_schedule.add_noise(image_latent_real, t_real)
        image_noisy_fake, _ = image_schedule.add_noise(image_latent_fake, t_fake)
        
        # Forward through model
        scores_real = model(audio_noisy_real, image_noisy_real, t_real)
        scores_fake = model(audio_noisy_fake, image_noisy_fake, t_fake)
        
        # Compute loss
        loss, metrics = loss_fn(scores_real, scores_fake)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        metrics['loss'] = loss.item()
        metrics['t_mean'] = (t_real.float().mean() + t_fake.float().mean()).item() / 2
        
        return metrics
    
    @staticmethod
    def pixel_space_step(
        model, batch,
        schedule_1, schedule_2,
        loss_fn, optimizer,
        device='cuda'
    ):
        """Training step for pixel space models (DDPM) - unchanged from before."""
        # Get data - note the key names are reused from audio-visual setup
        input_1 = batch['audio'].to(device)  # night images
        input_2 = batch['image'].to(device)  # day images
        is_real = batch['is_real'].to(device)
        
        # Separate real/fake
        real_mask = is_real > 0.5
        fake_mask = ~real_mask
        
        if real_mask.sum() == 0 or fake_mask.sum() == 0:
            return {'loss': 0, 'skipped': True}
        
        input_1_real = input_1[real_mask]
        input_2_real = input_2[real_mask]
        input_1_fake = input_1[fake_mask]
        input_2_fake = input_2[fake_mask]
        
        # Sample timesteps
        batch_size_real = len(input_1_real)
        batch_size_fake = len(input_1_fake)
        
        t_real = torch.randint(0, 1000, (batch_size_real,), device=device)
        t_fake = torch.randint(0, 1000, (batch_size_fake,), device=device)
        
        # Add noise directly to pixel space
        noisy_1_real, _ = schedule_1.add_noise(input_1_real, t_real)
        noisy_1_fake, _ = schedule_1.add_noise(input_1_fake, t_fake)
        noisy_2_real, _ = schedule_2.add_noise(input_2_real, t_real)
        noisy_2_fake, _ = schedule_2.add_noise(input_2_fake, t_fake)
        
        # Forward through model
        scores_real = model(noisy_1_real, noisy_2_real, t_real)
        scores_fake = model(noisy_1_fake, noisy_2_fake, t_fake)
        
        # Compute loss
        loss, metrics = loss_fn(scores_real, scores_fake)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        metrics['loss'] = loss.item()
        metrics['t_mean'] = (t_real.float().mean() + t_fake.float().mean()).item() / 2
        
        return metrics
    
    # @staticmethod
    # def validation_step(
    #     model, batch,
    #     schedule_1, schedule_2,
    #     loss_fn, device='cuda',
    #     vae_1=None, vae_2=None,
    #     vae_scale_factor=0.18215
    # ):
    #     """
    #     Validation step that works for both latent and pixel space.
        
    #     Args:
    #         model: MI model
    #         batch: Data batch
    #         schedule_1, schedule_2: Diffusion schedules
    #         loss_fn: Loss function
    #         device: Device
    #         vae_1, vae_2: Optional VAEs for latent space encoding
    #         vae_scale_factor: VAE scaling factor
    #     """
    #     with torch.no_grad():
    #         # Get data
    #         input_1 = batch['audio'].to(device)
    #         input_2 = batch['image'].to(device)
    #         is_real = batch['is_real'].to(device)
            
    #         # Separate real/fake
    #         real_mask = is_real > 0.5
    #         fake_mask = ~real_mask
            
    #         if real_mask.sum() == 0 or fake_mask.sum() == 0:
    #             return {'loss': 0, 'skipped': True}
            
    #         # Process based on whether VAEs are provided
    #         if vae_1 is not None and vae_2 is not None:
    #             # Latent space validation
    #             input_2_real = input_2[real_mask]
    #             input_2_fake = input_2[fake_mask]
                
    #             # Normalize images
    #             input_2_real = (input_2_real - 0.5) * 2.0
    #             input_2_fake = (input_2_fake - 0.5) * 2.0
                
    #             latent_2_real = vae_2.encode(input_2_real.half()).latent_dist.sample() * vae_scale_factor
    #             latent_2_fake = vae_2.encode(input_2_fake.half()).latent_dist.sample() * vae_scale_factor
                
    #             # Audio encoding
    #             input_1_real = input_1[real_mask].unsqueeze(1)
    #             input_1_fake = input_1[fake_mask].unsqueeze(1)
                
    #             latent_1_real = vae_1.encode(input_1_real.half()).latent_dist.sample()
    #             latent_1_fake = vae_1.encode(input_1_fake.half()).latent_dist.sample()
                
    #             # Convert to float32
    #             data_1_real = latent_1_real.float()
    #             data_1_fake = latent_1_fake.float()
    #             data_2_real = latent_2_real.float()
    #             data_2_fake = latent_2_fake.float()
    #         else:
    #             # Pixel space validation
    #             data_1_real = input_1[real_mask]
    #             data_2_real = input_2[real_mask]
    #             data_1_fake = input_1[fake_mask]
    #             data_2_fake = input_2[fake_mask]
            
    #         # Sample timesteps
    #         t_real = torch.randint(0, 1000, (len(data_1_real),), device=device)
    #         t_fake = torch.randint(0, 1000, (len(data_1_fake),), device=device)
            
    #         # Add noise
    #         noisy_1_real, _ = schedule_1.add_noise(data_1_real, t_real)
    #         noisy_1_fake, _ = schedule_1.add_noise(data_1_fake, t_fake)
    #         noisy_2_real, _ = schedule_2.add_noise(data_2_real, t_real)
    #         noisy_2_fake, _ = schedule_2.add_noise(data_2_fake, t_fake)
            
    #         # Forward through model
    #         scores_real = model(noisy_1_real, noisy_2_real, t_real)
    #         scores_fake = model(noisy_1_fake, noisy_2_fake, t_fake)
            
    #         # Compute loss
    #         loss, metrics = loss_fn(scores_real, scores_fake)
    #         metrics['loss'] = loss.item()
            
    #         return metrics