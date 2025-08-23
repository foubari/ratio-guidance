# utils/trainer.py
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Dict, Any
from utils.training_utils import MITrainingStep
from torch.utils.tensorboard import SummaryWriter


class MITrainer:
    """Unified trainer for MI models."""
    
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        schedule_1,
        schedule_2,
        train_loader,
        val_loader,
        device='cuda',
        model_type='sd_audioldm',  # or 'ddpm_night2day'
        save_path='checkpoints',
        patience=3,
        vae_1=None,  # For latent space models
        vae_2=None,
        vae_scale_factor=0.18215
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.schedule_1 = schedule_1
        self.schedule_2 = schedule_2
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model_type = model_type
        self.save_path = Path(save_path)
        self.save_path.mkdir(exist_ok=True)
        self.patience = patience
        self.vae_1 = vae_1
        self.vae_2 = vae_2
        self.vae_scale_factor = vae_scale_factor
        
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.training_step = MITrainingStep()
        
        # TensorBoard writer
        self.writer = SummaryWriter(f'runs/{model_type}_{loss_fn.loss_type}')

    def train_epoch(self, epoch, num_epochs):
        """Train for one epoch."""
        self.model.train()
        train_metrics = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for i, batch in enumerate(pbar):
            if self.model_type == 'sd_audioldm':
                metrics = self.training_step.latent_space_step(
                    self.model, batch,
                    self.vae_1, self.vae_2,  # audio_vae, image_vae
                    self.schedule_1, self.schedule_2,  # audio_schedule, image_schedule
                    self.loss_fn, self.optimizer, self.device,
                    self.vae_scale_factor
                )
            else:  # ddpm_night2day
                metrics = self.training_step.pixel_space_step(
                    self.model, batch,
                    self.schedule_1, self.schedule_2,
                    self.loss_fn, self.optimizer, self.device
                )
            
            if 'skipped' not in metrics:
                train_metrics.append(metrics)
                
                # Log batch-level metrics to TensorBoard
                global_step = epoch * len(self.train_loader) + i
                if i % 10 == 0:  # Log every 10 batches
                    self.writer.add_scalar('Batch/loss', metrics['loss'], global_step)
                    if 't_mean' in metrics:
                        self.writer.add_scalar('Batch/t_mean', metrics['t_mean'], global_step)
                    
                    # Log loss-specific metrics
                    for key, value in metrics.items():
                        if key not in ['loss', 't_mean', 'skipped']:
                            self.writer.add_scalar(f'Metrics/{key}', value, global_step)
                
                if len(train_metrics) % 10 == 0:
                    avg_loss = np.mean([m['loss'] for m in train_metrics[-10:]])
                    pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        return train_metrics
    
    def validate(self):
        """Run validation."""
        self.model.eval()
        val_metrics = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                if self.model_type == 'sd_audioldm':
                    # Latent space validation
                    metrics = self._validate_latent_step(batch)
                else:
                    # Pixel space validation
                    metrics = self._validate_pixel_step(batch)
                
                if 'skipped' not in metrics:
                    val_metrics.append(metrics)
        
        return val_metrics
    
    def _validate_latent_step(self, batch):
        """Validation step for latent space models."""
        audio = batch['audio'].to(self.device)
        image = batch['image'].to(self.device)
        is_real = batch['is_real'].to(self.device)
        
        real_mask = is_real > 0.5
        fake_mask = ~real_mask
        
        if real_mask.sum() == 0 or fake_mask.sum() == 0:
            return {'loss': 0, 'skipped': True}
        
        # Encode to latent space
        image_real = image[real_mask]
        image_fake = image[fake_mask]
        image_real = (image_real - 0.5) * 2.0
        image_fake = (image_fake - 0.5) * 2.0
        
        image_latent_real = self.vae_2.encode(image_real.half()).latent_dist.sample() * self.vae_scale_factor
        image_latent_fake = self.vae_2.encode(image_fake.half()).latent_dist.sample() * self.vae_scale_factor
        
        audio_real = audio[real_mask].unsqueeze(1)
        audio_fake = audio[fake_mask].unsqueeze(1)
        
        audio_latent_real = self.vae_1.encode(audio_real.half()).latent_dist.sample()
        audio_latent_fake = self.vae_1.encode(audio_fake.half()).latent_dist.sample()
        
        audio_latent_real = audio_latent_real.float()
        audio_latent_fake = audio_latent_fake.float()
        image_latent_real = image_latent_real.float()
        image_latent_fake = image_latent_fake.float()
        
        # Sample timesteps and add noise
        t_real = torch.randint(0, 1000, (len(audio_latent_real),), device=self.device)
        t_fake = torch.randint(0, 1000, (len(audio_latent_fake),), device=self.device)
        
        audio_noisy_real, _ = self.schedule_1.add_noise(audio_latent_real, t_real)
        audio_noisy_fake, _ = self.schedule_1.add_noise(audio_latent_fake, t_fake)
        image_noisy_real, _ = self.schedule_2.add_noise(image_latent_real, t_real)
        image_noisy_fake, _ = self.schedule_2.add_noise(image_latent_fake, t_fake)
        
        # Forward and compute loss
        scores_real = self.model(audio_noisy_real, image_noisy_real, t_real)
        scores_fake = self.model(audio_noisy_fake, image_noisy_fake, t_fake)
        
        loss, metrics = self.loss_fn(scores_real, scores_fake)
        metrics['loss'] = loss.item()
        
        return metrics
    
    def _validate_pixel_step(self, batch):
        """Validation step for pixel space models."""
        night = batch['audio'].to(self.device)
        day = batch['image'].to(self.device)
        is_real = batch['is_real'].to(self.device)
        
        real_mask = is_real > 0.5
        fake_mask = ~real_mask
        
        if real_mask.sum() == 0 or fake_mask.sum() == 0:
            return {'loss': 0, 'skipped': True}
        
        night_real = night[real_mask]
        day_real = day[real_mask]
        night_fake = night[fake_mask]
        day_fake = day[fake_mask]
        
        # Sample timesteps and add noise
        t_real = torch.randint(0, 1000, (len(night_real),), device=self.device)
        t_fake = torch.randint(0, 1000, (len(night_fake),), device=self.device)
        
        night_real_noisy, _ = self.schedule_1.add_noise(night_real, t_real)
        night_fake_noisy, _ = self.schedule_1.add_noise(night_fake, t_fake)
        day_real_noisy, _ = self.schedule_2.add_noise(day_real, t_real)
        day_fake_noisy, _ = self.schedule_2.add_noise(day_fake, t_fake)
        
        # Forward and compute loss
        scores_real = self.model(night_real_noisy, day_real_noisy, t_real)
        scores_fake = self.model(night_fake_noisy, day_fake_noisy, t_fake)
        
        loss, metrics = self.loss_fn(scores_real, scores_fake)
        metrics['loss'] = loss.item()
        
        return metrics

    def train(self, num_epochs):
        """Full training loop."""
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch, num_epochs)
            avg_train_loss = np.mean([m['loss'] for m in train_metrics]) if train_metrics else float('inf')
            
            # Validate
            val_metrics = self.validate()
            avg_val_loss = np.mean([m['loss'] for m in val_metrics]) if val_metrics else float('inf')
            
            # Log epoch-level metrics to TensorBoard
            self.writer.add_scalar('Loss/train', avg_train_loss, epoch)
            self.writer.add_scalar('Loss/val', avg_val_loss, epoch)
            self.writer.add_scalar('Learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Calculate and log additional epoch metrics
            if train_metrics:
                avg_t_mean = np.mean([m.get('t_mean', 0) for m in train_metrics])
                self.writer.add_scalar('Epoch/avg_timestep', avg_t_mean, epoch)
            
            print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
            
            # Save best model and check early stopping
            if avg_val_loss < self.best_loss:
                self.best_loss = avg_val_loss
                self.patience_counter = 0
                
                # Include loss_type in filename if loss_fn has it
                loss_type = getattr(self.loss_fn, 'loss_type', 'unknown')
                save_name = f'{self.model_type}_{loss_type}_mi_model_best.pt'

                # In utils/trainer.py, update the save block (around line 195):
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'train_loss': avg_train_loss,
                    'val_loss': self.best_loss,
                    'loss_type': loss_type,
                    'model_type': self.model_type,
                    # NEW: Save loss hyperparameters
                    'loss_hparams': {
                        'use_exp_w': getattr(self.loss_fn, 'use_exp_w', False),
                        'rulsif_alpha': getattr(self.loss_fn, 'rulsif_alpha', None),
                        'rulsif_link': getattr(self.loss_fn, 'rulsif_link', None),
                    }
                }, self.save_path / save_name)

                # torch.save({
                #     'model_state_dict': self.model.state_dict(),
                #     'optimizer_state_dict': self.optimizer.state_dict(),
                #     'epoch': epoch,
                #     'train_loss': avg_train_loss,
                #     'val_loss': self.best_loss,
                #     'loss_type': loss_type,
                #     'model_type': self.model_type
                # }, self.save_path / save_name)
                print(f'  -> Saved best model (val_loss: {self.best_loss:.4f})')
            else:
                self.patience_counter += 1
                print(f'  -> No improvement. Patience: {self.patience_counter}/{self.patience}')
                
                if self.patience_counter >= self.patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    break
        
        # Close TensorBoard writer
        self.writer.close()
        print(f'Training complete! Best validation loss: {self.best_loss:.4f}')
