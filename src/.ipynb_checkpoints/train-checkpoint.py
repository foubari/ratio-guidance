# src/run_training.py
import argparse
import torch
import random
import numpy as np
from pathlib import Path

from configs.experiment_configs import ExperimentConfig
from models.mi_models import UnifiedMIModel
from utils.trainer import MITrainer
from utils.density_ratio_losses import DensityRatioLoss
from utils.diffusion_schedule import DiffusionSchedule
from utils.diffusion_loaders import DiffusionModelLoader

def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_data(config):
    """Setup data loaders based on config."""
    if config.model.model_type == 'sd_audioldm':
        from dataloaders.ave import AVEContrastiveDataModule
        data_module = AVEContrastiveDataModule(
            ave_data_dir=config.data.ave_data_dir,
            fake_data_dir=config.data.fake_data_dir,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            audio_duration=config.data.audio_duration,
            audio_sr=config.data.audio_sr,
            image_size=config.data.image_size,
            real_fake_ratio=config.data.real_fake_ratio
        )
    else:  # ddpm_night2day
        from dataloaders.night2day import Night2DayMIDataModule
        data_module = Night2DayMIDataModule(
            data_path=config.data.data_path,
            # val_path=config.data.val_path,
            fake_night_dir=config.data.fake_night_dir,#: str = "../data/samples/night/1_2_4/tr_stp_70000_stp1000/2025_07_21_06_55",
            fake_day_dir=config.data.fake_day_dir,#: str = "../data/samples/day/1_2_4/tr_stp_70000_stp1000/2025_07_20_22_40",
            resolution=config.data.resolution,
            batch_size=config.data.batch_size,
            real_fake_ratio=config.data.real_fake_ratio
        )
    
    data_module.setup()
    return data_module.train_dataloader(), data_module.val_dataloader()

def setup_schedules(config):
    """Setup diffusion schedules based on config."""
    device = config.training.device
    
    if config.model.model_type == 'sd_audioldm':
        schedule_1 = DiffusionSchedule.audioldm_schedule(device=device)
        schedule_2 = DiffusionSchedule.stable_diffusion_schedule(device=device)
    else:  # ddpm_night2day
        schedule_1 = DiffusionSchedule.ddpm_schedule(device=device)
        schedule_2 = DiffusionSchedule.ddpm_schedule(device=device)
    
    return schedule_1, schedule_2

def setup_vaes(config):
    """Setup VAEs for latent space models."""
    if config.model.model_type == 'sd_audioldm':
        loader = DiffusionModelLoader()
        sd_vae, audio_vae = loader.load_vaes()
        return audio_vae, sd_vae  # Return in order: vae_1, vae_2
    return None, None

def main(args):
    # Set seed if provided
    if args.seed is not None:
        set_seed(args.seed)
        print(f"Set random seed: {args.seed}")
    
    # Load config
    if args.experiment == 'sd_audioldm':
        config = ExperimentConfig.sd_audioldm_config()
    elif args.experiment == 'ddpm_night2day':
        config = ExperimentConfig.ddpm_night2day_config()
    else:
        raise ValueError(f"Unknown experiment: {args.experiment}")
    
    # Override config with command line args if provided
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.loss_type:
        config.training.loss_type = args.loss_type
    if args.patience:
        config.training.patience = args.patience
    
    print(f"Running experiment: {config.name}")
    print(f"Model type: {config.model.model_type}")
    print(f"Loss type: {config.training.loss_type}")
    
    # Setup components
    device = torch.device(config.training.device)
    
    # Data
    print("Setting up data loaders...")
    train_loader, val_loader = setup_data(config)
    
    # Model
    print("Creating model...")
    model = UnifiedMIModel(config.model.model_type).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    
    # Loss
    loss_fn = DensityRatioLoss(loss_type=args.loss_type).to(device)
    
    # Schedules
    print("Setting up diffusion schedules...")
    schedule_1, schedule_2 = setup_schedules(config)
    
    # VAEs (if needed)
    vae_1, vae_2 = setup_vaes(config)
    if vae_1 is not None:
        print("VAEs loaded successfully")
    
    # Create save path with loss type subfolder
    save_path = Path(config.training.save_path) / args.loss_type
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Create trainer
    trainer = MITrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        schedule_1=schedule_1,
        schedule_2=schedule_2,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config.training.device,
        model_type=config.model.model_type,
        save_path=save_path,  # Now includes loss type in path
        patience=config.training.patience,
        vae_1=vae_1,
        vae_2=vae_2,
        vae_scale_factor=config.training.vae_scale_factor
    )
    
    # Train
    print(f"Starting training for {config.training.num_epochs} epochs...")
    print(f"Models will be saved to: {save_path}")
    trainer.train(config.training.num_epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MI models')
    parser.add_argument('experiment', choices=['sd_audioldm', 'ddpm_night2day'],
                        help='Which experiment to run')
    parser.add_argument('--loss_type', choices=['dv', 'disc', 'ulsif', 'rulsif', 'kliep'],
                        default='disc', help='Loss type for density ratio estimation')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--patience', type=int, default=3, help='Number of epcohs without val loss improvement')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    main(args)