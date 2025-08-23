"""
main.py
Main script for training MI estimation models
Place this file in src/ directory
"""

import argparse
import torch
from pathlib import Path

# Import your dataloader
from dataloaders.ave import AVEContrastiveDataModule

# Import local modules
from models.image_sound_guide import MIModel, FlexibleMIModel
from utils.density_ratio_losses import DensityRatioLoss
from utils.trainers import MITrainer, NoisyMITrainer
from utils.diffusion_schedule import DiffusionSchedule
from utils.diffusion_loaders import DiffusionModelLoader, DiffusionConfig


def parse_args():
    parser = argparse.ArgumentParser(description='Train MI estimation models')
    
    # Data arguments
    parser.add_argument('--ave_data_dir', type=str, 
                       default='../data/AVE_Dataset/ave_processed',
                       help='Path to AVE dataset')
    parser.add_argument('--fake_data_dir', type=str,
                       default='unconditional_samples_50',
                       help='Path to fake/generated data')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='flexible',
                       choices=['basic', 'flexible'],
                       help='Type of model to use')
    parser.add_argument('--use_latent_space', action='store_true',
                       help='Use latent space (requires VAEs)')
    parser.add_argument('--use_timestep', action='store_true',
                       help='Use timestep conditioning (for diffusion)')
    parser.add_argument('--n_mels', type=int, default=64,
                       help='Number of mel bands for audio')
    parser.add_argument('--feature_dim', type=int, default=256,
                       help='Feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Hidden dimension')
    
    # Loss arguments
    parser.add_argument('--loss_type', type=str, default='disc',
                       choices=['dv', 'disc', 'ulsif', 'rulsif', 'kliep'],
                       help='Loss function type')
    parser.add_argument('--dv_use_ema', action='store_true', default=True,
                       help='Use EMA for DV loss')
    parser.add_argument('--dv_ema_rate', type=float, default=0.99,
                       help='EMA rate for DV loss')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Logging interval')
    parser.add_argument('--save_path', type=str, default='checkpoints/model.pt',
                       help='Path to save best model')
    
    # Diffusion arguments (for noisy training)
    parser.add_argument('--use_diffusion_noise', action='store_true',
                       help='Train with diffusion noise')
    parser.add_argument('--num_timesteps', type=int, default=1000,
                       help='Number of diffusion timesteps')
    parser.add_argument('--timestep_sampling', type=str, default='uniform',
                       choices=['uniform', 'importance', 'antithetic'],
                       help='Timestep sampling strategy')
    parser.add_argument('--min_timestep', type=int, default=0,
                       help='Minimum timestep for sampling')
    parser.add_argument('--max_timestep', type=int, default=999,
                       help='Maximum timestep for sampling')
    
    # Diffusion model paths (for latent space)
    parser.add_argument('--audioldm2_path', type=str, 
                       default='../pretrained_models/audioldm2',
                       help='Path to AudioLDM2 model')
    parser.add_argument('--sd_root', type=str, 
                       default='../pretrained_models/stable_diffusion',
                       help='Root path for Stable Diffusion files')
    parser.add_argument('--audio_latent_shape', type=int, nargs='+', default=[8, 32],
                       help='Shape of audio latents')
    parser.add_argument('--image_latent_shape', type=int, nargs='+', default=[4, 64, 64],
                       help='Shape of image latents')
    
    return parser.parse_args()


def setup_data(args):
    """Setup data module and loaders."""
    data_module = AVEContrastiveDataModule(
        ave_data_dir=args.ave_data_dir,
        fake_data_dir=args.fake_data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        audio_duration=10.0,
        audio_sr=16000,
        image_size=(512, 512),
        real_fake_ratio=0.5  # 50% real, 50% fake
    )
    
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    return train_loader, val_loader


def setup_model(args):
    """Setup model based on arguments."""
    if args.model_type == 'basic':
        if args.use_timestep:
            raise ValueError("Basic model doesn't support timestep conditioning")
        model = MIModel(
            n_mels=args.n_mels,
            feature_dim=args.feature_dim,
            hidden_dim=args.hidden_dim
        )
    else:  # flexible
        model = FlexibleMIModel(
            n_mels=args.n_mels,
            image_size=(512, 512),
            audio_latent_shape=tuple(args.audio_latent_shape) if args.use_latent_space else None,
            image_latent_shape=tuple(args.image_latent_shape) if args.use_latent_space else None,
            hidden_dim=args.hidden_dim,
            feature_dim=args.feature_dim,
            use_latent_space=args.use_latent_space,
            use_timestep=args.use_timestep
        )
    
    return model


def setup_loss(args):
    """Setup loss function based on arguments."""
    loss_kwargs = {}
    
    if args.loss_type == 'dv':
        loss_kwargs.update({
            'dv_use_ema': args.dv_use_ema,
            'dv_ema_rate': args.dv_ema_rate
        })
    # Add other loss-specific parameters as needed
    
    loss_fn = DensityRatioLoss(
        loss_type=args.loss_type,
        **loss_kwargs
    )
    
    return loss_fn


def setup_vaes(args):
    """Setup VAE encoders using the new diffusion loader."""
    sd_vae = None
    audio_vae = None
    loader = None
    
    if args.use_latent_space:
        print("Setting up VAEs for latent space training...")
        
        # Create configuration
        config = DiffusionConfig(
            audioldm2_path=args.audioldm2_path,
            sd_root=args.sd_root,
            device=args.device,
            dtype=torch.float16,
            enable_cpu_offload=True
        )
        
        # Create loader (will auto-resolve paths based on current directory)
        loader = DiffusionModelLoader(config, base_dir=Path.cwd())
        
        # Load VAEs
        try:
            sd_vae, audio_vae = loader.load_vaes()
            print("✅ VAEs loaded successfully")
        except Exception as e:
            print(f"❌ Error loading VAEs: {e}")
            print("Please check your model paths:")
            print(f"  AudioLDM2: {config.audioldm2_path}")
            print(f"  SD root: {config.sd_root}")
            raise
    
    return sd_vae, audio_vae, loader  # Return loader for cleanup later


def main():
    args = parse_args()
    
    # Create save directory
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Setup data
    print("Setting up data...")
    train_loader, val_loader = setup_data(args)
    
    # Setup model
    print("Setting up model...")
    model = setup_model(args)
    
    # Setup loss
    print("Setting up loss function...")
    loss_fn = setup_loss(args)
    
    # Setup trainer
    if args.use_diffusion_noise:
        print("Setting up noisy MI trainer...")
        
        # Setup VAEs if needed
        sd_vae, audio_vae, loader = setup_vaes(args)
        
        trainer = NoisyMITrainer(
            model=model,
            loss_fn=loss_fn,
            learning_rate=args.learning_rate,
            device=args.device,
            # Diffusion schedules
            audio_schedule=DiffusionSchedule.from_audio_ldm(args.num_timesteps),
            image_schedule=DiffusionSchedule.from_stable_diffusion(args.num_timesteps),
            num_timesteps=args.num_timesteps,
            # VAEs (optional)
            sd_vae=sd_vae,
            audio_vae=audio_vae,
            # Timestep sampling
            timestep_sampling=args.timestep_sampling,
            min_timestep=args.min_timestep,
            max_timestep=args.max_timestep
        )
    else:
        print("Setting up standard MI trainer...")
        loader = None  # No loader needed for standard training
        trainer = MITrainer(
            model=model,
            loss_fn=loss_fn,
            learning_rate=args.learning_rate,
            device=args.device
        )
    
    # Print training configuration
    print("\n" + "="*50)
    print("Training Configuration:")
    print(f"  Model type: {args.model_type}")
    print(f"  Loss type: {args.loss_type}")
    print(f"  Use latent space: {args.use_latent_space}")
    print(f"  Use timestep: {args.use_timestep}")
    print(f"  Use diffusion noise: {args.use_diffusion_noise}")
    if args.use_latent_space:
        print(f"  AudioLDM2 path: {args.audioldm2_path}")
        print(f"  SD root: {args.sd_root}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Device: {args.device}")
    print("="*50 + "\n")
    
    try:
        # Train
        print("Starting training...")
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            log_interval=args.log_interval,
            save_path=args.save_path
        )
        
        print(f"\nTraining complete! Model saved to {args.save_path}")
        
    finally:
        # Cleanup diffusion models if loaded
        if loader is not None:
            print("\nCleaning up diffusion models...")
            loader.cleanup()


if __name__ == "__main__":
    main()