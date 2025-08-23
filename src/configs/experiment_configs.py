from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class DataConfig:
    """Data configuration."""
    batch_size: int = 32
    num_workers: int = 4
    real_fake_ratio: float = 0.5
    
    # For AVE dataset
    ave_data_dir: Optional[str] = None
    fake_data_dir: Optional[str] = None
    audio_duration: float = 10.0
    audio_sr: int = 16000
    image_size: Tuple[int, int] = (512, 512)
    
    # For Night2Day dataset
    data_path: Optional[str] = None
    fake_night_dir: Optional[str] = None
    fake_day_dir: Optional[str] = None
    resolution: int = 64

@dataclass
class ModelConfig:
    """Model configuration."""
    model_type: str = 'sd_audioldm'  # or 'ddpm_night2day'
    feature_dim: int = 256
    hidden_dim: int = 512
    time_embed_dim: int = 128

@dataclass
class TrainingConfig:
    """Training configuration."""
    learning_rate: float = 1e-4
    num_epochs: int = 10
    patience: int = 3
    max_grad_norm: float = 5.0
    loss_type: str = 'disc'
    device: str = 'cuda'
    save_path: str = 'checkpoints'
    
    # Diffusion schedules
    num_timesteps: int = 1000
    beta_start_1: float = 0.0015  # AudioLDM or DDPM
    beta_end_1: float = 0.0195
    beta_start_2: float = 0.00085  # SD or DDPM
    beta_end_2: float = 0.012
    
    # VAE settings (for latent space)
    vae_scale_factor: float = 0.18215

@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    
    @classmethod
    def sd_audioldm_config(cls):
        """Factory method for SD+AudioLDM experiment."""
        return cls(
            name='sd_audioldm_ave',
            data=DataConfig(
                ave_data_dir="../data/AVE_Dataset/ave_processed",
                fake_data_dir="unconditional_samples_50",
                batch_size=32,
                audio_duration=10.0,
                audio_sr=16000,
                image_size=(512, 512)
            ),
            model=ModelConfig(
                model_type='sd_audioldm'
            ),
            training=TrainingConfig(
                learning_rate=1e-4,
                num_epochs=10,
                beta_start_1=0.0015,  # AudioLDM
                beta_end_1=0.0195,
                beta_start_2=0.00085,  # SD
                beta_end_2=0.012
            )
        )
    
    @classmethod
    def ddpm_night2day_config(cls):
        """Factory method for DDPM Night2Day experiment."""
        return cls(
            name='ddpm_night2day',
            data=DataConfig(
                data_path="../data/separated_night_day",
                # val_path="../data/separated_night_day/val",
                fake_night_dir= "../data/samples/night/1_2_4/tr_stp_70000_stp1000/2025_07_21_06_55",
                fake_day_dir= "../data/samples/day/1_2_4/tr_stp_70000_stp1000/2025_07_20_22_40",
                resolution=64,
                batch_size=32
            ),
            model=ModelConfig(
                model_type='ddpm_night2day'
            ),
            training=TrainingConfig(
                learning_rate=1e-4,
                num_epochs=10,
                beta_start_1=0.0001,  # DDPM linear schedule
                beta_end_1=0.02,
                beta_start_2=0.0001,
                beta_end_2=0.02
            )
        )