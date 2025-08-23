# models/mi_models.py
import torch
import torch.nn as nn
from models.unified_encoders import UnifiedEncoder

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal timestep embeddings."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, timesteps):
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=timesteps.device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class TimestepMIEstimator(nn.Module):
    """Estimates MI score given features and timestep."""
    def __init__(self, feature_dim=256, hidden_dim=512, time_embed_dim=128):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )
        
        self.network = nn.Sequential(
            nn.Linear(feature_dim * 2 + time_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, features_1, features_2, t):
        t_emb = self.time_embed(t.float())
        combined = torch.cat([features_1, features_2, t_emb], dim=-1)
        return self.network(combined).squeeze(-1)


class UnifiedMIModel(nn.Module):
    """Unified MI model that can work with different encoder configurations."""
    
    def __init__(self, model_type='sd_audioldm', feature_dim=256, hidden_dim=512):
        super().__init__()
        self.model_type = model_type
        
        # Create encoders based on model type
        if model_type == 'sd_audioldm':
            self.encoder_1 = UnifiedEncoder.create_audioldm_encoder()  # audio
            self.encoder_2 = UnifiedEncoder.create_sd_image_encoder()  # image
        elif model_type == 'ddpm_night2day':
            self.encoder_1 = UnifiedEncoder.create_ddpm_image_encoder()  # night
            self.encoder_2 = UnifiedEncoder.create_ddpm_image_encoder()  # day
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # MI estimator is shared
        self.mi_estimator = TimestepMIEstimator(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim
        )
    
    def forward(self, input_1, input_2, t):
        """
        Forward pass through the model.
        For sd_audioldm: input_1 is audio_latent, input_2 is image_latent
        For ddpm_night2day: input_1 is night_img, input_2 is day_img
        """
        features_1 = self.encoder_1(input_1)
        features_2 = self.encoder_2(input_2)
        return self.mi_estimator(features_1, features_2, t)
    
    @classmethod
    def create_sd_audioldm_model(cls):
        """Factory method for SD+AudioLDM model."""
        return cls(model_type='sd_audioldm')
    
    @classmethod
    def create_ddpm_night2day_model(cls):
        """Factory method for DDPM Night2Day model."""
        return cls(model_type='ddpm_night2day')