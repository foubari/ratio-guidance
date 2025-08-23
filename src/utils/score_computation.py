# utils/score_computation.py - Fixed version
import torch
import torch.autograd as autograd
from pathlib import Path
from models.mi_models import UnifiedMIModel

class ScoreComputer:
    """Computes scores and gradients for guided sampling."""
    
    @staticmethod
    def find_checkpoint(loss_type, model_type, checkpoint_dir='checkpoints'):
        """Auto-detect checkpoint path based on loss type and model type."""
        checkpoint_path = Path(checkpoint_dir) / loss_type / f'{model_type}_{loss_type}_mi_model_best.pt'
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path
    
    def __init__(self, checkpoint_path=None, loss_type=None, model_type=None, device='cuda'):
        """Load trained MI model from checkpoint."""
        self.device = device
        
        # Auto-detect checkpoint if not provided
        if checkpoint_path is None:
            if loss_type is None or model_type is None:
                raise ValueError("Provide either checkpoint_path or (loss_type, model_type)")
            checkpoint_path = self.find_checkpoint(loss_type, model_type)
        
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model_type = checkpoint['model_type']
        self.loss_type = checkpoint['loss_type']
        
        # Load loss hyperparameters if available
        self.loss_hparams = checkpoint.get('loss_hparams', {})
        self.use_exp_w = self.loss_hparams.get('use_exp_w', False)
        self.rulsif_alpha = self.loss_hparams.get('rulsif_alpha', 0.2)
        self.rulsif_link = self.loss_hparams.get('rulsif_link', 'softplus')
        
        # Initialize and load model
        self.model = UnifiedMIModel(self.model_type).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Loaded {self.model_type} model trained with {self.loss_type} loss")
        if self.loss_type == 'rulsif':
            print(f"  RuLSIF params: alpha={self.rulsif_alpha}, link={self.rulsif_link}")
    
    def get_log_ratio_grad(self, input_1, input_2, t, target='both'):
        """
        Compute gradient of log ratio w.r.t. inputs.
        
        Args:
            input_1, input_2: Noisy inputs at timestep t
            t: Timestep
            target: Which gradient to compute ('input_1', 'input_2', or 'both')
        
        Returns:
            grad_1, grad_2: Gradients w.r.t. input_1 and input_2
        """
        # Ensure we're computing gradients
        with torch.enable_grad():
            # Set up inputs with gradient tracking
            input_1 = input_1.detach().requires_grad_(target in ['input_1', 'both'])
            input_2 = input_2.detach().requires_grad_(target in ['input_2', 'both'])
            
            # Forward pass through model
            score = self.model(input_1, input_2, t)
            
            # Convert to log ratio based on loss type
            if self.loss_type in ['disc', 'dv']:
                # For disc/dv, the model directly outputs log(q/r)
                log_ratio = score
                
            elif self.loss_type in ['ulsif', 'kliep']:
                # For uLSIF/KLIEP, model outputs score where softplus(score) ≈ q/r
                if self.use_exp_w:
                    ratio = torch.exp(torch.clamp(score, max=40.0))
                else:
                    ratio = torch.nn.functional.softplus(score) + 1e-8
                log_ratio = torch.log(ratio)
                
            elif self.loss_type == 'rulsif':
                # For RuLSIF, model outputs score where link(score) = r_alpha
                # Need to convert r_alpha to q/r
                
                # 1) Map score to r_alpha using same link as training
                if self.rulsif_link == 'exp':
                    r_alpha = torch.exp(torch.clamp(score, max=40.0))
                elif self.rulsif_link == 'softplus':
                    r_alpha = torch.nn.functional.softplus(score) + 1e-8
                else:  # identity
                    r_alpha = torch.clamp(score, min=1e-8)
                
                # 2) Avoid pole at r_alpha = 1/alpha
                max_r_alpha = (1.0 / self.rulsif_alpha) - 1e-6
                r_alpha = torch.clamp(r_alpha, max=max_r_alpha)
                
                # 3) Convert relative ratio to true ratio: q/r = (1-α)r_α / (1 - α*r_α)
                numerator = (1.0 - self.rulsif_alpha) * r_alpha
                denominator = torch.clamp(1.0 - self.rulsif_alpha * r_alpha, min=1e-8)
                ratio = numerator / denominator
                log_ratio = torch.log(ratio + 1e-8)
                
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")
            
            # Compute gradients
            grad_1, grad_2 = None, None
            
            if target in ['input_1', 'both']:
                grad_1 = autograd.grad(
                    outputs=log_ratio.sum(),
                    inputs=input_1,
                    create_graph=False,
                    retain_graph=(target == 'both')
                )[0]
            
            if target in ['input_2', 'both']:
                grad_2 = autograd.grad(
                    outputs=log_ratio.sum(),
                    inputs=input_2,
                    create_graph=False,
                    retain_graph=False
                )[0]
        
        return grad_1, grad_2