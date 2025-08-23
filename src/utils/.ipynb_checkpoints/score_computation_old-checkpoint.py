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
        
        # Initialize and load model
        self.model = UnifiedMIModel(self.model_type).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Loaded {self.model_type} model trained with {self.loss_type} loss")
    
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
                log_ratio = score
            else:  # ulsif, kliep, rulsif
                ratio = torch.nn.functional.softplus(score) + 1e-8
                log_ratio = torch.log(ratio)
            
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