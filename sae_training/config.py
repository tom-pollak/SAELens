
from dataclasses import dataclass

import torch

import wandb


@dataclass
class LanguageModelSAERunnerConfig:
    """
    Configuration for training a sparse autoencoder on a language model.
    """

    # Data Generating Function (Model + Training Distibuion)
    model_name: str = "gelu-2l"
    hook_point: str = "blocks.0.hook_mlp_out"
    hook_point_layer: int = 0
    dataset_path: str = "NeelNanda/c4-tokenized-2b"
    is_dataset_tokenized: bool = True
    
    # SAE Parameters
    d_in: int = 512
    expansion_factor: int = 4
    
    # Training Parameters
    l1_coefficient: float = 1e-3
    lr: float = 3e-4
    train_batch_size: int = 4096
    context_size: int = 128
    
    # Resampling protocol args
    feature_sampling_method: str = "l2" # None, l2, or anthropic
    feature_sampling_window: int = 100
    feature_reinit_scale: float = 0.2
    dead_feature_threshold: float = 1e-8
    
    
    # Activation Store Parameters
    n_batches_in_buffer: int = 20
    total_training_tokens: int = 2_000_000
    store_batch_size: int = 4096
    
    # WANDB
    log_to_wandb: bool = True
    wandb_project: str = "mats_sae_training_language_model"
    wandb_entity: str = None
    wandb_log_frequency: int = 5
    
    # Misc
    device: str = "cpu"
    seed: int = 42
    n_checkpoints: int = 0
    checkpoint_path: str = "checkpoints"
    dtype: torch.dtype = torch.float32
    
    def __post_init__(self):
        self.d_sae = self.d_in * self.expansion_factor
        self.tokens_per_buffer = self.train_batch_size * self.context_size * self.n_batches_in_buffer
        
        if self.feature_sampling_method not in [None, "l2", "anthropic"]:
            raise ValueError(f"feature_sampling_method must be None, l2, or anthropic. Got {self.feature_sampling_method}")
        
        unique_id = wandb.util.generate_id()   
        self.checkpoint_path = f"{self.checkpoint_path}/{unique_id}"