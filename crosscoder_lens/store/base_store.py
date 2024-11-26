
from abc import ABC, abstractmethod

import numpy as np
import torch
from jaxtyping import Float
from tqdm import tqdm


class BaseStore(ABC):
    context_size: int
    d_in: int
    dtype: torch.dtype
    estimated_norm_scaling_factor: float = 1.0  # updated by SAETrainer

    @abstractmethod
    def reset_input_dataset(self):
        ...

    @abstractmethod
    def next_batch(
        self, raise_on_epoch_end: bool = False
    ) -> Float[torch.Tensor, "batch hook d_model"]:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @torch.no_grad()
    def estimate_norm_scaling_factor(self, n_batches_for_norm_estimate: int = int(1e3)):
        norms_per_batch = []
        for _ in tqdm(
            range(n_batches_for_norm_estimate), desc="Estimating norm scaling factor"
        ):
            acts = self.next_batch()
            norms_per_batch.append(acts.norm(dim=-1).mean().item())
        mean_norm = np.mean(norms_per_batch)
        scaling_factor = np.sqrt(self.d_in) / mean_norm
        self.reset_input_dataset()
        return scaling_factor

    def apply_norm_scaling_factor(self, activations: torch.Tensor) -> torch.Tensor:
        return activations * self.estimated_norm_scaling_factor

    def unscale(self, activations: torch.Tensor) -> torch.Tensor:
        return activations / self.estimated_norm_scaling_factor

    def get_norm_scaling_factor(self, activations: torch.Tensor) -> torch.Tensor:
        return (self.d_in**0.5) / activations.norm(dim=-1).mean()
