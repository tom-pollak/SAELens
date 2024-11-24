from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import einops
import numpy as np
import torch
from datasets import Dataset
from jaxtyping import Float
from sae_lens.config import CacheActivationsRunnerConfig
from tqdm import tqdm


class CachedActivationsStore:
    """
    Given a path to a cached activations dataset, create a dataloader that yields batches of shape (batch_size, hook, seq_len, d_model)

    Args:
        cached_activations_path: Path to a cached activations dataset
        column_names: List of column names to include in the dataloader. These will be the hook names of the activations.
        batch_size: Number of sequences per batch
        dl_kwargs: Keyword arguments for the dataloader
    """

    @classmethod
    def from_config(
        cls,
        cfg: CacheActivationsRunnerConfig,
        batch_size: int,
    ) -> "CachedActivationsStore":
        if cfg.activation_save_path is None:
            raise ValueError(
                "You must specify a new_cached_activations_path in your config."
            )

        return cls(
            cached_activations_path=Path(cfg.activation_save_path),
            column_names=[cfg.hook_name],
            batch_size=batch_size,
        )

    def __init__(
        self,
        cached_activations_path: Path,
        column_names: list[str],
        batch_size: int,
        dl_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.cached_activation_path = cached_activations_path
        assert (
            cached_activations_path.exists()
        ), f"Cache directory {cached_activations_path} does not exist."

        self.column_names = column_names
        self.batch_size = batch_size
        self.dl_kwargs = {} if dl_kwargs is None else dl_kwargs
        self.estimated_norm_scaling_factor = 1.0  # updated by SAETrainer

        self.ds = Dataset.load_from_disk(cached_activations_path)
        self.ds.set_format(type="torch")

        feats = self.ds.features
        assert feats is not None
        self.context_size, self.d_in = feats[list(feats.keys())[0]].shape

        self.dl = self._mk_cached_dl(self.dl_kwargs)
        self.reset_input_dataset()

    def _cached_dl_collate_fn(
        self,
        batch: list[dict[str, Float[torch.Tensor, "seq_len d_model"]]],
    ) -> torch.Tensor:
        """
        Takes a batch of a rows from dataset with columns of different features from the same input sequence

        batch = [{hook_name: [seq_len, d_model]}]

        Transforms into a single batch with shape: (hook, seq_len, d_model)
        """
        acts = torch.stack(
            [
                torch.stack([d[col_name] for col_name in self.column_names])
                for d in batch
            ]
        )
        return einops.rearrange(
            acts,
            "batch hook seq_len d_model -> (batch seq_len) hook d_model",
        )

    def _mk_cached_dl(
        self,
        dl_kwargs: dict[str, Any],
    ) -> torch.utils.data.DataLoader[Float[torch.Tensor, "batch hook d_model"]]:
        assert (
            self.batch_size % self.context_size == 0
        ), "batch_size must be divisible by seq_len"
        dl_batch_size = self.batch_size // self.context_size

        default_dl_kwargs: dict[str, Any] = {
            "num_workers": min(8, os.cpu_count() or 1),
            "prefetch_factor": 2,
            "persistent_workers": False,
            "pin_memory": True,
            "shuffle": True,
            "drop_last": True,
        }
        default_dl_kwargs.update(dl_kwargs)

        return torch.utils.data.DataLoader(
            self.ds,  # type: ignore
            batch_size=dl_batch_size,
            collate_fn=self._cached_dl_collate_fn,
            **default_dl_kwargs,
        )

    # █████████████████████████████████  Common fns  █████████████████████████████████

    def reset_input_dataset(self):
        self.dl_it = iter(self.dl)

    def next_batch(
        self, raise_on_epoch_end: bool = False
    ) -> Float[torch.Tensor, "batch hook d_model"]:
        try:
            return next(self.dl_it)
        except StopIteration:
            if raise_on_epoch_end:
                raise StopIteration("Ran out of cached activations.")
            else:
                self.reset_input_dataset()
                return next(self.dl_it)

    def __len__(self):
        return len(self.ds)

    # ████████████████████████████  ActivationStore fns  █████████████████████████████

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
