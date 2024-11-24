from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import torch
from datasets import Dataset, Sequence, Value
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
    ds: Dataset
    dl: torch.utils.data.DataLoader[Float[torch.Tensor, "batch hook d_model"]]
    dl_it: Iterator[Float[torch.Tensor, "batch hook d_model"]]
    column_names: list[str]
    batch_size: int
    context_size: int
    estimated_norm_scaling_factor: float
    d_in: int
    dl_kwargs: dict[str, Any]
    _default_dl_kwargs: dict[str, Any] = {
        "num_workers": min(8, os.cpu_count() or 1),
        "prefetch_factor": 4,
        "persistent_workers": False,
        "pin_memory": False,
        "shuffle": True,
        "drop_last": True,
    }

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
        self.ds.set_format(type="torch", columns=self.column_names)

        feats = self.ds.features
        assert feats is not None
        _first_feat = next(iter(feats.values()))
        self.context_size, self.d_in = _first_feat.shape
        self.dtype = _first_feat.dtype
        for feat in feats.values():
            assert feat.shape == (self.context_size, self.d_in), "All features must have the same (context_size, d_in)"
            assert feat.dtype == self.dtype, "All features must have the same dtype"

        self.dl = self._mk_cached_dl(self.dl_kwargs)
        self.reset_input_dataset()

    def _flatten_ds(self, ds: Dataset) -> Dataset:
        """
        I think this has a problem at the moment, each row is a 2D array (context_size, d_in),
        each batch is a 3D array (batch_size, context_size, d_in)

        Even shuffling a batch to (batch_size * context_size, d_in),
        this is not a perfectly shuffled since it will include the full context of each sequence.

        """
        def _expand_rows(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
            return {
                name: [row for arr in arrays for row in arr]
                for name, arrays in batch.items()
            }

        columns = ds.column_names
        ds = ds.map(
            _expand_rows,
            # remove_columns=columns,
            batched=True,
        )
        for column in columns:
            ds = ds.cast_column(column, Sequence(feature=Value(dtype=self.dtype), length=self.d_in))
        return ds

    def _cached_dl_collate_fn(
        self,
        batch: list[dict[str, Float[torch.Tensor, "seq_len d_model"]]],
    ) -> torch.Tensor:
        """
        Takes a batch of a rows from dataset with columns of different features from the same input sequence

        batch = [{ hook_name: tensor(d_model) }]

        Transforms into a single batch with shape: (batch, hook, d_model)
        """
        acts = torch.stack([
            torch.stack(list(d.values())) for d in batch
        ])
        return einops.rearrange(
            acts,
            "batch hook seq_len d_model -> (batch seq_len) hook d_model",
        )
        return acts

    def _mk_cached_dl(
        self,
        dl_kwargs: dict[str, Any],
    ) -> torch.utils.data.DataLoader[torch.Tensor]:
        flattened_ds = self._flatten_ds(self.ds)
        assert len(flattened_ds) > self.batch_size, "Flattened dataset must have more rows than batch_size"
        return torch.utils.data.DataLoader(
            flattened_ds,  # type: ignore
            batch_size=self.batch_size,
            collate_fn=self._cached_dl_collate_fn,
            **{**self._default_dl_kwargs, **(dl_kwargs or {})}
        )

    # █████████████████████████████████  Common fns  █████████████████████████████████

    def reset_input_dataset(self):
        if hasattr(self, "dl_it"):
            del self.dl_it
        self.dl_it = iter(self.dl)

    def next_batch(
        self, raise_on_epoch_end: bool = False
    ) -> Float[torch.Tensor, "batch hook d_model"]:
        try:
            return next(self.dl_it)
        except StopIteration:
            self.reset_input_dataset()
            if raise_on_epoch_end:
                raise StopIteration("Ran out of cached activations, refreshing dataset.")
            else:
                return next(self.dl_it)

    def __len__(self):
        return len(self.dl)

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
