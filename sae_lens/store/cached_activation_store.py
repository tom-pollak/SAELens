from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator, Optional

import torch
from datasets import Dataset
from jaxtyping import Float
from sae_lens.config import CacheActivationsRunnerConfig
from sae_lens.store.base_store import BaseStore


class CachedActivationsStore(BaseStore):
    """
    Given a path to a cached activations dataset, create a dataloader that yields batches of shape (batch_size, hook, seq_len, d_model)

    Args:
        activation_save_path: Path to a cached activations dataset
        column_names: List of column names to include in the dataloader. These will be the hook names of the activations.
        batch_size: Number of sequences per batch
        dl_kwargs: Keyword arguments for the dataloader
    """

    # Dataset and Dataloader
    ds: Dataset
    dl: torch.utils.data.DataLoader[Float[torch.Tensor, "batch hook d_model"]]
    dl_it: Iterator[Float[torch.Tensor, "batch hook d_model"]]

    # Args for __init__
    activation_save_path: Path
    column_names: list[str]
    batch_size: int
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
    ) -> CachedActivationsStore:
        if cfg.new_cached_activations_path is None:
            raise ValueError(
                "You must specify a new_cached_activations_path in your config."
            )
        activation_save_path = Path(cfg.new_cached_activations_path)
        assert (
            activation_save_path.exists()
        ), f"Cache directory {activation_save_path} does not exist."
        ds = Dataset.load_from_disk(activation_save_path)

        return cls(
            ds=ds,
            column_names=[cfg.hook_name],
            batch_size=batch_size,
            context_size=cfg.context_size,
        )


    def __init__(
        self,
        ds: Dataset,
        column_names: list[str],
        batch_size: int,
        context_size: int,
        dl_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.ds = ds
        self.ds.set_format(type="torch", columns=self.column_names)

        self.column_names = column_names
        self.batch_size = batch_size
        self.context_size = context_size
        self.dl_kwargs = {} if dl_kwargs is None else dl_kwargs
        self.estimated_norm_scaling_factor = 1.0  # updated by SAETrainer

        feats = self.ds.features
        assert feats is not None
        _first_feat = next(iter(feats.values()))
        self.d_in = _first_feat.length
        assert self.d_in > 0, "d_in must be greater than 0"
        self.dtype = _first_feat.dtype
        for feat in feats.values():
            assert feat.length == self.d_in, "All features must have the same length"
            assert feat.dtype == self.dtype, "All features must have the same dtype"

        self.dl = self._mk_cached_dl(self.dl_kwargs)
        self.reset_input_dataset()

    def _cached_dl_collate_fn(
        self,
        batch: list[dict[str, Float[torch.Tensor, "*seq_len d_model"]]],
    ) -> torch.Tensor:
        """
        Takes a batch of a rows from dataset with columns of different features from the same input sequence

        [ { hook: Tensor(d_model) } ]

        Transforms into a single batch with shape: (batch, hook, d_model)
        """
        acts = torch.stack([torch.stack(list(d.values())) for d in batch])
        return acts

    def _mk_cached_dl(
        self,
        dl_kwargs: dict[str, Any],
    ) -> torch.utils.data.DataLoader[torch.Tensor]:
        assert (
            len(self.ds) > self.batch_size
        ), "Flattened dataset must have more rows than batch_size"
        return torch.utils.data.DataLoader(
            self.ds,  # type: ignore
            batch_size=self.batch_size,
            collate_fn=self._cached_dl_collate_fn,
            **{**self._default_dl_kwargs, **(dl_kwargs or {})},
        )

    #### BaseStore fns ####

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
                raise StopIteration(
                    "Ran out of cached activations, refreshing dataset."
                )
            else:
                return next(self.dl_it)

    def __len__(self):
        return len(self.dl)
