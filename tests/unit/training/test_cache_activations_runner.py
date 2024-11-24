import dataclasses
import math
import os
from pathlib import Path
from typing import Any, Tuple

import datasets
import pytest
import torch
from datasets import Dataset, load_dataset
from sae_lens.cache_activations_runner import CacheActivationsRunner
from sae_lens.config import (
    DTYPE_MAP,
    CacheActivationsRunnerConfig,
    LanguageModelSAERunnerConfig,
)
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.cached_activation_store import CachedActivationsStore
from tqdm import trange
from transformer_lens import HookedTransformer


def _default_cfg(
    tmp_path: Path,
    batch_size: int = 16,
    context_size: int = 8,
    dataset_num_rows: int = 128,
    n_buffers: int = 4,
    **kwargs: Any,
) -> CacheActivationsRunnerConfig:
    d_in = 512
    dtype = "float32"
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    activation_store_kwargs = dict(
        prepend_bos=False,
        normalize_activations="none",
        **kwargs.pop("activation_store_kwargs", {}),
    )

    # Calculate buffer_size_gb to achieve desired n_buffers
    bytes_per_token = d_in * DTYPE_MAP[dtype].itemsize
    tokens_per_buffer = math.ceil(dataset_num_rows * context_size / n_buffers)
    buffer_size_gb = (tokens_per_buffer * bytes_per_token) / 1_000_000_000

    cfg = CacheActivationsRunnerConfig(
        activation_save_path=str(tmp_path),
        hf_dataset_path="chanind/c4-10k-mini-tokenized-16-ctx-gelu-1l-tests",
        model_name="gelu-1l",
        hook_name="blocks.0.hook_mlp_out",
        final_hook_layer=0,
        ### Parameters
        dataset_num_rows=dataset_num_rows,
        model_batch_size=batch_size,
        buffer_size_gb=buffer_size_gb,
        context_size=context_size,
        ###
        d_in=d_in,
        shuffle=False,
        activation_store_kwargs=activation_store_kwargs,
        device=device,
        seed=42,
        dtype=dtype,
        **kwargs,
    )
    assert cfg.n_buffers == n_buffers
    assert cfg.total_training_tokens == dataset_num_rows * context_size
    assert cfg.tokens_in_buffer == cfg.batches_in_buffer * batch_size * context_size
    return cfg


# The way to run this with this command:
# poetry run py.test tests/unit/test_cache_activations_runner.py --profile-svg -s
def test_cache_activations_runner(tmp_path: Path):
    cfg = _default_cfg(tmp_path)
    runner = CacheActivationsRunner(cfg)
    dataset = runner.run()

    assert len(dataset) == cfg.n_buffers * (cfg.tokens_in_buffer // cfg.context_size)
    assert cfg.dataset_num_rows == len(dataset)
    assert dataset.num_columns == 1 and dataset.column_names == [cfg.hook_name]

    features = dataset.features
    for hook_name in [cfg.hook_name]:
        assert isinstance(features[hook_name], datasets.Array2D)
        assert features[hook_name].shape == (cfg.context_size, cfg.d_in)


def test_load_cached_activations(tmp_path: Path):
    cfg = _default_cfg(tmp_path, batch_size=32)
    runner = CacheActivationsRunner(cfg)
    ds = runner.run()
    ds.set_format("torch")
    tokens = ds[cfg.hook_name]
    assert isinstance(tokens, torch.Tensor)
    assert tokens.shape == (cfg.dataset_num_rows, cfg.context_size, cfg.d_in)
    tokens = tokens.flatten(0, 1)[:, None, :]  # (batch, hook [only 1], d_in)

    for batch_size in [8, 16, 24, 32]:
        activations_store = CachedActivationsStore(
            cached_activations_path=tmp_path,
            column_names=[cfg.hook_name],
            batch_size=batch_size,
            dl_kwargs=dict(shuffle=False, drop_last=False),
        )
        assert len(activations_store) == cfg.total_training_tokens // batch_size
        for i in range(0, len(tokens), batch_size):
            batch = activations_store.next_batch()
            torch.testing.assert_close(batch, tokens[i : i + batch_size])


def test_load_cached_activations_from_config_refreshes_when_runs_out(tmp_path: Path):
    cfg = _default_cfg(tmp_path, batch_size=32)
    runner = CacheActivationsRunner(cfg)
    runner.run()

    batch_size = 16
    activations_store = CachedActivationsStore.from_config(cfg, batch_size=batch_size)
    assert len(activations_store) == cfg.total_training_tokens // batch_size
    for _ in range(len(activations_store)):
        batch = activations_store.next_batch(raise_on_epoch_end=True)
        assert batch.shape == (
            batch_size,
            1,
            cfg.d_in,
        )
    with pytest.raises(StopIteration):
        activations_store.next_batch(raise_on_epoch_end=True)

    for _ in range(len(activations_store)):
        batch = activations_store.next_batch(raise_on_epoch_end=True)
        assert batch.shape == (
            batch_size,
            1,
            cfg.d_in,
        )

    # Doesn't raise StopIteration
    batch = activations_store.next_batch(raise_on_epoch_end=False)
    assert batch.shape == (batch_size, 1, cfg.d_in)

def test_activations_store_refreshes_dataset_when_it_runs_out(tmp_path: Path):

    context_size = 8
    n_batches_in_buffer = 4
    store_batch_size = 1
    total_training_steps = 4
    batch_size = 4
    total_training_tokens = total_training_steps * batch_size

    cache_cfg = _default_cfg(tmp_path)
    runner = CacheActivationsRunner(cache_cfg)
    runner.run()

    cfg = LanguageModelSAERunnerConfig(
        cached_activations_path=str(tmp_path),
        use_cached_activations=True,
        model_name="gelu-1l",
        hook_name="blocks.0.hook_mlp_out",
        hook_layer=0,
        d_in=512,
        dataset_path="",
        context_size=context_size,
        is_dataset_tokenized=True,
        prepend_bos=True,
        training_tokens=total_training_tokens // 2,
        train_batch_size_tokens=8,
        n_batches_in_buffer=n_batches_in_buffer,
        store_batch_size_prompts=store_batch_size,
        normalize_activations="none",
        device="cpu",
        seed=42,
        dtype="float16",
    )

    class MockModel:
        def to_tokens(self, *args: Tuple[Any, ...], **kwargs: Any) -> torch.Tensor:
            return torch.ones(context_size)

        @property
        def W_E(self) -> torch.Tensor:
            return torch.ones(16, 16)

        @property
        def cfg(self) -> LanguageModelSAERunnerConfig:
            return cfg

    dataset = Dataset.from_list(
        [
            {"text": "hello world1"},
        ]
        * 64
    )

    model = MockModel()
    activations_store = ActivationsStore.from_config(
        model,  # type: ignore
        cfg,
        override_dataset=dataset,
    )
    for _ in range(16):
        _ = activations_store.get_batch_tokens(batch_size, raise_at_epoch_end=True)

    # assert a stop iteration is raised when we do one more get_batch_tokens

    pytest.raises(
        StopIteration,
        activations_store.get_batch_tokens,
        batch_size,
        raise_at_epoch_end=True,
    )

    # no errors are ever raised if we do not ask for raise_at_epoch_end
    for _ in range(32):
        _ = activations_store.get_batch_tokens(batch_size, raise_at_epoch_end=False)


def test_compare_cached_activations_end_to_end_with_ground_truth(tmp_path: Path):
    """
    Creates activations using CacheActivationsRunner and compares them with ground truth
    model.run_with_cache
    """

    torch.manual_seed(42)
    cfg = _default_cfg(tmp_path)
    runner = CacheActivationsRunner(cfg)
    activation_dataset = runner.run()
    activation_dataset.set_format("torch")
    dataset_acts: torch.Tensor = activation_dataset[cfg.hook_name]  # type: ignore

    model = HookedTransformer.from_pretrained(cfg.model_name, device=cfg.device)
    token_dataset: Dataset = load_dataset(cfg.hf_dataset_path, split=f"train[:{cfg.dataset_num_rows}]")  # type: ignore
    token_dataset.set_format("torch", device=cfg.device)

    ground_truth_acts = []
    for i in trange(0, cfg.dataset_num_rows, cfg.model_batch_size):
        tokens = token_dataset[i : i + cfg.model_batch_size]["input_ids"][
            :, : cfg.context_size
        ]
        _, layerwise_activations = model.run_with_cache(
            tokens,
            names_filter=[cfg.hook_name],
            stop_at_layer=cfg.final_hook_layer + 1,
        )
        acts = layerwise_activations[cfg.hook_name]
        ground_truth_acts.append(acts)

    ground_truth_acts = torch.cat(ground_truth_acts, dim=0).cpu()

    assert torch.allclose(ground_truth_acts, dataset_acts, rtol=1e-3, atol=5e-2)


def test_load_activations_store_with_nonexistent_dataset(tmp_path: Path):
    cfg = _default_cfg(tmp_path)

    # Attempt to load from a non-existent dataset
    with pytest.raises(
        FileNotFoundError,
        match="is neither a `Dataset` directory nor a `DatasetDict` directory.",
    ):
        CachedActivationsStore.from_config(cfg, batch_size=1)


def test_cache_activations_runner_with_nonempty_directory(tmp_path: Path):
    # Create a file to make the directory non-empty
    with open(tmp_path / "some_file.txt", "w") as f:
        f.write("test")

    with pytest.raises(
        Exception, match="is not empty. Please delete it or specify a different path."
    ):
        cfg = _default_cfg(tmp_path)
        runner = CacheActivationsRunner(cfg)
        runner.run()


def test_cache_activations_runner_with_incorrect_d_in(tmp_path: Path):
    correct_cfg = _default_cfg(tmp_path)

    # d_in different from hook
    wrong_d_in_cfg = CacheActivationsRunnerConfig(
        **dataclasses.asdict(correct_cfg),
    )
    wrong_d_in_cfg.d_in = 513

    runner = CacheActivationsRunner(wrong_d_in_cfg)
    with pytest.raises(
        RuntimeError,
        match=r"The expanded size of the tensor \(513\) must match the existing size \(512\) at non-singleton dimension 2.",
    ):
        runner.run()


def test_cache_activations_runner_with_valid_seqpos(tmp_path: Path):
    cfg = _default_cfg(
        tmp_path,
        batch_size=1,
        context_size=16,
        n_buffers=3,
        dataset_num_rows=12,
        activation_store_kwargs=dict(seqpos_slice=(3, -3)),
    )
    runner = CacheActivationsRunner(cfg)

    activation_dataset = runner.run()
    activation_dataset.set_format("torch", device=cfg.device)
    dataset_acts: torch.Tensor = activation_dataset[cfg.hook_name]  # type: ignore

    assert os.path.exists(tmp_path)

    # assert that there are n_buffer files in the directory.
    buffer_files = [
        f
        for f in os.listdir(tmp_path)
        if f.startswith("data-") and f.endswith(".arrow")
    ]
    assert len(buffer_files) == cfg.n_buffers

    for act in dataset_acts:
        # should be 16 - 3 - 3 = 10
        assert act.shape == (10, cfg.d_in)
