import io
import json
from dataclasses import asdict
from pathlib import Path
from typing import Generator

import datasets
import einops
import numpy as np
import pyarrow as pa
import torch
from datasets import Dataset, Features, Sequence, Value
from huggingface_hub import HfApi
from jaxtyping import Float
from transformer_lens.HookedTransformer import HookedRootModule

from sae_lens.config import DTYPE_MAP, CacheActivationsRunnerConfig
from sae_lens.load_model import load_model
from sae_lens.training.activations_store import ActivationsStore


def _mk_activations_store(
    model: HookedRootModule,
    cfg: CacheActivationsRunnerConfig,
) -> ActivationsStore:
    """
    Internal method used in CacheActivationsRunner. Used to create a cached dataset
    from a ActivationsStore.
    """
    return ActivationsStore(
        model=model,
        dataset=cfg.dataset_path,
        streaming=cfg.streaming,
        hook_name=cfg.hook_name,
        hook_layer=cfg.hook_layer,
        hook_head_index=None,
        context_size=cfg.context_size,
        d_in=cfg.d_in,
        n_batches_in_buffer=cfg.n_batches_in_buffer,
        total_training_tokens=cfg.training_tokens,
        store_batch_size_prompts=cfg.model_batch_size,
        train_batch_size_tokens=-1,
        prepend_bos=cfg.prepend_bos,
        normalize_activations="none",
        device=torch.device("cpu"),  # since we're saving to disk
        dtype=cfg.dtype,
        cached_activations_path=None,
        model_kwargs=cfg.model_kwargs,
        autocast_lm=cfg.autocast_lm,
        dataset_trust_remote_code=cfg.dataset_trust_remote_code,
        seqpos_slice=cfg.seqpos_slice,
    )


class CacheActivationDataset(datasets.ArrowBasedBuilder):
    cfg: CacheActivationsRunnerConfig
    activation_store: ActivationsStore
    # info: datasets.DatasetInfo # By DatasetBuilder

    pa_dtype: pa.DataType
    schema: pa.Schema

    hook_names: list[str]  # while we can only use one hook

    def __init__(
        self,
        cfg: CacheActivationsRunnerConfig,
        activation_store: ActivationsStore,
    ):
        self.cfg = cfg
        self.activation_store = activation_store
        self.hook_names = [cfg.hook_name]

        if cfg.dtype == "float32":
            self.pa_dtype = pa.float32()
        elif cfg.dtype == "float16":
            self.pa_dtype = pa.float16()
        else:
            raise ValueError(f"dtype {cfg.dtype} not supported")

        self.schema = pa.schema(
            [
                pa.field(hook_name, pa.list_(self.pa_dtype, list_size=cfg.d_in))
                for hook_name in self.hook_names
            ]
        )

        features = Features(
            {
                hook_name: Sequence(Value(dtype=cfg.dtype), length=cfg.d_in)
                for hook_name in [cfg.hook_name]
            }
        )
        assert cfg.new_cached_activations_path is not None
        activation_save_path = Path(cfg.new_cached_activations_path)
        activation_save_path.mkdir(parents=True, exist_ok=True)
        assert activation_save_path.is_dir()
        if any(activation_save_path.iterdir()):
            raise ValueError(
                f"Activation save path {activation_save_path} is not empty. Please delete it or specify a different path"
            )
        cache_dir = activation_save_path.parent
        dataset_name = activation_save_path.name
        super().__init__(
            cache_dir=str(cache_dir),
            dataset_name=dataset_name,
            info=datasets.DatasetInfo(features=features),
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager | datasets.StreamingDownloadManager
    ) -> list[datasets.SplitGenerator]:
        return [
            datasets.SplitGenerator(name=str(datasets.Split.TRAIN)),
        ]

    def _generate_tables(self) -> Generator[tuple[int, pa.Table], None, None]:  # type: ignore
        for i in range(self.cfg.n_buffers):
            buffer = self.activation_store.get_buffer(
                self.cfg.n_batches_in_buffer, shuffle=False
            )
            assert buffer.device.type == "cpu"
            buffer = einops.rearrange(
                buffer, "batch hook d_in -> hook batch d_in"
            ).numpy()
            table = pa.Table.from_pydict(
                {
                    hn: self.np2pa_2d(buf, d_in=self.cfg.d_in)
                    for hn, buf in zip(self.hook_names, buffer)
                },
                schema=self.schema,
            )
            yield i, table

    @staticmethod
    def np2pa_2d(data: Float[np.ndarray, "batch d_in"], d_in: int) -> pa.Array:  # type: ignore
        """
        Convert a 2D numpy array to a PyArrow FixedSizeListArray.
        """
        assert data.ndim == 2, "Input array must be 2-dimensional."
        _, d_in_found = data.shape
        if d_in_found != d_in:
            raise RuntimeError(f"d_in {d_in_found} does not match expected d_in {d_in}")
        flat = data.ravel()  # no copy if possible
        pa_data = pa.array(flat)
        return pa.FixedSizeListArray.from_arrays(pa_data, d_in)


class CacheActivationsRunner:
    def __init__(self, cfg: CacheActivationsRunnerConfig):
        self.cfg = cfg
        self.model: HookedRootModule = load_model(
            model_class_name=self.cfg.model_class_name,
            model_name=self.cfg.model_name,
            device=self.cfg.device,
            model_from_pretrained_kwargs=self.cfg.model_from_pretrained_kwargs,
        )
        if self.cfg.compile_llm:
            self.model = torch.compile(self.model, mode=self.cfg.llm_compilation_mode)  # type: ignore
        self.activations_store = _mk_activations_store(self.model, self.cfg)

    def summary(self):
        """
        Print the number of tokens to be cached.
        Print the number of buffers, and the number of tokens per buffer.
        Print the disk space required to store the activations.

        """

        bytes_per_token = (
            self.cfg.d_in * self.cfg.dtype.itemsize
            if isinstance(self.cfg.dtype, torch.dtype)
            else DTYPE_MAP[self.cfg.dtype].itemsize
        )
        total_training_tokens = self.cfg.training_tokens
        total_disk_space_gb = total_training_tokens * bytes_per_token / 10**9

        print(
            f"Activation Cache Runner:\n"
            f"Total training tokens: {total_training_tokens}\n"
            f"Number of buffers: {self.cfg.n_buffers}\n"
            f"Tokens per buffer: {self.cfg.n_tokens_in_buffer}\n"
            f"Disk space required: {total_disk_space_gb:.2f} GB\n"
            f"Configuration:\n"
            f"{self.cfg}"
        )

    @torch.no_grad()
    def run(self) -> Dataset:
        builder = CacheActivationDataset(self.cfg, self.activations_store)
        builder.download_and_prepare()
        dataset = builder.as_dataset(split="train")  # type: ignore
        assert isinstance(dataset, Dataset)

        ### Concatenate shards and push to Huggingface Hub

        if self.cfg.shuffle:
            print("Shuffling...")
            dataset = dataset.shuffle(seed=self.cfg.seed)

        if self.cfg.hf_repo_id:
            print("Pushing to Huggingface Hub...")
            dataset.push_to_hub(
                repo_id=self.cfg.hf_repo_id,
                num_shards=self.cfg.hf_num_shards or self.cfg.n_buffers,
                private=self.cfg.hf_is_private_repo,
                revision=self.cfg.hf_revision,
            )

            meta_io = io.BytesIO()
            meta_contents = json.dumps(
                asdict(self.cfg), indent=2, ensure_ascii=False
            ).encode("utf-8")
            meta_io.write(meta_contents)
            meta_io.seek(0)

            api = HfApi()
            api.upload_file(
                path_or_fileobj=meta_io,
                path_in_repo="cache_activations_runner_cfg.json",
                repo_id=self.cfg.hf_repo_id,
                repo_type="dataset",
                commit_message="Add cache_activations_runner metadata",
            )

        return dataset
