import io
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import einops
import torch
from datasets import (
    Dataset,
    DatasetInfo,
    Features,
    Sequence,
    Value,
    disable_caching,
    enable_caching,
)
from datasets.arrow_dataset import (
    InMemoryTable,
)
from datasets.arrow_writer import OptimizedTypedSequence
from datasets.fingerprint import generate_fingerprint
from huggingface_hub import HfApi
from jaxtyping import Float
from sae_lens.config import DTYPE_MAP, CacheActivationsRunnerConfig
from sae_lens.load_model import load_model
from sae_lens.training.activations_store import ActivationsStore
from tqdm import tqdm


def fast_from_dict(mapping: dict[str, Any], features: Features) -> Dataset:
    """Like datasets.Dataset.from_dict but skips fingerprint generation"""
    arrow_typed_mapping = {}
    for col, data in mapping.items():
        data = OptimizedTypedSequence(
            features.encode_column(data, col),
            type=features[col],
            col=col,
        )
        print(type(data))
        arrow_typed_mapping[col] = data

    pa_table = InMemoryTable.from_pydict(mapping=arrow_typed_mapping)

    info = DatasetInfo(features=Features({
        col: data.get_inferred_type()
        for col, data in arrow_typed_mapping.items()
    }))
    return Dataset(pa_table, info=info, fingerprint="dummy")

class CacheActivationsRunner:
    def __init__(self, cfg: CacheActivationsRunnerConfig):
        self.cfg = cfg
        self.model = load_model(
            model_class_name=self.cfg.model_class_name,
            model_name=self.cfg.model_name,
            device=self.cfg.device,
            model_from_pretrained_kwargs=self.cfg.model_from_pretrained_kwargs,
        )
        self.activations_store = ActivationsStore._from_save_activations(
            self.model,
            self.cfg,
        )
        self.context_size = self._get_sliced_context_size(
            self.cfg.context_size,
            self.cfg.activation_store_kwargs.get("seqpos_slice", None),
        )
        self.features = Features(
            {
                hook_name: Sequence(feature=Value(dtype=self.cfg.dtype), length=self.cfg.d_in)
                for hook_name in [self.cfg.hook_name]
            }
        )

    def __str__(self):
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
        total_training_tokens = self.cfg.dataset_num_rows * self.context_size
        total_disk_space_gb = total_training_tokens * bytes_per_token / 10**9

        return (
            f"Activation Cache Runner:\n"
            f"Total training tokens: {total_training_tokens}\n"
            f"Number of buffers: {self.cfg.n_buffers}\n"
            f"Tokens per buffer: {self.cfg.tokens_in_buffer}\n"
            f"Disk space required: {total_disk_space_gb:.2f} GB\n"
            f"Configuration:\n"
            f"{self.cfg}"
        )

    @staticmethod
    def _consolidate_shards(
        source_dir: Path, output_dir: Path, copy_files: bool = True
    ) -> Dataset:
        """Consolidate sharded datasets into a single directory without rewriting data.

        Each of the shards must be of the same format, aka the full dataset must be able to
        be recreated like so:

        ```
        ds = concatenate_datasets(
            [Dataset.load_from_disk(str(shard_dir)) for shard_dir in sorted(source_dir.iterdir())]
        )

        ```

        Sharded dataset format:
        ```
        source_dir/
            shard_00000/
                dataset_info.json
                state.json
                data-00000-of-00002.arrow
                data-00001-of-00002.arrow
            shard_00001/
                dataset_info.json
                state.json
                data-00000-of-00001.arrow
        ```

        And flattens them into the format:

        ```
        output_dir/
            dataset_info.json
            state.json
            data-00000-of-00003.arrow
            data-00001-of-00003.arrow
            data-00002-of-00003.arrow
        ```

        allowing the dataset to be loaded like so:

        ```
        ds = datasets.load_from_disk(output_dir)
        ```

        Args:
            source_dir: Directory containing the sharded datasets
            output_dir: Directory to consolidate the shards into
            copy_files: If True, copy files; if False, move them and delete source_dir
        """
        first_shard_dir_name = "shard_00000"  # shard_{i:05d}

        assert source_dir.exists() and source_dir.is_dir()
        assert (
            output_dir.exists()
            and output_dir.is_dir()
            and not any(p for p in output_dir.iterdir() if not p.name == ".tmp_shards")
        )
        if not (source_dir / first_shard_dir_name).exists():
            raise Exception(f"No shards in {source_dir} exist!")

        transfer_fn = shutil.copy2 if copy_files else shutil.move

        # Move dataset_info.json from any shard (all the same)
        transfer_fn(
            source_dir / first_shard_dir_name / "dataset_info.json",
            output_dir / "dataset_info.json",
        )

        arrow_files = []
        file_count = 0

        for shard_dir in sorted(source_dir.iterdir()):
            if not shard_dir.name.startswith("shard_"):
                continue

            # state.json contains arrow filenames
            state = json.loads((shard_dir / "state.json").read_text())

            for data_file in state["_data_files"]:
                src = shard_dir / data_file["filename"]
                new_name = f"data-{file_count:05d}-of-{len(list(source_dir.iterdir())):05d}.arrow"
                dst = output_dir / new_name
                transfer_fn(src, dst)
                arrow_files.append({"filename": new_name})
                file_count += 1

        new_state = {
            "_data_files": arrow_files,
            "_fingerprint": None,  # temporary
            "_format_columns": None,
            "_format_kwargs": {},
            "_format_type": None,
            "_output_all_columns": False,
            "_split": None,
        }

        # fingerprint is generated from dataset.__getstate__ (not includeing _fingerprint)
        with open(output_dir / "state.json", "w") as f:
            json.dump(new_state, f, indent=2)

        ds = Dataset.load_from_disk(str(output_dir))
        fingerprint = generate_fingerprint(ds)
        del ds

        with open(output_dir / "state.json", "r+") as f:
            state = json.loads(f.read())
            state["_fingerprint"] = fingerprint
            f.seek(0)
            json.dump(state, f, indent=2)
            f.truncate()

        if not copy_files:  # cleanup source dir
            shutil.rmtree(source_dir)

        return Dataset.load_from_disk(output_dir)

    @torch.no_grad()
    def run(self) -> Dataset:
        print(str(self))
        activation_save_path = self.cfg.activation_save_path
        assert activation_save_path is not None

        ### Paths setup
        final_cached_activation_path = Path(activation_save_path)
        final_cached_activation_path.mkdir(exist_ok=True, parents=True)
        if any(final_cached_activation_path.iterdir()):
            raise Exception(
                f"Activations directory ({final_cached_activation_path}) is not empty. Please delete it or specify a different path. Exiting the script to prevent accidental deletion of files."
            )

        tmp_cached_activation_path = final_cached_activation_path / ".tmp_shards/"
        tmp_cached_activation_path.mkdir(exist_ok=False, parents=False)

        ### Create temporary sharded datasets

        print(f"Started caching activations for {self.cfg.hf_dataset_path}")
        disable_caching()
        for i in tqdm(range(self.cfg.n_buffers), desc="Caching activations"):
            try:
                buffer = self.activations_store.get_buffer(
                    self.cfg.batches_in_buffer, shuffle=False
                )
                shard = self._create_shard(buffer)
                shard.save_to_disk(
                    f"{tmp_cached_activation_path}/shard_{i:05d}", num_shards=1
                )
                del buffer, shard

            except StopIteration:
                print(
                    f"Warning: Ran out of samples while filling the buffer at batch {i} before reaching {self.cfg.n_buffers} batches."
                )
                break
        enable_caching()

        ### Concatenate shards and push to Huggingface Hub

        dataset = self._consolidate_shards(
            tmp_cached_activation_path, final_cached_activation_path, copy_files=False
        )

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

            # Tacks on username to the repo id
            user_repo_id = api.create_repo(
                self.cfg.hf_repo_id,
                private=self.cfg.hf_is_private_repo,
                repo_type="dataset",
                exist_ok=True,  # should exist already
            ).repo_id

            api.upload_file(
                path_or_fileobj=meta_io,
                path_in_repo="cache_activations_runner_cfg.json",
                repo_id=user_repo_id,
                revision=self.cfg.hf_revision,
                repo_type="dataset",
                commit_message="Add cache_activations_runner metadata",
            )

        return dataset

    def _create_shard(
        self,
        buffer: Float[torch.Tensor, "batch layer d_in"],
    ) -> Dataset:
        assert buffer.device.type == "cpu"

        hook_names = [self.cfg.hook_name]
        buffer = einops.rearrange(buffer, "batch layer d_in -> layer batch d_in")
        shard = fast_from_dict(
            {hook_name: act for hook_name, act in zip(hook_names, buffer.unbind(dim=0))},
            features=self.features,
        )
        return shard

    @staticmethod
    def _get_sliced_context_size(
        context_size: int, seqpos_slice: tuple[int, int] | None
    ) -> int:
        if seqpos_slice:
            context_size = len(range(context_size)[slice(*seqpos_slice)])
        return context_size
