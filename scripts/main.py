import logging
import time
from os import cpu_count
from pathlib import Path

import hydra
import srsly
import torch
from datasets import DatasetDict, concatenate_datasets, load_from_disk
from lightning.fabric import seed_everything
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers.utils.logging import set_verbosity_warning

from src.models import HFCausalLM
from src.utilities import jsonl2parquet, ld_to_dl, remove_dir

SEP_LINE = f"{'=' * 70}"
MODEL_CACHE_DIR = ".model_cache"

log = logging.getLogger("hydra")

set_verbosity_warning()


def collator_fn(batch: list[dict[str, int | list[int]]]) -> dict[str, list[int] | torch.LongTensor]:
    new_batch = ld_to_dl(batch)
    new_batch["input_ids"] = torch.tensor(new_batch["input_ids"], dtype=torch.long)  # type: ignore
    return new_batch  # type: ignore


@hydra.main(version_base=None, config_path="../conf", config_name="conf")
def main(cfg: DictConfig) -> None:
    # =============================
    # Step 1. Prepare configuration
    # =============================
    start_time = time.time()
    OmegaConf.resolve(cfg)
    OmegaConf.save(cfg, "./hparams.yaml")
    log.info(f"\n{OmegaConf.to_yaml(cfg)}\n{SEP_LINE}")

    seed_everything(cfg.seed)
    log.info(f"Seed enabled: {cfg.seed}")

    # ================================
    # Step 2. Load model and tokenizer
    # ================================
    log.info("Loading model")
    model = HFCausalLM(
        f"EleutherAI/pythia-{cfg.model_size}-deduped",
        revision=cfg.revision,
        accelerator=cfg.accelerator,
        precision=cfg.precision,
        tf32_mode=cfg.tf32_mode,
        deterministic=cfg.deterministic,
        cache_dir=cfg.cache_dir,
    )
    model.summary()

    # =================
    # Step 3. Load data
    # =================
    log.info("Loading data")
    data_path = Path(cfg.data_path) / "pile-deduped-subset"
    dataset_dict: DatasetDict = load_from_disk(str(data_path), keep_in_memory=True)  # type: ignore
    dataset = concatenate_datasets(list(dataset_dict.values()))
    data_loader = DataLoader(
        dataset,  # type: ignore
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cpu_count(),  # type: ignore
        pin_memory=True,
        persistent_workers=False,
        collate_fn=collator_fn,
        multiprocessing_context=cfg.multiprocessing_context,
    )
    data_loader = model.init_dataloader(data_loader)
    log.info(f"Data has {len(dataset)} unique sequences and {len(data_loader)} batches")

    # =====================
    # Step 4. Run inference
    # =====================
    log.info("Running inference")
    pbar = tqdm(data_loader, desc="Running Inference")
    write_buffer = []
    FILENAME = f"data-pythia-deduped-{cfg.model_size}-{cfg.revision}"
    seen = 0

    for batch in pbar:
        pbar.set_postfix_str(f"Buffer size: {len(write_buffer)}")

        # Compute statistics
        batch = model.transfer_to_device(batch)
        out = model.compute_statistics(batch["input_ids"], only_sequence_surprisal=cfg.only_sequence_surprisal)
        out["seq_idx"] = batch["seq_idx"]
        write_buffer.append(out)
        seen += 1

        # Write to disk
        if len(write_buffer) == cfg.write_interval or seen == len(pbar):
            pbar.set_description_str("Writing to disk")
            srsly.write_jsonl(f"{FILENAME}.jsonl", [write_buffer], append=True)

            pbar.set_description_str("Running Inference")
            write_buffer = []

    # clean-up
    jsonl2parquet(filepath=f"{FILENAME}.jsonl", out_dir=".")
    remove_dir(cfg.cache_dir)
    log.info(f"Total time: {(time.time() - start_time) // 60} minutes")


if __name__ == "__main__":
    main()
