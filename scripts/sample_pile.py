from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk
from sklearn.utils import check_random_state
from tqdm.auto import tqdm

from pythia.utils.mmap_dataset import MMapIndexedDataset

# fixed info from Pythia configuration
BATCH_SIZE = 1024
NUM_BATCHES_PER_COHORT = 1000
NUM_SEQ_PER_COHORT = BATCH_SIZE * NUM_BATCHES_PER_COHORT

# hard-coded, unfortunately
TRAIN_SEED = 1994
VALIDATION_SEED = 1234


def sample_data_from_pile(
    pile: MMapIndexedDataset, n_batches_cohort: int, n_seq_batch: int, initial_seed: int
) -> pd.DataFrame:
    """Hierarchically sample `n_batches_cohort` batches per cohort. Then, once we have the batches
    sample `n_seq_batch` sequences per batch.
    """
    batch_ids = np.array(range(len(pile) // BATCH_SIZE))
    cohorts = set((batch_ids // NUM_BATCHES_PER_COHORT).tolist())
    seed = initial_seed
    rng = check_random_state(seed)

    samples = []
    for cohort_id in tqdm(cohorts):
        batch_ids = (
            rng.choice(range(NUM_BATCHES_PER_COHORT), size=n_batches_cohort, replace=False)
            + NUM_BATCHES_PER_COHORT * cohort_id
        ).tolist()

        for batch_id in batch_ids:
            seq_ids = (rng.choice(range(BATCH_SIZE), size=n_seq_batch, replace=False) + BATCH_SIZE * batch_id).tolist()

            samples.append({"cohort": cohort_id, "batch_idx": batch_id, "seq_idx": seq_ids})

            seed = rng.randint(1_000_000)
            rng = check_random_state(seed)

    # create dataframe
    df = pd.DataFrame(samples).explode("seq_idx")
    df = df.sort_values("seq_idx")
    df["input_ids"] = df["seq_idx"].map(lambda x: pile[x].tolist())  # type: ignore

    return df


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--n_batches_cohort", type=int, default=10)
    parser.add_argument("--n_seq_batch", type=int, default=10)
    parser.add_argument("--n_validation", type=int, default=2000)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    splits = {}

    # Sample training data from the pile
    pile = MMapIndexedDataset(str(data_path / "pile-deduped-train" / "document"), skip_warmup=True)
    train_df = sample_data_from_pile(
        pile, n_batches_cohort=args.n_batches_cohort, n_seq_batch=args.n_seq_batch, initial_seed=TRAIN_SEED
    )

    splits["train"] = Dataset.from_pandas(train_df[["input_ids", "seq_idx"]], preserve_index=False)
    # df.to_parquet(data_path / "pile-deduped-train-sample.parquet", index=False)

    # Sample validation data
    validation_path = data_path / "pile-deduped-validation" / "packed"
    if validation_path.exists():
        dataset: Dataset = load_from_disk(str(validation_path))  # type: ignore

        rng = check_random_state(VALIDATION_SEED)
        subset_ids = rng.choice(range(len(dataset)), size=args.n_validation, replace=False)
        validation_subset = dataset.select(subset_ids.tolist())

        splits["validation"] = validation_subset
        # df.to_parquet(data_path / "pile-deduped-validation-sample.parquet", index=False)

    # Save
    DatasetDict(splits).save_to_disk(data_path / "pile-deduped-subset")
