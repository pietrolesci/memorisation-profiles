# ===========================================================================================
# This scripts creates the validation split. This split is not available online anymore. We
# got access via EleutherAI. We do not publish the data, but only aggregate statistics on the
# data. If you want to replicate our analysis, there are sources where it is possible to
# download the pile unofficially.
#
# To do so,
#   1. Get dataset pile_val.jsonl.zst from
#   >>> magnet:?xt=urn:btih:0d366035664fdf51cfbe9f733953ba325776e667&dn=EleutherAI_ThePile_v1
#
#   2. Decompress it with
#   >>> zstd --decompress pile_val.jsonl.zst
#
#   3. Move it under {data_path}/raw/pile_val.jsonl
# ===========================================================================================
from argparse import ArgumentParser
from collections.abc import Generator
from itertools import count, islice, takewhile
from os import cpu_count
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

# Pythia config
SEQ_LENGTH = 2049


def chunk(stream: Generator, size: int) -> takewhile[list[Any]]:
    return takewhile(bool, (list(islice(stream, size)) for _ in count()))


def stream(dataset: Dataset) -> Generator[Any, Any, None]:
    for seq in dataset:
        yield from seq["input_ids"]  # type: ignore


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/pile-deduped-validation")
    args = parser.parse_args()

    data_path = Path(args.data_path)

    # Load data
    ds: Dataset = load_dataset("json", data_files={"validation": str(data_path / "raw" / "pile_val.jsonl")})[  # type: ignore
        "validation"
    ]

    # Tokenize
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
    ds = ds.map(lambda ex: tok(ex["text"], return_attention_mask=False), batched=True, num_proc=cpu_count())
    ds.save_to_disk(str(data_path / "tokenized"))

    # NOTE: Pack dataset with no eos token: https://github.com/EleutherAI/pythia/issues/123
    # Potentially inefficient but it's a small dataset
    g = stream(ds)
    packed_dataset = list(chunk(g, SEQ_LENGTH))

    # Drop remainder
    df = pd.DataFrame({"input_ids": filter(lambda x: len(x) == SEQ_LENGTH, packed_dataset)})
    df = df.reset_index().assign(seq_idx=lambda _df: -100 - _df["index"]).drop(columns=["index"])

    # Cast to dataset
    dataset = Dataset.from_pandas(df, preserve_index=False)

    # Save
    dataset.save_to_disk(str(data_path / "packed"))
