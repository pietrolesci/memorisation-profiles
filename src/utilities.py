import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import srsly
from datasets import load_dataset
from pyarrow import json
from pyarrow.lib import ArrowInvalid


def convert_json_to_pandas(path: str) -> pd.DataFrame:
    try:
        # read with pyarrow
        df = json.read_json(path).to_pandas()
    except ArrowInvalid:
        # read with HuggingFace datasets
        df: pd.DataFrame = load_dataset("json", data_files={"df": str(path)})["df"].to_pandas()  # type: ignore

    return df


def remove_dir(path: str | Path) -> None:
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)


def remove_file(path: str | Path) -> None:
    path = Path(path)
    path.unlink(missing_ok=True)


def ld_to_dl(ld: list[dict]) -> dict[str, list]:
    return {k: [dic[k] for dic in ld] for k in ld[0]}


def flatten(x) -> list[Any]:
    return [i for j in x for i in j]


def jsonl2parquet(filepath: str | Path, out_dir: str | Path) -> None:
    filepath = Path(filepath)
    assert filepath.name.endswith(".jsonl"), "Not a jsonl file"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fl = srsly.read_jsonl(filepath)
    df = pd.DataFrame({k: flatten(v) for k, v in ld_to_dl(line).items()} for line in fl)  # type: ignore
    df = df.explode(df.columns.tolist())

    df.to_parquet(out_dir / f"{filepath.name.removesuffix('.jsonl')}.parquet", index=False)
