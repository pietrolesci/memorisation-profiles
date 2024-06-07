from argparse import ArgumentParser
from pathlib import Path

import duckdb as db


def check_runs(run_paths: list[Path]) -> None:
    done = set(int(p.name.split("-")[-1].removesuffix(".parquet").removeprefix("step")) for p in run_paths)
    todo = set(range(0, 144000, 1000))
    diff = todo.difference(done)

    if len(diff) > 0:
        msg = f"Some runs are missing. Interrupting script.\nMissing runs {diff}"
        raise RuntimeError(msg)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_path", type=str, default="./outputs/multirun")
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--model_size", type=str, default="12b")
    args = parser.parse_args()

    out_path = Path(args.data_path) / "pile-deduped-stats"
    out_path.mkdir(exist_ok=True, parents=True)

    run_path = Path(args.run_path)
    run_paths = [p for p in (run_path / args.model_size).rglob("pythia*/data-pythia*.parquet")]

    # Load all run data for a specific model size
    tbl = db.sql(
        f"""
    select 
        cast(seq_idx as int32) as seq_idx,
        sup,
        cast(rank as int32[]) as rank,
        entropy,
        cast(split_part(split_part(filename, 'step', 2), '_', 1) as int) as step
    from read_parquet({[str(p) for p in run_paths]}, filename=True)
    """
    )

    # Aggregate
    ptbl = db.sql(
        """
    select
        seq_idx,
        step,
        list_avg(sup) as sup_seq,
        list_avg(entropy) as entr_seq,
        cast(len(list_filter(rank, x -> x == 0)) as int32) / len(rank) as acc_seq,
        list_avg(rank) as avg_rank
    from tbl
    """
    )

    # Save
    db.sql(f"copy ptbl to '{out_path}/{args.model_size}.parquet' (format 'parquet', codec 'zstd')")
