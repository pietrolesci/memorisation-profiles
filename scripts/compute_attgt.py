import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from differences import ATTgt

# config
batch_size = 1024
cohort_size = 1000
BIG_NUM = 200_000_000


def load_data(data_path: str | Path, ref_path: str | Path, remove_dups: bool) -> pd.DataFrame:
    # load data and tidy
    df = pd.read_parquet(data_path)
    ref_df = pd.read_parquet(ref_path)["seq_idx"].unique().tolist()

    df = df.loc[df["seq_idx"].isin(ref_df)]
    assert df["seq_idx"].nunique() == len(ref_df)

    treated_df = (
        df.query("seq_idx > 0")
        .assign(batch_idx=lambda _df: np.ceil(_df["seq_idx"] / batch_size).astype(int))
        .assign(cohort=lambda _df: np.ceil(_df["batch_idx"] / cohort_size).astype(int) * cohort_size)
    )
    untreated_df = df.query("seq_idx <= 0").assign(batch_idx=BIG_NUM, cohort=np.nan)
    data = pd.concat([treated_df, untreated_df], axis=0, ignore_index=False)

    if remove_dups:
        print("Removing dups")
        data = data.query("((cohort <= 95000) | (cohort.isna())) & (step <= 95000)")

    return data


def compute_cs_did(
    df: pd.DataFrame,
    unit_col: str,
    time_col: str,
    cohort_col: str,
    target_col: str,
    bootstrap: bool,
    num_proc: int = 32,
) -> dict[str, pd.DataFrame]:
    att_model = ATTgt(data=df.set_index([unit_col, time_col]), cohort_name=cohort_col)
    att_results = att_model.fit(
        target_col,
        est_method="dr",
        control_group="never_treated",
        n_jobs=num_proc,
        random_state=1234,
        boot_iterations=1000 if bootstrap else 0,
    )
    cohort_results = att_model.aggregate("cohort")
    time_results = att_model.aggregate("time")
    event_results = att_model.aggregate("event")

    att_results.columns = att_results.columns.droplevel([0, 1])
    cohort_results.columns = cohort_results.columns.droplevel([0, 1])
    time_results.columns = time_results.columns.droplevel([0, 1])
    event_results.columns = event_results.columns.droplevel([0, 1])

    return {
        "all": att_results.reset_index(),
        "cohort": cohort_results.reset_index(),
        "time": time_results.reset_index(),
        "event": event_results.reset_index(),
    }


def compute_simple_diff(
    df: pd.DataFrame, unit_col: str, time_col: str, cohort_col: str, target_col: str
) -> pd.DataFrame:
    data = df.copy()
    data[cohort_col] = data[cohort_col].fillna(BIG_NUM).astype(int)
    # demean

    data[target_col] = (
        data[target_col]
        - data.groupby(time_col)[target_col].transform("mean")
        - data.groupby(unit_col)[target_col].transform("mean")
        + data[target_col].mean()
    )

    # aggregate
    att = data.groupby([cohort_col, time_col])[target_col].agg(["mean", "std", "size"]).reset_index()
    att = pd.merge(
        att.query(f"{cohort_col} == {BIG_NUM}"),
        att.query(f"{cohort_col} != {BIG_NUM}"),
        on=time_col,
        suffixes=["", "_val"],
    )
    att = att.assign(
        ATT=lambda _df: _df["mean"] - _df["mean_val"]
        # effect_std=lambda _df: (_df["std"] ** 2 + _df["std_val"] ** 2) / (_df["size"] + _df["size_val"]),
    )

    return att


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/pile-deduped-stats")
    parser.add_argument("--out_dir", type=str, default="./data/pile-deduped-effects")
    parser.add_argument("--target_col", type=str, default="sup_seq")
    parser.add_argument("--remove_dups", action="store_true")
    parser.add_argument("--only_simple_diff", action="store_true")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--model_size", type=str)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    data_path = data_dir / f"{args.model_size}.parquet"
    ref_path = data_dir / "70m.parquet"
    data = load_data(data_path, ref_path, args.remove_dups)

    if not args.only_simple_diff:
        results = compute_cs_did(
            data,
            unit_col="seq_idx",
            time_col="step",
            cohort_col="cohort",
            target_col=args.target_col,
            bootstrap=args.bootstrap,
            num_proc=8,
        )

        for k, v in results.items():
            name = f"cs_{args.model_size}-{args.target_col}-{k}"
            if args.remove_dups:
                name += "_nodups"
            if args.bootstrap:
                name += "_boot"
            v.to_parquet(out_dir / f"{name}.parquet", index=False)

    results_diff = compute_simple_diff(
        data, unit_col="seq_idx", time_col="step", cohort_col="cohort", target_col=args.target_col
    )

    name = f"diff_{args.model_size}-{args.target_col}"
    if args.remove_dups:
        name += "_nodups"

    results_diff.to_parquet(out_dir / f"{name}.parquet", index=False)
