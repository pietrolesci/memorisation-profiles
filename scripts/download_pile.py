import os
from argparse import ArgumentParser

from huggingface_hub import snapshot_download

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/pile-deduped-train")
    parser.add_argument("--cache_dir", type=str, default=".pile_data_cache")
    args = parser.parse_args()

    path = snapshot_download(
        repo_id="EleutherAI/pile-deduped-pythia-preshuffled",
        repo_type="dataset",
        cache_dir=str(args.data_path / args.cache_dir),
        local_dir=str(args.data_path),
        resume_download=True,
        max_workers=os.cpu_count(),  # type: ignore
    )

    print(f"Data downloaded in folder {path}")
