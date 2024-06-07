import pickle
from collections import Counter

import numpy as np
from tqdm import tqdm

from pythia.utils.mmap_dataset import MMapIndexedDataset

PATH = "data/pile-deduped-train/document"
CHUNK_SIZE = 5 * 1024

if __name__ == "__main__":
    array = MMapIndexedDataset(PATH, skip_warmup=True)

    # Function to count items within a chunk using numpy
    def count_chunk(chunk):
        unique, counts = np.unique(chunk, return_counts=True)
        return dict(zip(unique, counts, strict=False))

    # Initialize a shared dictionary to accumulate counts
    shared_dict = Counter()

    # Get the total number of chunks
    num_chunks = len(array) // CHUNK_SIZE + (1 if len(array) % CHUNK_SIZE != 0 else 0)

    # Process chunks in parallel and update the shared dictionary
    with tqdm(total=num_chunks, desc="Processing chunks") as pbar:
        for i in range(0, len(array), CHUNK_SIZE):
            chunk = array[i : i + CHUNK_SIZE]
            chunk_counter = count_chunk(chunk)
            shared_dict.update(chunk_counter)
            pbar.update(1)

    with open("pile_token_frequency.pkl", "wb") as f:
        pickle.dump(shared_dict, f)
