"""
Usage: python insert_pause.py base_in_path base_out_path maximum_p pause_token_idx
Where:
    base_in_path    = the name of the .idx and .bin files to insert pause tokens into, but without their extensions. Example: `data/minipile` for `data/minipile.bin`
    base_out_path   = name of the output .idx and .bin files without their extensions. Example: `output` for `output.bin` and `output.idx`
    maximum_p       = decimal value of the maximum pause token percent. The average amount will be half this value. Example: 0.2 means a training example can have at most 20% pause tokens, a minimum of 0%, and an average of 10%.
    pause_token_idx = index of the pause token in your vocabulary

Example:
python insert_pause.py data/minipile data/minipile_with_pauses 0.2 65530
"""

import numpy as np
import random, math
import time, sys, gc
from src.binidx import MMapIndexedDataset

random.seed(0)

class MMapIndexedDatasetBuilder(object):
    def __init__(self, out_file, dtype=np.uint16):
        self._data_file = open(out_file, "wb")
        self._dtype = dtype
        self._sizes = []
        self._doc_idx = [0]
    def add_item(self, np_array):
        assert np_array.dtype == self._dtype
        self._data_file.write(np_array.tobytes(order="C"))
        self._sizes.append(np_array.size)
    def end_document(self):
        self._doc_idx.append(len(self._sizes))
    def finalize(self, index_file):
        self._data_file.close()
        with MMapIndexedDataset.Index.writer(index_file, self._dtype) as index:
            index.write(self._sizes, self._doc_idx)

def uniform_random_insert(p: float, seq: np.ndarray) -> np.ndarray:
    temp = []
    for token in seq[:-1]:
        temp.append(token)
        while random.random() < p:
            temp.append(PAUSE_TOKEN)
    temp.append(seq[-1])
    return np.array(temp, dtype=np.uint16)


if len(sys.argv) != 5:
    print("\nInvalid number of arguments.\n")
    print("Usage: python insert_pause.py in_base_path out_base_path maximum_p pause_token_idx")
    print("Where:")
    print("    in_base_path    = the name of the .idx and .bin files to insert pause tokens into, but without their extensions. Example: `data/minipile` for `data/minipile.bin`")
    print("    out_base_path   = name of the output .idx and .bin files without their extensions. Example: `output` for `output.bin` and `output.idx`")
    print("    maximum_p       = decimal value of the maximum pause token percent. The average amount will be half this value. Example: 0.2 means a training example can have at most 20% pause tokens, a minimum of 0%, and an average of 10%.")
    print("    pause_token_idx = index of the pause token in your vocabulary")
    quit()

IN_NAME = sys.argv[1].strip()
OUT_NAME = sys.argv[2].strip()
P_MAX = float(sys.argv[3].strip())
PAUSE_TOKEN = int(sys.argv[4].strip())

print("Adding pause tokens to dataset...")
print(f"    Input files:  {IN_NAME}.bin, {IN_NAME}.idx")
print(f"    Output files: {OUT_NAME}.bin, {OUT_NAME}.idx")
print(f"    Pause token proportion: 0.00 - {P_MAX * 100.0:.2f}% randomly choosen per document. Average: ~{P_MAX * 50.0:.2f}% pause tokens")
print(f"    Pause token: {PAUSE_TOKEN}")

dataset = MMapIndexedDataset(IN_NAME)
builder = MMapIndexedDatasetBuilder(f"{OUT_NAME}.bin")

dataset_len = len(dataset)

print(f"    Input dataset: {dataset_len} documents, {len(dataset._bin_buffer) // dataset._index._dtype_size} tokens\n\n")

tokens = 0
for idx, seq in enumerate(dataset):
    p = random.random() * P_MAX
    seq = uniform_random_insert(p, seq)
    
    builder.add_item(seq)
    builder.end_document()
    tokens += len(seq)

    if idx % 1000 == 0:
        print(f"    {idx + 1} / {dataset_len}     ({(idx + 1) / dataset_len * 100:.3f}%)      \r", end="\n" if idx % 100000 == 0 else "")

builder.finalize((f"{OUT_NAME}.idx"))
print(f"\n\nOutput dataset: {dataset_len} documents, {tokens} tokens\n")

# Calculate magic numbers for common ctx lens
print("Magic numbers:")
prime_flags = np.full(tokens // 16, True)
for i in range(2, math.floor(math.sqrt(tokens // 16)) + 1):
    if prime_flags[i]:
        prime_flags[i*i::i] = False
primes = np.flatnonzero(prime_flags)

for i in range(4, 18):
    limit = tokens / 2 ** i - 1
    highest_prime = 0
    for prime in primes:
        if prime >= limit:
            break
        if prime % 3 == 2:
            highest_prime = prime
    print(f"    ctx_len: {2 ** i}, magic number: {highest_prime}")
