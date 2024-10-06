"""
IMPORTANT: ctx_len MUST be set to the training ctx len to prevent data leakage between documents.

Usage: python insert_pause.py base_in_path base_out_path maximum_p pause_token_idx ctx_len
Where:
    base_in_path    = the name of the .idx and .bin files to insert pause tokens into, but without their extensions. Example: `data/minipile` for `data/minipile.bin`
    base_out_path   = name of the output .idx and .bin files without their extensions. Example: `output` for `output.bin` and `output.idx`
    maximum_p       = decimal value of the maximum pause token percent. The average amount will be half this value. Example: 0.2 means a training example can have at most 20% pause tokens, a minimum of 0%, and an average of 10%.
    pause_token_idx = index of the pause token in your vocabulary
    ctx_len         = Context size of the dataset. IMPORTANT: This must be set correctly to prevent leakage between documents.

Example:
python insert_pause.py data/minipile data/minipile_with_pauses 0.2 65530 512
"""

import numpy as np
import random
import time, sys
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
    len = 0
    for token in seq:
        temp.append(token)
        len += 1
        while random.random() < p and len < CTX_LEN:
            temp.append(PAUSE_TOKEN)
            len += 1
        if len == CTX_LEN + 1:
            break
    return np.array(temp, dtype=np.uint16)


if len(sys.argv) != 6:
    print("Invalid number of arguments.")
    print("Usage: python insert_pause.py in_base_path out_base_path maximum_p pause_token_idx ctx_len")
    print("Where:")
    print("    in_base_path    = the name of the .idx and .bin files to insert pause tokens into, but without their extensions. Example: `data/minipile` for `data/minipile.bin`")
    print("    out_base_path   = name of the output .idx and .bin files without their extensions. Example: `output` for `output.bin` and `output.idx`")
    print("    maximum_p       = decimal value of the maximum pause token percent. The average amount will be half this value. Example: 0.2 means a training example can have at most 20% pause tokens, a minimum of 0%, and an average of 10%.")
    print("    pause_token_idx = index of the pause token in your vocabulary")
    print("    ctx_len         = Context size of the dataset. IMPORTANT: This must be set correctly to prevent leakage between documents.")
    print("\n\nIMPORTANT: ctx_len MUST be set to the training ctx len to prevent data leakage between documents.")
    quit()

IN_NAME = sys.argv[1].strip()
OUT_NAME = sys.argv[2].strip()
P_MAX = float(sys.argv[3].strip())
PAUSE_TOKEN = int(sys.argv[4].strip())
CTX_LEN = int(sys.argv[5].strip())

dataset = MMapIndexedDataset(IN_NAME)
builder = MMapIndexedDatasetBuilder(f"{OUT_NAME}.bin")

dataset_len = len(dataset)

for idx, seq in enumerate(dataset):
    p = random.random() * P_MAX
    builder.add_item(uniform_random_insert(p, seq))
    builder.end_document()

    if idx % 1000 == 0:
        print(f"    {idx + 1} / {dataset_len}     ({(idx + 1) / dataset_len * 100:.3f}%)      \r", end="")

builder.finalize((f"{OUT_NAME}.idx"))
