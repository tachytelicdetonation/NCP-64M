from torch.utils.data import Dataset
import torch
import numpy as np
import os

class PretrainDataset(Dataset):
    """Fixed-length windows over a pretokenized token stream (dataset/*.bin).

    The stream is a flat concatenation of documents separated by EOS; windows are
    non-overlapping so every position is a real token — no padding, which keeps
    k-token concept chunks and mean pooling clean.
    """
    def __init__(self, bin_path, seq_len=512):
        super().__init__()
        self.seq_len = seq_len
        self.tokens = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.n_samples = len(self.tokens) // (seq_len + 1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        start = index * (self.seq_len + 1)
        buf = torch.from_numpy(self.tokens[start:start + self.seq_len + 1].astype(np.int64))
        return buf[:-1], buf[1:]
