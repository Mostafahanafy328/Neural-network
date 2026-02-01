import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from math import gcd
from itertools import product


MOD = 26

# =============== Hill Cipher ====================
def random_invertible_key(n):
    while True:
        K = np.random.randint(0, MOD, size=(n, n)).astype(np.int64)
        det = int(round(np.linalg.det(K))) % MOD
        if gcd(det, MOD) == 1:
            return K


def generate_data(dim, Key,num_sample):
    #P = np.array(list(product(range(MOD), repeat=n)), dtype=np.int64)
    P = np.random.randint(0, 26, size=(num_sample,dim)).astype(np.int64)
    C = (P.dot(Key.T) % MOD).astype(np.int64)
    return C, P


# =============== Dataset ====================
class Dataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]