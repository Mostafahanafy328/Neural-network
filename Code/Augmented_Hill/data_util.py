import numpy as np
import torch
from torch.utils.data import Dataset
from math import gcd

MOD = 32

# ================= utils =================
def random_invertible_key(n):
    while True:
        K = np.random.randint(0, MOD, size=(n, n))
        det = int(round(np.linalg.det(K))) % MOD
        if gcd(det, MOD) == 1:
            return K

# ================= data generation =================
def generate_data(n, block_size, num_samples):
    K = [random_invertible_key(n) for _ in range(3)]

    P = np.random.randint(0, MOD, size=(num_samples, block_size, n, n))
    C = np.zeros_like(P)

    for s in range(num_samples):
        Cdash_prev = K[2]
        for t in range(block_size):
            Cp = (P[s, t] @ K[t % 3]+ K[(t + 1) % 3]) % MOD
            C[s, t] = Cp ^ Cdash_prev
            Cdash_prev = Cp

    return C,P, K

# ================= Dataset =================
class ADataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
