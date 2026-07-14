import numpy as np
import torch
from torch.utils.data import Dataset
from math import gcd
import config as con

MOD = con.Mod

# ================= invertible Key =================
def random_invertible_key(n):
    while True:
        K = np.random.randint(0, MOD, size=(n, n))
        det = int(round(np.linalg.det(K))) % MOD
        if gcd(det, MOD) == 1:
            return K

# ================= data generation =================
def generate_data(n, num_samples):
    K = [random_invertible_key(n) for _ in range(3)]

    P = np.random.randint(0, MOD, size=(num_samples, n, n))
    C = np.zeros_like(P)

    for s in range(num_samples):
        Cp = (K[0] @ P[s] + K[1]) % MOD
        C[s] = Cp ^ K[2]
        
    return P, C, K

# ================= Dataset =================
class ADataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
