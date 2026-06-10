import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

MOD = 32

# =================================================================================
# MAPS — Tensor Lookup Tables
# Instead of applying operations directly on integers (non-differentiable),
# we build a 3D lookup table of shape (MOD, MOD, MOD) where:
#   MAP[a, b, v] = 1.0   if   f(a, b) == v
#   MAP[a, b, v] = 0.0   otherwise
# This lets us apply any modular operation on probability distributions.
# =================================================================================

#Lookup table for Multiplication
def build_mul_maps():
    A = np.zeros((MOD, MOD, MOD), dtype=np.float32)
    for p in range(MOD):
        for k in range(MOD):
            A[p, k, (p * k) % MOD] = 1.0
    return torch.tensor(A)

#Lookup table for XOR
def build_xor_map():
    A = np.zeros((MOD, MOD, MOD), dtype=np.float32)
    for a in range(MOD):
        for b in range(MOD):
            A[a, b, a ^ b] = 1.0
    return torch.tensor(A)

#Lookup table for Addition
def build_add_map():
    A = np.zeros((MOD, MOD, MOD), dtype=np.float32)
    for a in range(MOD):
        for b in range(MOD):
            A[a, b, (a + b) % MOD] = 1.0
    return torch.tensor(A)

#Lookup table for Sub
def build_sub_map():
    A = np.zeros((MOD, MOD, MOD), dtype=np.float32)
    for a in range(MOD):
        for b in range(MOD):
            A[a, b, (a - b) % MOD] = 1.0
    return torch.tensor(A)

# Build all lookup tables once at startup

SUB_MAP = build_sub_map()
MUL_MAP = build_mul_maps()
XOR_MAP = build_xor_map()
ADD_MAP = build_add_map()

# =================================================================================
# OPS — Differentiable operations on probability distributions
# Each function takes two distributions (a, b) of shape (B, MOD)
# and returns the distribution of the result, also shape (B, MOD).
#
# Unified formula:  einsum("bi,bj,ijv->bv", a, b, MAP)
#   b = batch dimension
#   i = possible values of a  (0 .. MOD-1)
#   j = possible values of b  (0 .. MOD-1)
#   v = possible values of the result  (0 .. MOD-1)
#
# Intuition: result[v] = sum of a[i]*b[j] for all (i,j) where f(i,j)==v
# =================================================================================

def Xor_map(a, b):
    return torch.einsum("bi,bj,ijv->bv", a, b, XOR_MAP)
def Add_map(a, b):
    return torch.einsum("bi,bj,ijv->bv", a, b, ADD_MAP)
def Sub_map(a, b):
    return torch.einsum("bi,bj,ijv->bv", a, b, SUB_MAP)
def Mul_map(a, b):
    return torch.einsum("bi,bj,ijv->bv", a, b, MUL_MAP)

def inverse_constraint_loss(K_probs, K_inv_probs, n):
    loss = 0

    for i in range(3): # loop over the 3 key matrices

        for r in range(n):
            for c in range(n):

                dist = None

                for k in range(n):

                    a = K_probs[i, r, k]       # (MOD)
                    b = K_inv_probs[i, k, c]   # (MOD)

                    prod = Mul_map(a.unsqueeze(0), b.unsqueeze(0)).squeeze(0)

                    if dist is None:
                        dist = prod
                    else:
                        dist = Add_map(dist.unsqueeze(0), prod.unsqueeze(0)).squeeze(0)

                target = torch.zeros_like(dist)

                if r == c:
                    target[1] = 1.0   # diagonal: value 1 (multiplicative identity)
                else:
                    target[0] = 1.0   # off-diagonal: value 0 (additive identity)
                
                # MSE between computed distribution and target
                loss += torch.mean((dist - target) ** 2)
    # Normalize by total number of matrix entries across all 3 keys
    return loss / (3 * n * n)


# ================= MODEL =================
class AugmentedHillModel(nn.Module):

    def __init__(self, n):
        super().__init__()

        self.n = n
        N = n * n
        
        # K     : (3, N, MOD) — 3 key matrices, each entry is a distribution over 0..MOD-1
        # K_inv : (3, N, MOD) — 3 inverse key matrices, same structure
        
        self.K = nn.Parameter(torch.randn(3, N, MOD) * 0.0001)
        self.K_inv = nn.Parameter(torch.randn(3, N, MOD) * 0.0001)

        self.register_buffer("A", MUL_MAP)

    def forward(self, C):

        B = C.size(0)  # batch size
        block_size = C.size(1) # number of blocks per sample
        
        n = self.n
        
        # Reshape C to (B, block_size, n, n)

        C = C.view(B, block_size, n, n)

        # one-hot
        C_onehot = torch.zeros(B, block_size, n, n, MOD)
        C_onehot.scatter_(4, C.unsqueeze(-1), 1.0)

        # key,K_inv probs
        K_probs = torch.softmax(self.K, dim=-1).view(3, n, n, MOD)
        K_inv_probs = torch.softmax(self.K_inv, dim=-1).view(3, n, n, MOD)

        outputs = []

        # chaining
        """
            K2_hard = F.one_hot(K_probs[2].argmax(dim=-1), num_classes=MOD).float()
            C_prev = K2_hard.reshape(n*n, MOD).unsqueeze(0).expand(B, -1, -1)
        """
        C_prev = K_probs[2].reshape(n*n, MOD).unsqueeze(0).expand(B, -1, -1)

        for t in range(block_size):

            # XOR
            C_flat = C_onehot[:, t].reshape(B*n*n, MOD)

            Cdash = Xor_map(
                C_flat,
                C_prev.reshape(B*n*n, MOD)
            ).reshape(B, n*n, MOD)

            
            block_out = []
            for r in range(n):
                for k in range(n):
                    
                    final_dist = None

                    for j in range(n):

                        Cp_dist = Cdash[:, r*n + j, :]
                        bias = K_probs[(t + 1) % 3, r, j].unsqueeze(0).expand(B, -1)
  
                        Cp_dist = Sub_map(Cp_dist,bias)

                        key_dist = K_inv_probs[t % 3, j, k].unsqueeze(0).expand(B, -1)

                        dist_k = Mul_map(Cp_dist, key_dist)

                        if final_dist is None:
                            final_dist = dist_k
                        else:
                            final_dist = Add_map(final_dist, dist_k)

                    block_out.append(final_dist)            

            P_block = torch.stack(block_out, dim=1)
            outputs.append(P_block)

            C_prev = Cdash

        out_stack = torch.stack(outputs, dim=1)
         
        # Convert to log-probabilities for NLLLoss
        # Adding 1e-12 to avoid log(0)
        
        log_probs = torch.log(out_stack + 1e-12)

        return log_probs, K_probs, K_inv_probs