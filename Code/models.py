import torch
import torch.nn as nn
import numpy as np
from modddd import Mod26Model

class sincosNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.groups = nn.ModuleList(
            [nn.Linear(input_dim, 1, bias=False) for _ in range(input_dim)]
        )
        self.SCALE = 2028 #if dim=3 (26*26+26*26+26*26)
        # ================== Load frozen Mod26 model ==================
        self.modelmod = Mod26Model()
        self.modelmod.load_state_dict(torch.load("mod26_frozen.pth"))

        for p in self.modelmod.parameters():
            p.requires_grad = False
        
    def forward(self, x):

        B = x.size(0)

        # ----- matrix multiplication learning -----
        outs = [g(x) for g in self.groups]     # each: (B, 1)
        y = torch.cat(outs, dim=1)             # (B, 3)
        f1 = y / self.SCALE
        f2 = torch.sin(y * 2 * torch.pi / 26)
        f3 = torch.cos(y * 2 * torch.pi / 26)
        features = torch.stack([f1, f2, f3], dim=2)  # (B, 3, 3)
        # ----- flatten AFTER feature construction -----
        features = features.view(-1, 3)  # (B*3, 3)
        # ----- frozen mod26 model -----
        out = self.modelmod(features)  # (B*3, 26)

        return out
    
MOD = 26
# =============== Build multiplication map ====================
def build_mul_maps():
    A = np.zeros((MOD, MOD, MOD), dtype=np.float32)
    for p in range(MOD):
        for k in range(MOD):
            v = (p * k) % MOD
            A[p, k, v] = 1.0
    return A

MUL_MAP = build_mul_maps()


# =============== Circular Convolution ====================
def circ_conv_batch(a, b):
    #FFT (Fast Fourier Transform) ,r is real number not complex
    Fa = torch.fft.rfft(a, n=MOD)
    Fb = torch.fft.rfft(b, n=MOD)
    Fc = Fa * Fb
    c = torch.fft.irfft(Fc, n=MOD)
    c = torch.clamp(c, min=0.0)
    s = c.sum(dim=-1, keepdim=True)
    s[s == 0] = 1.0
    return c / s

# =============== FFT Network ====================
class FFTNetwork(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.key = nn.Parameter(torch.randn(n * n, MOD) * 0.01)

    def forward(self, P):
        B = P.size(0)

        key_probs = torch.softmax(self.key , dim=-1).view(self.n, self.n, MOD) #softmax + reshape to (n,n,mod=26)

        A = torch.tensor(MUL_MAP, dtype=torch.float32) #to convert map from (np) to (tensor)

        P_onehot = torch.zeros(B, self.n, MOD)#plaintext (batchsize,n,mod=26)
    
        P_onehot.scatter_(2, P.unsqueeze(-1), 1.0) #(Dim=2,add one more Dim for plaintext, add one in matrix)

        Kp_all = torch.tensordot(key_probs, A, dims=([2], [1]))  # (n,n,mod,mod)

        prod = torch.einsum('blp, lipc -> blic', P_onehot, Kp_all) #b=>batch , l is letter plain , i is key col,c is cipher

        out_dists = []
        for i in range(self.n):
            dist = prod[:, 0, i, :]
            for j in range(1, self.n):
                dist = circ_conv_batch(dist, prod[:, j, i, :])
            out_dists.append(dist)
        
        out_stack = torch.stack(out_dists, dim=1)
        eps = 1e-12
        log_probs = torch.log(out_stack + eps)
        
        return log_probs, key_probs