# Correct path no matter where the script is run from
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Old NN

class NN(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 32)

    def forward(self, x, return_logits=True):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        logits = self.fc3(x)

        if return_logits:
            return logits

        return F.softmax(logits, dim=1)


# Load pretrained models and freezing them


mul_model = NN()
mul_model.load_state_dict(torch.load(os.path.join(SCRIPT_DIR, "mul_model.pth")))
mul_model.eval()

add_model = NN()
add_model.load_state_dict(torch.load(os.path.join(SCRIPT_DIR, "add_model.pth")))
add_model.eval()

xor_model = NN()
xor_model.load_state_dict(torch.load(os.path.join(SCRIPT_DIR, "xor_model.pth")))
xor_model.eval()

for net in (mul_model, add_model, xor_model):
    for p in net.parameters():
        p.requires_grad = False  # Keep models frozen


# Augmented Hill Network

class AugmentedHillNet(nn.Module):

    def __init__(self, n, mod):

        super().__init__()

        self.n = n
        self.mod = mod

        # K0 , K1 , K2 - Initialize with small random values
        self.keys = nn.Parameter(torch.randn(3, n, n, mod) * 0.01)
        
        # Register pretrained frozen models as submodules 
        # so that we can fine-tune them later 
        self.mul_model = mul_model
        self.add_model = add_model
        self.xor_model = xor_model
        
        # Learnable scaling factors to adjust output magnitudes
        # This increase the performance to 75% from 30% accuracy
        self.mul_scale = nn.Parameter(torch.tensor(1.0))
        self.add_scale = nn.Parameter(torch.tensor(1.0))
        self.xor_scale = nn.Parameter(torch.tensor(1.0))

    def neural_mul(self,key_distribution,plain_symbol):

        plain = F.one_hot(
            plain_symbol,
            num_classes=self.mod
        ).float()

        sample = torch.cat(
            [plain, key_distribution],
            dim=1
        )

        out = self.mul_model(sample)
        out = out * self.mul_scale # use the scale to adapt the distribution to what the model was learned on initially 

        return out

    def neural_add(self,a_logits,b_distribution):

        a = F.softmax(
            a_logits,
            dim=1
        )

        sample = torch.cat(
            [a, b_distribution],
            dim=1
        )

        out = self.add_model(sample)
        out = out * self.add_scale

        return out

    # ------------------------------------------------------

    def neural_xor(self,a_logits,b_distribution):

        a = F.softmax(
            a_logits,
            dim=1
        )

        sample = torch.cat(
            [a, b_distribution],
            dim=1
        )

        out = self.xor_model(sample)
        out = out * self.xor_scale

        return out

    # ------------------------------------------------------
    def affine_block(self,P,k0,k1, debug=False):
        #C=k0*P+k1

        B = P.size(0)
        n = self.n

        outputs = []

        for row in range(n):

            row_outputs = []

            for col in range(n):

                result = None

                for k in range(n):

                    key_dist = k0[row, k] # a representation of length MOD for an integer (one cell of the key matrix)

                    plain_symbol = P[:, k, col] # a representation of length MOD for an integer (one cell of the plain text matrix)
                    

                    logits = self.neural_mul(
                        key_dist.expand(B, self.mod),
                        plain_symbol
                    ) # a representation of length MOD for the multiplication k0[row, k] * P[:, k, col]
                    
                    # DEBUG: Check mul_model output distribution
                    if debug and row == 0 and col == 0 and k == 0:
                        print(f"\n[DEBUG] neural_mul output:")
                        print(f"  Min: {logits.min():.4f}, Max: {logits.max():.4f}, Mean: {logits.mean():.4f}, Std: {logits.std():.4f}")

                    if result is None:

                        result = logits

                    else:
                        
                        logits_softmax = F.softmax(logits, dim=1)
                        
                        # DEBUG: Check add_model input distribution
                        if debug and row == 0 and col == 0:
                            print(f"\n[DEBUG] neural_add input:")
                            print(f"  result (from previous add) - Min: {result.min():.4f}, Max: {result.max():.4f}")
                            print(f"  logits_softmax - Min: {logits_softmax.min():.4f}, Max: {logits_softmax.max():.4f}")
                            print(f"  logits_softmax sum per sample: {logits_softmax.sum(dim=1)[:3]}  (should be ~1.0)")

                        result = self.neural_add(
                            result,
                            logits_softmax
                        )
                        
                        # DEBUG: Check add_model output
                        if debug and row == 0 and col == 0:
                            print(f"\n[DEBUG] neural_add output:")
                            print(f"  Min: {result.min():.4f}, Max: {result.max():.4f},  Mean: {result.mean():.4f}, Std: {result.std():.4f}")

                add_dist = k1[row, col]

                result = self.neural_add(
                    result,
                    add_dist.expand(B, self.mod)
                )
                
                row_outputs.append(result)

            outputs.append(
                torch.stack(row_outputs, dim=1)
            )

        outputs = torch.stack(outputs, dim=1)

        return outputs


    def forward(self, plain, temperature=1.0, debug=False):

        B = plain.size(0)

        n = self.n
        
        # Use regular softmax with temperature scaling (no random noise)
        # High temperature = soft keys, low temperature = sharp keys
        key_probs = F.softmax(
            self.keys / temperature,
            dim=3
        )

        k0 = key_probs[0]
        k1 = key_probs[1]
        k2 = key_probs[2]

        P = plain.view(B,n,n)

        Cp = self.affine_block(P,k0,k1, debug=debug)

        # XOR K2

        outputs = []

        for row in range(n):

            row_outputs = []

            for col in range(n):

                cp_logits = Cp[:, row, col, :]

                kk2 = k2[row, col]

                out = self.neural_xor(
                    cp_logits,
                    kk2.expand(B, self.mod)
                )

                row_outputs.append(out)

            outputs.append(
                torch.stack(row_outputs, dim=1)
            )

        outputs = torch.stack(outputs, dim=1)

        return outputs.view(B,n * n,self.mod)