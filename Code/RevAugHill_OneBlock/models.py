# Correct path no matter where the script is run from
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================================
# Basic Neural Network
# ======================================================

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


# ======================================================
# Load pretrained neural operators
# ======================================================

mul_model = NN()
mul_model.load_state_dict(
    torch.load(os.path.join(SCRIPT_DIR, "mul_model.pth"))
)
mul_model.eval()

add_model = NN()
add_model.load_state_dict(
    torch.load(os.path.join(SCRIPT_DIR, "add_model.pth"))
)
add_model.eval()

sub_model = NN()
sub_model.load_state_dict(
    torch.load(os.path.join(SCRIPT_DIR, "sub_model.pth"))
)
sub_model.eval()

xor_model = NN()
xor_model.load_state_dict(
    torch.load(os.path.join(SCRIPT_DIR, "xor_model.pth"))
)
xor_model.eval()


# Freeze pretrained models

for net in (mul_model, add_model, sub_model, xor_model):
    for p in net.parameters():
        p.requires_grad = False


# ======================================================
# Augmented Hill Decryption Network
# ======================================================

class AugmentedHillNet(nn.Module):

    def __init__(self, n, mod):

        super().__init__()

        self.n = n
        self.mod = mod

        self.keys = nn.Parameter(
            torch.randn(3, n, n, mod) * 0.01
        )

        self.mul_model = mul_model
        self.add_model = add_model
        self.sub_model = sub_model
        self.xor_model = xor_model

        self.mul_scale = nn.Parameter(torch.tensor(1.0))
        self.add_scale = nn.Parameter(torch.tensor(1.0))
        self.sub_scale = nn.Parameter(torch.tensor(1.0))
        self.xor_scale = nn.Parameter(torch.tensor(1.0))

    # ==================================================

    def neural_add(self, a_logits, b_distribution):

        a = F.softmax(
            a_logits,
            dim=1
        )

        sample = torch.cat(
            [a, b_distribution],
            dim=1
        )

        out = self.add_model(sample)

        return out * self.add_scale

    # ==================================================

    def neural_sub(self, a_logits, b_distribution):

        a = F.softmax(
            a_logits,
            dim=1
        )

        sample = torch.cat(
            [a, b_distribution],
            dim=1
        )

        out = self.sub_model(sample)

        return out * self.sub_scale

    # ==================================================

    def neural_xor(self, a_onehot, b_distribution):


        sample = torch.cat(
            [a_onehot, b_distribution],
            dim=1
        )

        out = self.xor_model(sample)

        return out * self.xor_scale

    # ==================================================

    def neural_mul(self, product_logits, key_distribution):

        product = F.softmax(
            product_logits,
            dim=1
        )

        sample = torch.cat(
            [product, key_distribution],
            dim=1
        )

        out = self.mul_model(sample)

        return out * self.mul_scale

    # ==================================================

    def forward(self, cipher, temperature=1.0):

        B = cipher.size(0)
        n = self.n

        key_probs = F.softmax(
            self.keys / temperature,
            dim=3
        )

        k0 = key_probs[0]
        k1 = key_probs[1]
        k2 = key_probs[2]

        C = cipher.view(B, n, n)

        # Step 1 : Remove XOR

        after_xor = []

        for row in range(n):

            row_outputs = []

            for col in range(n):

                cipher_onehot = F.one_hot(
                    C[:, row, col],
                    num_classes=self.mod
                ).float()

                logits = self.neural_xor(
                    cipher_onehot,
                    k2[row, col].expand(B, self.mod)
                )

                row_outputs.append(logits)

            after_xor.append(
                torch.stack(row_outputs, dim=1)
            )

        after_xor = torch.stack(after_xor, dim=1)

        # Step 2 : Remove Addition
        # C''_0 = C'_0 - K1

        after_sub = []

        for row in range(n):

            row_outputs = []

            for col in range(n):

                logits = self.neural_sub(
                    after_xor[:, row, col, :],
                    k1[row, col].expand(B, self.mod)
                )

                row_outputs.append(logits)

            after_sub.append(
                torch.stack(row_outputs, dim=1)
            )

        after_sub = torch.stack(after_sub, dim=1)

        # Step 3 : Recover Plain

        plain = []

        for row in range(n):

            row_outputs = []

            for col in range(n):

                result = None

                for k in range(n):

                    logits = self.neural_mul(
                        after_sub[:, k, col, :],      # FIXED: was after_sub[:, row, k, :]
                        k0[row, k].expand(B, self.mod)
                    )

                    if result is None:

                        result = logits

                    else:

                        result = self.neural_add(
                            result,
                            F.softmax(logits, dim=1)
                        )

                row_outputs.append(result)

            plain.append(
                torch.stack(row_outputs, dim=1)
            )

        plain = torch.stack(plain, dim=1)

        return plain.view(B, n * n, self.mod)