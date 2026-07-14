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
mul_model.load_state_dict(torch.load("mul_model.pth"))
mul_model.eval()

add_model = NN()
add_model.load_state_dict(torch.load("add_model.pth"))
add_model.eval()

xor_model = NN()
xor_model.load_state_dict(torch.load("xor_model.pth"))
xor_model.eval()

for net in (mul_model, add_model, xor_model):
    for p in net.parameters():
        p.requires_grad = False


# Augmented Hill Network

class AugmentedHillNet(nn.Module):

    def __init__(self, n, mod):

        super().__init__()

        self.n = n
        self.mod = mod

        # K0 , K1 , K2
        self.keys = nn.Parameter(torch.randn(3,n,n,mod) * 0.01)


    def neural_mul(self,key_distribution,plain_symbol):

        plain = F.one_hot(
            plain_symbol,
            num_classes=self.mod
        ).float()

        sample = torch.cat(
            [plain, key_distribution],
            dim=1
        )

        out = mul_model(sample)

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

        out = add_model(sample)

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

        out = xor_model(sample)

        return out

    # ------------------------------------------------------
    def affine_block(self,P,k0,k1):
        #C=k0*P+k1

        B = P.size(0)
        n = self.n

        outputs = []

        for row in range(n):

            row_outputs = []

            for col in range(n):

                result = None

                for k in range(n):

                    key_dist = k0[row, k]

                    plain_symbol = P[:, k, col]

                    logits = self.neural_mul(
                        key_dist.expand(B, self.mod),
                        plain_symbol
                    )

                    if result is None:

                        result = logits

                    else:

                        result = self.neural_add(
                            result,
                            F.softmax(logits, dim=1)
                        )

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


    def forward(self,plain,temperature=1.0):

        B = plain.size(0)

        n = self.n

        key_probs = F.softmax(
            self.keys / temperature,
            dim=3
        )

        k0 = key_probs[0]
        k1 = key_probs[1]
        k2 = key_probs[2]

        P = plain.view(B,n,n)

        Cp = self.affine_block(P,k0,k1)

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