import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Parameters

ALPHABET_SIZE = 32
EPOCHS = 2000
LR = 0.005
MODEL_NAME = "xor_model.pth"

def build_xor_onehot(mod):

    table = torch.zeros(mod, mod, mod)
    for a in range(mod):
        for b in range(mod):
            v = a ^ b
            table[a, b, v] = 1.0
    return table

XOR_ONEHOT = build_xor_onehot(ALPHABET_SIZE)


def exact_xor_target(a_dist, b_dist):

    return torch.einsum("ba,bc,acv->bv", a_dist, b_dist, XOR_ONEHOT)


def sample_soft_distribution(batch, mod):
    concentration = torch.exp(torch.empty(batch, 1).uniform_(-2.0, 4.0))
    alpha = concentration.expand(batch, mod)
    return torch.distributions.Dirichlet(alpha).sample()



def build_dataset():
    # 1) Hard Samples
    a_idx = torch.arange(ALPHABET_SIZE).repeat_interleave(ALPHABET_SIZE)
    b_idx = torch.arange(ALPHABET_SIZE).repeat(ALPHABET_SIZE)
    a_hard = F.one_hot(a_idx, ALPHABET_SIZE).float()
    b_hard = F.one_hot(b_idx, ALPHABET_SIZE).float()
    Y_hard = exact_xor_target(a_hard, b_hard)
    X_hard = torch.cat([a_hard, b_hard], dim=1)

    # 2) Soft Samples
    N_SOFT = 8000
    a_soft = sample_soft_distribution(N_SOFT, ALPHABET_SIZE)
    b_soft = sample_soft_distribution(N_SOFT, ALPHABET_SIZE)
    Y_soft = exact_xor_target(a_soft, b_soft)
    X_soft = torch.cat([a_soft, b_soft], dim=1)

    X = torch.cat([X_hard, X_soft], dim=0)
    Y = torch.cat([Y_hard, Y_soft], dim=0)
    return X, Y


X, Y = build_dataset()
print("Total samples =", len(X))


class XorNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(ALPHABET_SIZE * 2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, ALPHABET_SIZE)

    def forward(self, x, return_logits=True):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits if return_logits else F.softmax(logits, dim=1)


model = XorNet()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_loss = float("inf")
N_HARD = ALPHABET_SIZE * ALPHABET_SIZE

# Training

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()

    logits = model(X)
    log_probs = F.log_softmax(logits, dim=1)
    loss = F.kl_div(log_probs, Y, reduction="batchmean")

    loss.backward()
    optimizer.step()

    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(model.state_dict(), MODEL_NAME)

    if (epoch + 1) % 50 == 0:
        with torch.no_grad():
            hard_pred = logits[:N_HARD].argmax(dim=1)
            hard_true = Y[:N_HARD].argmax(dim=1)
            acc = (hard_pred == hard_true).float().mean().item() * 100
        print(f"Epoch {epoch+1:04d} | KL={loss.item():.6f} | HardAcc={acc:.2f}%")

print("\nBest KL-Loss =", best_loss)


model.load_state_dict(torch.load(MODEL_NAME))
model.eval()

"""
with torch.no_grad():
    logits = model(X[:N_HARD])
    pred = logits.argmax(dim=1)
    true = Y[:N_HARD].argmax(dim=1)
    full_acc = (pred == true).float().mean().item() * 100

print("Full Hard-Table Accuracy =", full_acc, "%")

# Example

a, b = 7, 21
a_vec = F.one_hot(torch.tensor(a), ALPHABET_SIZE).float()
b_vec = F.one_hot(torch.tensor(b), ALPHABET_SIZE).float()
sample = torch.cat([a_vec, b_vec]).unsqueeze(0)

with torch.no_grad():
    pred = model(sample).argmax(dim=1).item()

print(f"\nExample: {a} XOR {b} = {pred} (true = {a ^ b})")
"""