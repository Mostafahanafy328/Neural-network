import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ALPHABET_SIZE = 32
EPOCHS = 2000
LR = 0.005
MODEL_NAME = "add_model.pth"

def build_add_onehot(mod):
    table = torch.zeros(mod, mod, mod)
    for a in range(mod):
        for b in range(mod):
            v = (a + b) % mod
            table[a, b, v] = 1.0
    return table

ADD_ONEHOT = build_add_onehot(ALPHABET_SIZE)

def exact_add_target(a_dist, b_dist):
    return torch.einsum("ba,bc,acv->bv", a_dist, b_dist, ADD_ONEHOT)

def sample_soft_distribution(batch, mod):
    concentration = torch.exp(torch.empty(batch, 1).uniform_(-2.0, 4.0))
    alpha = concentration.expand(batch, mod)
    return torch.distributions.Dirichlet(alpha).sample()

def build_dataset():
    # Hard Samples
    a_idx = torch.arange(ALPHABET_SIZE).repeat_interleave(ALPHABET_SIZE)
    b_idx = torch.arange(ALPHABET_SIZE).repeat(ALPHABET_SIZE)
    a_hard = F.one_hot(a_idx, ALPHABET_SIZE).float()
    b_hard = F.one_hot(b_idx, ALPHABET_SIZE).float()
    Y_hard = exact_add_target(a_hard, b_hard)
    X_hard = torch.cat([a_hard, b_hard], dim=1)

    # Soft Samples
    N_SOFT = 8000
    a_soft = sample_soft_distribution(N_SOFT, ALPHABET_SIZE)
    b_soft = sample_soft_distribution(N_SOFT, ALPHABET_SIZE)
    Y_soft = exact_add_target(a_soft, b_soft)
    X_soft = torch.cat([a_soft, b_soft], dim=1)

    X = torch.cat([X_hard, X_soft], dim=0)
    Y = torch.cat([Y_hard, Y_soft], dim=0)
    return X, Y

X, Y = build_dataset()
print("Total samples =", len(X))

class ADDNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 32)

    def forward(self, x, return_logits=True):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits if return_logits else F.softmax(logits, dim=1)

model = ADDNet()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_loss = float("inf")
N_HARD = ALPHABET_SIZE * ALPHABET_SIZE

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