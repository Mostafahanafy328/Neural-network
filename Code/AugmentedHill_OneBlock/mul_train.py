import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ALPHABET_SIZE = 32
EPOCHS = 2000
LR = 0.005
MODEL_NAME = "mul_model.pth"

def build_mul_onehot(mod):
    table = torch.zeros(mod, mod, mod)
    for p in range(mod):
        for k in range(mod):
            v = (p * k) % mod
            table[p, k, v] = 1.0
    return table

MUL_ONEHOT = build_mul_onehot(ALPHABET_SIZE)   # (plain, key, value)

def exact_mul_target(plain_idx, key_dist):
    table_p = MUL_ONEHOT[plain_idx]                     # (B, K, V)
    return torch.einsum("bk,bkv->bv", key_dist, table_p)


def sample_soft_distribution(batch, mod):

    concentration = torch.exp(torch.empty(batch, 1).uniform_(-2.0, 4.0)) # create empty tensor batch*1 then fill it numbers between -2 and 4 then exp result (it will be positive)
    
    alpha = concentration.expand(batch, mod) # if concentration=(2,3) and mod=4 then result (2,3) (2,3) (2,3) (2,3)
    
    return torch.distributions.Dirichlet(alpha).sample() # Dirichlet means "give you distribution sum=1" ; when alpha is small then distribution become sharp when alpha large dis become soft


def build_dataset():
    #hard
    plains = torch.arange(ALPHABET_SIZE).repeat_interleave(ALPHABET_SIZE) #generate numbers from 0 to Alph-1 and repeat it Alph time [0 0 0,1 1 1,2 2 2,3 3 3,...]
    keys   = torch.arange(ALPHABET_SIZE).repeat(ALPHABET_SIZE) #generate numbers from 0 to Alph-1 and repeat it Alph time [0 1 2 3,0 1 2 3,0 1 2 3,0 1 2 3,...]
    plain_vec = F.one_hot(plains, ALPHABET_SIZE).float() #convert plains to one hot vector
    key_vec   = F.one_hot(keys, ALPHABET_SIZE).float() #convert key to one hot vector
    target    = exact_mul_target(plains, key_vec) #compute target

    X_hard = torch.cat([plain_vec, key_vec], dim=1) #horizontal
    Y_hard = target

    #soft
    N_SOFT = 8000
    plains_soft = torch.randint(0, ALPHABET_SIZE, (N_SOFT,)) #generate 8k letter
    plain_vec_s = F.one_hot(plains_soft, ALPHABET_SIZE).float() #convert it
    key_dist_s  = sample_soft_distribution(N_SOFT, ALPHABET_SIZE) #explained
    target_s    = exact_mul_target(plains_soft, key_dist_s)

    X_soft = torch.cat([plain_vec_s, key_dist_s], dim=1)
    Y_soft = target_s

    X = torch.cat([X_hard, X_soft], dim=0) # vertical (hard first then soft after all hard done)
    Y = torch.cat([Y_hard, Y_soft], dim=0)
    return X, Y


X, Y = build_dataset()
print("Total samples =", len(X))


class MulNet(nn.Module):
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


model = MulNet()
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