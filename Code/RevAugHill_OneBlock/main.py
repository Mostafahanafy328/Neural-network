import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from sympy import Matrix

import data_util as du
import config as con
import models as M
import visualization as vis

MOD = con.Mod
BLOCK_SIZE = con.Dim     
EPOCHS = con.Epoch
LR = con.LR
NUM_SAMPLES = con.num_sample

# Two-phase training parameters
PHASE1_EPOCHS = 200  # Phase 1: Train with frozen models
PHASE1_LR = LR  # Normal learning rate
PHASE2_LR = LR / 100  # Slower training for the second phase 

# Dataset

# TODO: one of the visualizations is to look at the distribution of the letters in P. 
# This might be important to understand how the model is learning the distribution of the letters in P. 
P, C, K = du.generate_data(BLOCK_SIZE, NUM_SAMPLES)

P_flat = P.reshape(NUM_SAMPLES, BLOCK_SIZE * BLOCK_SIZE)
C_flat = C.reshape(NUM_SAMPLES, BLOCK_SIZE * BLOCK_SIZE)

X_train, X_temp, Y_train, Y_temp = train_test_split(
    C_flat,P_flat,
    test_size=con.TEST_SIZE + con.VAL_SIZE,
)

X_val, X_test, Y_val, Y_test = train_test_split(
    X_temp, Y_temp,
    test_size=0.5,
)

train_loader = DataLoader(du.ADataset(X_train, Y_train), batch_size=con.B_S, shuffle=True)
val_loader   = DataLoader(du.ADataset(X_val, Y_val), batch_size=con.B_S, shuffle=False)
test_loader  = DataLoader(du.ADataset(X_test, Y_test), batch_size=con.B_S, shuffle=False)

"""
def get_temperature(epoch, total_epochs):
    warmup = total_epochs * 0.1
    cooldown_start = total_epochs * 0.7

    if epoch < warmup:
        return 2.0
    elif epoch < cooldown_start:
        progress = (epoch - warmup) / (cooldown_start - warmup)
        return 2.0 - progress * 1.5
    else:
        progress = (epoch - cooldown_start) / (total_epochs - cooldown_start)
        return 0.5 - progress * 0.4
"""

def key_to_logits(key_matrix, mod, sharpness=20.0):
    n = key_matrix.shape[0]
    logits = torch.zeros(n, n, mod)
    for i in range(n):
        for j in range(n):
            val = int(key_matrix[i, j])
            logits[i, j, val] = sharpness
    return logits

K0_inv = Matrix(K[0].tolist()).inv_mod(MOD)
K0_inv_np = np.array(K0_inv.tolist(), dtype=np.int64) % MOD

# Model

criterion = nn.CrossEntropyLoss()

MAX_ATTEMPTS = 2
SUCCESS_THRESHOLD = 80

attempt = 0
success = False
Best=0

while attempt < MAX_ATTEMPTS and not success:

    attempt += 1
    print(f"\n{'='*20} Attempt {attempt} {'='*20}")

    model = M.AugmentedHillNet(n=BLOCK_SIZE, mod=MOD)
    """
    #assign correct key when start model to make sure all equations are correct

    with torch.no_grad():
        true_keys = torch.stack([
            key_to_logits(K0_inv_np, MOD),   # keys[0] = K0_inverse
            key_to_logits(K[1], MOD),         # keys[1] = K1
            key_to_logits(K[2], MOD),         # keys[2] = K2
        ], dim=0)
        model.keys.copy_(true_keys)
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=PHASE1_LR)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    phase2_started = False
    
    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        
        # # Gentle temperature annealing for frozen models
        # # Keep soft initially (high temperature), then very gently sharpen
        # progress = epoch / EPOCHS
        
        # # Schedule: Keep at 1.0 for first 50% of training
        # # Then gently descent from 1.0 to 0.3 in second 50%
        # if progress < 0.5:
        #     temperature = 1.0
        # else:
        #     # Gentle descent: 1.0 -> 0.3 over 50% of training
        #     descent_progress = (progress - 0.5) / 0.5  # 0 to 1
        #     temperature = 1.0 - (1.0 - 0.3) * descent_progress  # 1.0 to 0.3

        # Phase switching: Unfreeze individual models and lower learning rate after PHASE1_EPOCHS
        if epoch == PHASE1_EPOCHS and not phase2_started:
            print(f"SWITCHING TO PHASE 2: Unfreezing models, lowering LR to {PHASE2_LR}")
            
            # Unfreeze the models
            for net in (M.mul_model, M.add_model,M.sub_model, M.xor_model):
                for p in net.parameters():
                    p.requires_grad = True
            
            # Lower the learning rate for the current optimizer
            for param_group in optimizer.param_groups:
                param_group['lr'] = PHASE2_LR
            
            phase2_started = True

        model.train()
        #T = get_temperature(epoch, EPOCHS)
        total_loss, total_acc, n_batches = 0.0, 0.0, 0

        for X_batch, Y_batch in train_loader:

            optimizer.zero_grad()
            # out = model(X_batch, temperature=temperature) # B x n*n x MOD
            out = model(X_batch) # B x n*n x MOD
            loss = 0

            for i in range(BLOCK_SIZE * BLOCK_SIZE):
                loss += criterion(out[:, i, :], Y_batch[:, i])

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = out.argmax(dim=2)
                acc = (pred == Y_batch).float().mean().item()

            total_loss += loss.item()
            total_acc += acc
            n_batches += 1

        #scheduler.step()
        #current_lr = scheduler.get_last_lr()[0]

        avg_train_loss = total_loss / n_batches
        train_losses.append(avg_train_loss)
        Accuracy = 100 * total_acc / n_batches

        if epoch % 5 == 0:
            print(f"\nEpoch {epoch+1:03d}")
            print(f"Loss = {avg_train_loss:.4f}")
            print(f"Accuracy = {Accuracy:.2f}%")
            
            # Monitor key sharpness
            with torch.no_grad():
                for i in range(3):
                    key_probs = F.softmax(model.keys[i], dim=2)
                    max_prob = key_probs.max(dim=2).values.mean()
                    print(f"  K[{i}] confidence (mean): {max_prob:.4f}", end=" ")
                print()
        
        # -------- Validation --------
        model.eval()
        val_loss, val_acc, n_val = 0.0, 0.0, 0

        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                #out = model(X_batch,temperature=T)

                out = model(X_batch)

                loss = 0
                for i in range(BLOCK_SIZE * BLOCK_SIZE):
                    loss += criterion(out[:, i, :], Y_batch[:, i])

                pred = out.argmax(dim=2)
                acc = (pred == Y_batch).float().mean().item()

                val_loss += loss.item()
                val_acc += acc
                n_val += 1

        avg_val_loss = val_loss / n_val
        val_losses.append(avg_val_loss)
        avg_val_acc = 100 * val_acc / n_val

        if avg_val_acc >= SUCCESS_THRESHOLD:
            success = True
        if Accuracy >= Best:
            Best=Accuracy
        if Accuracy >= 98:
            print(Accuracy)
            break
    
    print(f"Validation Loss = {avg_val_loss:.4f}")
    print(f"Validation Accuracy = {avg_val_acc:.2f}%")
    print("-" * 50)
    print("Best : ",Best)
"""
with torch.no_grad():
    key_probs = F.softmax(model.keys, dim=-1)
    max_probs = key_probs.max(dim=-1).values

    print("Mean : ", max_probs.mean().item())
    print("Min:", max_probs.min().item())
    print("Max:", max_probs.max().item())
"""
with torch.no_grad():
    learned_keys = F.softmax(model.keys, dim=3).argmax(dim=3)

print("\n========== Keys Comparison ==========")
K[0]=K0_inv_np #because k0 is k0 inverse in decryption
for i in range(3):
    print(f"\nTrue K[{i}]")
    print(K[i])
    print(f"Learned K[{i}]")
    print(learned_keys[i].numpy())
    
    # Calculate accuracy of key matching
    key_match = (K[i] == learned_keys[i].numpy()).mean() * 100
    print(f"Key Match Accuracy: {key_match:.2f}%")
    
    # Check key confidence (how sharp are the softmax distributions?)
    key_probs = F.softmax(model.keys[i], dim=2)
    max_probs = key_probs.max(dim=2).values
    print(f"Key confidence (mean max prob): {max_probs.mean():.4f} (higher = sharper)")
    print(f"Key confidence (min): {max_probs.min():.4f}, max: {max_probs.max():.4f}")

# Test

model.eval()
test_loss, test_acc, n_test = 0.0, 0.0, 0

with torch.no_grad():
    for X_batch, Y_batch in test_loader:
        out = model(X_batch)

        loss = 0
        for i in range(BLOCK_SIZE * BLOCK_SIZE):
            loss += criterion(out[:, i, :], Y_batch[:, i])

        pred = out.argmax(dim=2)
        acc = (pred == Y_batch).float().mean().item()

        test_loss += loss.item()
        test_acc += acc
        n_test += 1

print("\n========== Test ==========")
print(f"Loss = {test_loss/n_test:.4f}")
print(f"Accuracy = {100*test_acc/n_test:.2f}%")

# Plot

vis.visual(
    loss_values=[train_losses, val_losses],
    graph_names=["Train Loss", "Validation Loss"],
    title_name="Loss vs Epochs",
    Xlabel="Epoch",
    Ylabel="Loss"
)