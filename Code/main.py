import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch.optim as optim
import time
from sympy import Matrix
import data_util as du
import models as M
import config as con
import visualization as vis
import copy
# ===============
if __name__ == "__main__":
    MOD=con.Mod
    n = con.Dim
    epochs = con.Epoch
    batch_size = con.B_S
    lr = con.LR
    num_sample=con.num_sample
    x=con.WM
# =============== 
    K_true = du.random_invertible_key(n)
    K_inv = Matrix(K_true).inv_mod(MOD)

    P, C = du.generate_data(dim=n, Key=K_true,num_sample=num_sample)
    X_train, X_temp, Y_train, Y_temp = train_test_split(P, C, test_size=con.TEST_SIZE+con.VAL_SIZE)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.8)

    train_loader = DataLoader(du.Dataset(X_train, Y_train), batch_size, shuffle=True)
    val_loader = DataLoader(du.Dataset(X_val, Y_val), batch_size)
    test_loader = DataLoader(du.Dataset(X_test, Y_test), batch_size)
if x==1:
    model = M.FFTNetwork(n)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.NLLLoss() #Negative Log Likelihood Loss

    train_losses = []
    val_losses = []

    start_time = time.time()

    for ep in range(epochs):

        # ================== Train ==================
        model.train()
        total_train_loss = 0.0

        for xb, yb in train_loader:
            opt.zero_grad()
            
            logits, key_probs = model(xb)
            loss = loss_fn(logits.view(-1, MOD), yb.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            total_train_loss += loss.item() * xb.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # ================== Validation ==================
        model.eval()
        total_val_loss = 0.0
        correct, total = 0, 0

        with torch.no_grad():
            for xb, yb in val_loader:
                logits, _ = model(xb)

                loss = loss_fn(logits.view(-1, MOD), yb.view(-1))
                total_val_loss += loss.item() * xb.size(0)

                preds = logits.argmax(dim=-1)
                correct += (preds == yb).sum().item()
                total += yb.numel()

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        # ================== Logging ==================
        if ep % 5 == 0:
            print(
                f"\nEpoch {ep}"
                f" | Train Loss = {avg_train_loss:.4f}"
                f" | Val Loss = {avg_val_loss:.4f}"
                f" | Val Acc = {100 * correct / total:.2f}%"
            )
    # Test
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            logits, _ = model(xb)
            preds = logits.argmax(dim=-1)
            correct += (preds == yb).sum().item()
            total += yb.numel()

    print(f"\n✅ Test Accuracy = {100*correct/total:.2f}%")

    # Key 
    with torch.no_grad():
        kp = torch.softmax(model.key, dim=-1).view(n, n, MOD).numpy()
        K_learn = kp.argmax(axis=-1)
    print("original key :\n", K_true)
    print("Learned Key :\n", K_learn.T)
    print("inverseKey Key-1:\n", np.array(K_inv))
    vis.visual(loss_values=[train_losses, val_losses],graph_names=["Train Loss", "Validation Loss"],title_name="Loss vs Epochs",Xlabel="Epoch",Ylabel="Loss")
if x==2:
    
    train_losses = []
    val_losses = []
    total_train_loss=0
    total_val_loss=0
    model = M.sincosNetwork(n)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(100):
        total_loss = 0.0

        for x, y in train_loader:
            x=x.float()

            optimizer.zero_grad()

            out = model(x)          # (B*2, 26)
            y_flat = y.view(-1)     # (B*2)

            loss = loss_fn(out, y_flat)
            loss.backward()
            #for i, g in enumerate(model.groups):
            #    print(f"Grad norm of group {i}: {g.weight.grad.norm().item()}")
            optimizer.step()

            total_loss += loss.item()
            total_train_loss += loss.item() * x.size(0)
        # ================== Validation ==================

        with torch.no_grad():
            for x, y in val_loader:
                x=x.float()

                out = model(x)          # (B*2, 26)
                y_flat = y.view(-1)     # (B*2)

                loss = loss_fn(out, y_flat)

                total_loss += loss.item()
                total_val_loss += loss.item() * x.size(0)

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        if (epoch + 1) % 10 == 0:
            print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")
    vis.visual(loss_values=[train_losses, val_losses],graph_names=["Train Loss", "Validation Loss"],title_name="Loss vs Epochs",Xlabel="Epoch",Ylabel="Loss")