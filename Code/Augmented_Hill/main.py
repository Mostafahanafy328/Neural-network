import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch.optim as optim
import time

import data_util as du
import models as M
import config as con
import visualization as vis

if __name__ == "__main__":

    MOD = con.Mod
    n = con.Dim
    block_size = con.BlockSize

    epochs = con.Epoch
    batch_size = con.B_S
    lr = con.LR
    num_sample = con.num_sample

# Generate Dataset

    C,P, keys = du.generate_data(
        n=n,
        block_size=block_size,
        num_samples=num_sample
    )

    K_true = keys

# Train / Val / Test Split

    X_train, X_temp, Y_train, Y_temp = train_test_split(
        C,P,
        test_size=con.TEST_SIZE + con.VAL_SIZE
    )

    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp,
        test_size=0.8
    )

# DataLoader

    train_loader = DataLoader(
        du.ADataset(X_train, Y_train),
        batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        du.ADataset(X_val, Y_val),
        batch_size
    )

    test_loader = DataLoader(
        du.ADataset(X_test, Y_test),
        batch_size
    )

    """
    def key_supervision_loss(K_probs, K_true):
        loss = 0
        for i in range(3):
            for r in range(n):
                for c in range(n):
                    loss += -torch.log(
                        K_probs[i, r, c, K_true[i][r][c]] + 1e-12
                    )
        return loss / (3 * n * n)
    """

    loss_fn = nn.NLLLoss()

    trying=0
    Val_acc=0
    # Training
    
    while Val_acc<90:
        model = M.AugmentedHillModel(n)

        opt = optim.Adam(model.parameters(), lr=lr)

        trying+=1
        train_losses = []
        val_losses = []

        for ep in range(epochs):

            # ================== Train ==================

            model.train()
            total_train_loss = 0.0

            for xb, yb in train_loader:

                opt.zero_grad()

                logits, K_probs,K_inv_probs = model(xb)
                
                loss_p = loss_fn(
                    logits.view(-1, MOD),
                    yb.view(-1)
                )


                loss = loss_p

                #OR 

                #inv_loss = M.inverse_constraint_loss(K_probs, K_inv_probs,2)
                
                #loss = loss_p + 0.001* inv_loss

                #OR

                #key_loss=key_supervision_loss(K_probs,K_true)
                
                #loss = loss_p + 0.01 * inv_loss + 0.1 * key_loss

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 5.0
                )

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

                    logits, _,_ = model(xb)

                    loss = loss_fn(
                        logits.view(-1, MOD),
                        yb.view(-1)
                    )

                    total_val_loss += loss.item() * xb.size(0)

                    preds = logits.argmax(dim=-1)
                    
                    yb_flat = yb.view(yb.size(0), yb.size(1), -1)

                    correct += (preds == yb_flat).sum().item()
                    total += yb_flat.numel()
                    
            avg_val_loss = total_val_loss / len(val_loader.dataset)

            val_losses.append(avg_val_loss)

            Val_acc=100 * correct / total
            if ep % 25 == 0:

                print(
                    f"\nEpoch {ep}"
                    f" | Trying = {trying:d}"
                    f" | Train Loss = {avg_train_loss:.4f}"
                    f" | Val Loss = {avg_val_loss:.4f}"
                    f" | Val Acc = {Val_acc:.2f}%"
                    
                )
                if Val_acc>99:
                    break

    # Test

    model.eval()

    correct, total = 0, 0

    with torch.no_grad():

        for xb, yb in test_loader:

            logits, _ ,_= model(xb)

            preds = logits.argmax(dim=-1)

            yb_flat = yb.view(yb.size(0), yb.size(1), -1)

            correct += (preds == yb_flat).sum().item()
            total += yb_flat.numel()

    print(f"\n Test Accuracy = {100*correct/total:.2f}%")

    # Recover Keys

    with torch.no_grad():

        K_learn = K_probs.argmax(dim=-1).numpy()


    print("\nTrue Keys:")

    for i in range(3):
        print(f"K{i}:\n", K_true[i])


    print("\nRecovered Keys:")

    for i in range(3):
        print(f"K{i} learned:\n", K_learn[i])

    # Plot

    vis.visual(
        loss_values=[train_losses, val_losses],
        graph_names=["Train Loss", "Validation Loss"],
        title_name="Loss vs Epochs",
        Xlabel="Epoch",
        Ylabel="Loss"
    )
