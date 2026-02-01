import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset

# ------------------ Mod26 Model ------------------
class Mod26Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 26)
        )

    def forward(self, x):
        return self.net(x)

# ------------------ Dataset ------------------
class Mod26Dataset(Dataset):
    def __init__(self, n_samples=5000):
        self.X = np.random.randint(-1000, 3000, size=(n_samples,))
        self.Y = self.X % 26

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        features = np.array([
            x / 2028,
            np.sin(x * 2 * np.pi / 26),
            np.cos(x * 2 * np.pi / 26)
        ], dtype=np.float32)
        return torch.tensor(features), torch.tensor(self.Y[idx], dtype=torch.long)

# ------------------ Training ------------------
if __name__ == "__main__":
    model = Mod26Model()
    dataset = Mod26Dataset()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(100):
        total_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"[Epoch {epoch + 1}] Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "mod26_frozen.pth")
    print("\n✅ Mod26 Model Saved ✅")
