# src/behavior_cloning.py
import glob, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class BCDataset(Dataset):
    def __init__(self, demo_paths):
        X=[]; Y=[]
        for p in demo_paths:
            r = pickle.load(open(p,'rb'))
            frames = r['frames']  # (T,H,W)
            acts = r['actions']   # (T,)
            T = frames.shape[0]
            for i in range(3, T):
                stack = np.stack([frames[i-3], frames[i-2], frames[i-1], frames[i]], axis=0)
                X.append(stack)
                Y.append(int(acts[i]))
        self.X = np.array(X).astype(np.float32) / 255.0
        self.Y = np.array(Y).astype(np.int64)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]

class BCNet(nn.Module):
    def __init__(self, n_actions=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4,32,8,4), nn.ReLU(),
            nn.Conv2d(32,64,4,2), nn.ReLU(),
            nn.Conv2d(64,64,3,1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(64*7*7, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, n_actions)
        )
    def forward(self, x):
        return self.fc(self.conv(x))

def main():
    paths = sorted(glob.glob("data/run_*.pkl"))
    ds = BCDataset(paths)
    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)
    device = torch.device("cpu")
    model = BCNet(n_actions=16).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    for epoch in range(8):
        losses=[]
        accs=[]
        for xb,yb in loader:
            xb=xb.to(device); yb=yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
            pred = logits.argmax(dim=1)
            accs.append((pred==yb).float().mean().item())
        print(f"[BC] epoch {epoch+1} loss={np.mean(losses):.4f} acc={np.mean(accs):.4f}")
    import os
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/bc_model.pth")
    print("[BC] saved models/bc_model.pth")


if __name__ == "__main__":
    main()
