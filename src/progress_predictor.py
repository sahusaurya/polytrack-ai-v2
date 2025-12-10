# src/progress_predictor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleProgressNet(nn.Module):
    def __init__(self):
        super().__init__()
        # expects (B,4,84,84)
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(self.conv(x))

def frames_to_tensor(frames):
    # frames: np.array (4,H,W) uint8
    x = frames.astype(np.float32) / 255.0
    x = np.expand_dims(x, 0)  # (1,4,H,W)
    return torch.from_numpy(x)

def predict_progress(model, frames):
    model.eval()
    x = frames.astype(np.float32) / 255.0
    x = torch.from_numpy(np.expand_dims(x, 0)).float()
    with torch.no_grad():
        p = model(x).item()
    return float(p)

def train_from_demos(model, demos, epochs=6, lr=1e-4, batch_size=64, device='cpu'):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = []
    Y = []
    for r in demos:
        frames = r['frames']  # (T,H,W)
        # convert to 4-frame stacks (simple consecutive 4)
        T = frames.shape[0]
        timestamps = r['timestamps']
        lap = max(1e-6, float(r.get('lap_time', timestamps[-1])))
        for i in range(3, T):
            stack = np.stack([frames[i-3], frames[i-2], frames[i-1], frames[i]], axis=0)
            frac = float(timestamps[i] / lap)
            X.append(stack)
            Y.append(frac)
    X = np.array(X).astype(np.float32) / 255.0
    Y = np.array(Y).astype(np.float32).reshape(-1,1)
    n = X.shape[0]
    idx = np.arange(n)
    for ep in range(epochs):
        np.random.shuffle(idx)
        losses = []
        for i in range(0, n, batch_size):
            b = idx[i:i+batch_size]
            xb = torch.from_numpy(X[b]).to(device)
            yb = torch.from_numpy(Y[b]).to(device)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"[progress_train] epoch {ep+1}/{epochs} loss={np.mean(losses):.6f}")
    return model
