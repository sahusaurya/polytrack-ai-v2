# src/train_progress_predictor.py
import glob, pickle
from progress_predictor import SimpleProgressNet, train_from_demos
import torch

def main():
    paths = sorted(glob.glob("data/run_*.pkl"))
    demos = []
    for p in paths:
        with open(p, "rb") as f:
            demos.append(pickle.load(f))
    print(f"[info] Loaded {len(demos)} demos")
    model = SimpleProgressNet()
    import os
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/progress_net.pth")
    print("[info] Saved models/progress_net.pth")


if __name__ == "__main__":
    main()
