# Polytrack AI — Autonomous Racing Agent (Imitation Learning + Reinforcement Learning)

This project implements a full end-to-end autonomous agent capable of driving and completing tracks in the PC game Polytrack using only screen pixels and W/A/S/D keyboard inputs. The system learns from human demonstrations and improves beyond them using reinforcement learning.

The agent uses:
- Human demonstration recording
- Behavior Cloning (supervised imitation learning)
- A self-supervised Progress Predictor network
- PPO reinforcement learning with reward shaping
- A custom Gymnasium-compatible environment
- Real-time screen capture and keyboard simulation

The final result is an AI that can independently race the track and optimize its lap time.

---

## Features

### 1. Human Demonstration Recording
- Records screen frames in real time.
- Captures all W/A/S/D combinations using a 4-bit action mask (16 actions).
- Includes hotkeys:
  - T = restart current run
  - F = save completed run
- Saves timestamps, frames, actions, and lap times.

### 2. Progress Predictor (Self-Supervised)
Trains a CNN model to estimate how far along the track the car is (0 to 1).  
This provides dense reward for the RL agent, solving the sparse-reward problem.

### 3. Behavior Cloning (Imitation Learning)
A supervised neural network that learns to imitate human driving from demonstration data.  
Produces a baseline model: `bc_model.pth`.

### 4. Mixed Reinforcement Learning (BC Warm-Start + PPO)
PPO agent warm-starts from the BC model and learns to improve beyond human demonstrations using shaped rewards derived from the progress predictor.

### 5. Real-Time Evaluation
Test script loads the trained PPO model and drives Polytrack automatically.

---

## Repository Structure
```
polytrack-ai/
│
├── src/
│ ├── record_demo.py
│ ├── progress_predictor.py
│ ├── train_progress_predictor.py
│ ├── behavior_cloning.py
│ ├── polytrack_env.py
│ ├── train_mixed.py
│ ├── test_agent.py
│ ├── reward_shaping.py
│ └── init.py
│
├── data/ # human demos (excluded from Git)
├── models/ # trained models (.pth, .zip)
│
├── requirements.txt
├── .gitignore
└── README.md
```
---

## Installation

Create and activate a virtual environment:
python3 -m venv venv_polytrack
source venv_polytrack/bin/activate

Install dependencies:
```
pip install -r requirements.txt
```
---

## Training Pipeline

### Step 1 — Record Human Demonstrations
```
python src/record_demo.py
```
Controls:
- Use W / A / S / D to drive (Down Arrow acts as S)
- Press T to restart the current run
- Press F to save a completed run

Each saved run becomes:  
`data/run_XXX.pkl`

Record at least 20 runs (50 recommended, used the same in personal training session too).

---

### Step 2 — Train Progress Predictor
```
python src/train_progress_predictor.py
```
Outputs:
- `progress_net.pth`

This model estimates track progress from frame stacks.

---

### Step 3 — Train Behavior Cloning Model
```
python src/behavior_cloning.py
```
Outputs:
- `bc_model.pth`

This gives the PPO agent a good initialization.

---

### Step 4 — Mixed RL Training (BC + PPO)

Start Polytrack and ensure:
- The game window is fully visible
- The window is not minimized
- The window position matches the configured capture region

Then run:
```
python src/train_mixed.py
```
Outputs:
- `ppo_mixed.zip` (final trained agent)

---

### Step 5 — Test the Trained Agent
```
python src/test_agent.py
```
This script loads `ppo_mixed.zip` and drives the game automatically.

---

## Notes

- The game must remain visible during RL training and testing.
- Do not move or resize the game window after recording demos.

---

## Requirements

See `requirements.txt` for full dependency list.

---

## License

MIT License. You may use, modify, and distribute this project with attribution.





