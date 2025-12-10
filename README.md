# Polytrack AI — Autonomous Racing Agent (Imitation Learning + Reinforcement Learning)

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-ML-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()

Polytrack AI is a full end-to-end autonomous driving system built for the PC racing game Polytrack.  
The agent learns using only screen pixels and issues real-time W/A/S/D key presses to control the car.

The system integrates:
- Human demonstration capture  
- Behavior Cloning (imitation learning)  
- A self-supervised progress predictor  
- PPO reinforcement learning  
- Real-time screen capture  
- Reward shaping  
- Autonomous evaluation

The final trained agent can drive the track independently and improve beyond human performance.

---

## 1. Project Architecture Overview

Pipeline:

Human Gameplay → Demo Dataset → Progress Predictor → Behavior Cloning → Mixed RL Training → Autonomous Agent

### Components:
1. **Demonstration Recorder**  
   Captures frames, timestamps, and multi-key actions.

2. **Progress Predictor (CNN)**  
   Learns to estimate 0.0–1.0 race progress from raw frames.

3. **Behavior Cloning**  
   Warm-starts the PPO policy using supervised learning.

4. **Custom Gymnasium Environment**  
   Uses screen capture and keyboard simulation.

5. **Mixed BC + PPO Reinforcement Learning**  
   Combines BC initialization with shaped progress rewards.

6. **Evaluation Script**  
   Drives the game automatically using the final PPO model.

---

## 2. Features

### Human Demonstration Recording
- Captures real screen frames  
- Encodes multi-key actions as a 4-bit mask (16 possible actions)  
- Hotkeys:
  - `T` = restart run  
  - `F` = save run  

### Progress Predictor Network
A self-supervised CNN that estimates how far along the track the agent is.  
This gives PPO dense and meaningful rewards.

### Behavior Cloning
Supervised learner that imitates human gameplay and gives PPO a strong starting policy.

### Mixed Reinforcement Learning
PPO learns to optimize lap time with:
- BC warm-start  
- Progress-shaped rewards  
- Screen-based observations  

### Agent Evaluation
A script that loads the PPO model and controls the game live.

---

## 3. Repository Structure

```
polytrack-ai-v2/
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
├── .gitignore
├── requirements.txt
└── README.md
```
---

## 4. Installation

Create virtual environment:

python3 -m venv venv_polytrack
source venv_polytrack/bin/activate

Install dependencies:

pip install -r requirements.txt

---

## 5. Training Pipeline

### Step 1 — Record Demonstrations

python src/record_demo.py

Controls:
- Drive with W/A/S/D  
- Down Arrow = S  
- `T` = restart  
- `F` = save run  

Record 20–50 runs.

---

### Step 2 — Train Progress Predictor

python src/train_progress_predictor.py

Output:
models/progress_net.pth

---

### Step 3 — Train Behavior Cloning Model

python src/behavior_cloning.py

Output:
models/bc_model.pth

---

### Step 4 — Mixed BC + PPO Reinforcement Learning

Polytrack must be open and visible.  
Window position and size must match the demo recording.

python src/train_mixed.py

Output:
models/ppo_mixed.zip

---

### Step 5 — Evaluate the Trained Model

python src/test_agent.py

The agent will drive autonomously.

---

## 6. Requirements

Dependencies (see requirements.txt):
- PyTorch  
- Stable-Baselines3  
- Gymnasium  
- OpenCV  
- mss  
- pynput  
- numpy  
- pillow  
- pyautogui  

---

## 7. Notes

- The game window must remain visible during RL training and evaluation.  
- All demos and trained models are ignored via `.gitignore`.  
- Changing resolution or window position will break screen capture.  

---

## 8. License

MIT License.

---

## 9. Acknowledgments

Inspired by:
- Reinforcement learning for autonomous driving  
- Reward shaping with progress estimation  
- Behavior cloning for policy initialization  
- Vision-based RL pipelines  
