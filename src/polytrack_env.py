# src/polytrack_env.py
import gymnasium as gym
import numpy as np
import cv2
import time
from mss import mss
from gymnasium import spaces
from pynput.keyboard import Controller
import torch
from progress_predictor import SimpleProgressNet, predict_progress

# update GAME_REGION using find_region.py
GAME_REGION = {"top": 44, "left": 5, "width": 1460, "height": 908}
FRAME_SIZE = (84, 84)

class PolytrackEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        # 16 discrete actions (0..15) representing bitmask W/A/S/D
        self.action_space = spaces.Discrete(16)
        # observation: 4 stacked grayscale frames (4,H,W)
        self.observation_space = spaces.Box(low=0, high=255, shape=(4, FRAME_SIZE[0], FRAME_SIZE[1]), dtype=np.uint8)

        self.sct = mss()
        self.kb = Controller()
        self.frames = []
        self.start_time = None
        self.prev_progress = 0.0
        # load progress model
        self.progress_model = SimpleProgressNet()
        self.progress_model.load_state_dict(torch.load("models/progress_net.pth", map_location="cpu"))
        self.progress_model.eval()
        self.done = False

    def _grab(self):
        img = np.array(self.sct.grab(GAME_REGION))
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        small = cv2.resize(gray, FRAME_SIZE)
        return small

    def _apply_mask(self, mask):
        # mask is 0..15: bits: 1=W,2=A,4=S,8=D
        # First release all
        for k in ['w','a','s','d']:
            try: self.kb.release(k)
            except: pass
        # press required keys
        if mask & 1: self.kb.press('w')
        if mask & 2: self.kb.press('a')
        if mask & 4: self.kb.press('s')
        if mask & 8: self.kb.press('d')

    def reset(self, seed=None, options=None):
        self.done = False
        self.frames = []
        self.start_time = time.time()
        # fill initial 4 frames
        f = self._grab()
        for _ in range(4):
            self.frames.append(f)
        self.prev_progress = 0.0
        obs = np.stack(self.frames, axis=0)
        return obs, {}

    def step(self, action):
        # apply action mask
        self._apply_mask(int(action))
        # capture frame
        f = self._grab()
        self.frames.pop(0); self.frames.append(f)
        obs = np.stack(self.frames, axis=0)
        # compute progress
        prog = predict_progress(self.progress_model, obs)
        info = {"progress": float(prog), "progress_prev": float(self.prev_progress), "time": time.time() - self.start_time, "finished": False, "crash": False}
        # termination heuristics: long time or progress near 1.0
        if info["time"] > 45.0:
            info["finished"] = True
            self.done = True
        if prog > 0.995:
            info["finished"] = True
            self.done = True
        self.prev_progress = prog
        reward = 0.0  # will be shaped by the trainer wrapper
        return obs, reward, self.done, False, info
