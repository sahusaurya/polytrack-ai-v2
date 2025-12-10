# src/record_demo.py
import time
import pickle
import cv2
import numpy as np
import os
from mss import mss
from pynput import keyboard

# Configuration - adjust GAME_REGION using find_region.py as described earlier
GAME_REGION = {"top": 100, "left": 100, "width": 1280, "height": 720}
FRAME_SIZE = (84, 84)
SAVE_DIR = "data"
os.makedirs(SAVE_DIR, exist_ok=True)

# Internal state
recording = False
pressed = set()
frames = []
timestamps = []
actions = []
run_index = 0
start_time = None
sct = mss()

# Map pressed keys into 4-bit action (W,A,S,D)
def keys_to_action_mask(pressed_set):
    # bit order: W (1), A (2), S (4), D (8)
    mask = 0
    if 'W' in pressed_set: mask |= 1
    if 'A' in pressed_set: mask |= 2
    if 'S' in pressed_set: mask |= 4
    if 'D' in pressed_set: mask |= 8
    return mask

def grab_frame():
    img = np.array(sct.grab(GAME_REGION))
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    small = cv2.resize(gray, FRAME_SIZE)
    return small

def reset_run():
    global recording, frames, timestamps, actions, start_time
    recording = True
    frames = []
    timestamps = []
    actions = []
    start_time = time.time()
    print("[RECORDER] New run started")

def save_run():
    global run_index, frames, timestamps, actions, recording
    if len(frames) < 40:
        print("[RECORDER] Run too short; discarding.")
        reset_run()
        return
    run = {
        "frames": np.stack(frames),          # (T, H, W)
        "timestamps": np.array(timestamps),  # (T,)
        "actions": np.array(actions, dtype=np.uint8), # (T,) each 0..15
        "lap_time": float(timestamps[-1]) if len(timestamps) else 0.0
    }
    fname = os.path.join(SAVE_DIR, f"run_{run_index:03d}.pkl")
    with open(fname, "wb") as f:
        pickle.dump(run, f)
    print(f"[RECORDER] Saved run -> {fname} (frames={len(frames)})")
    run_index += 1
    reset_run()

def on_press(key):
    global pressed
    try:
        ch = key.char.lower()
    except AttributeError:
        ch = None
    if key == keyboard.KeyCode.from_char('t'):
        # Restart the run buffer (game restart presumably)
        print("[RECORDER] T pressed: restart run")
        reset_run()
        return
    if key == keyboard.KeyCode.from_char('f'):
        # Finish and save
        if recording:
            print("[RECORDER] F pressed: save run")
            save_run()
        return
    if ch in ('w','a','s','d'):
        pressed.add(ch.upper())
    # down arrow maps to S
    if key == keyboard.Key.down:
        pressed.add('S')

def on_release(key):
    try:
        ch = key.char.lower()
    except AttributeError:
        ch = None
    if ch in ('w','a','s','d'):
        try: pressed.remove(ch.upper())
        except: pass
    if key == keyboard.Key.down:
        try: pressed.remove('S')
        except: pass

def main():
    print("Recorder: Press T to restart a run, F to save current run after finish.")
    print("Controls recorded: W/A/S/D and DownArrow->S.")
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    print("Waiting for first T to start recording...")

    # main loop
    while True:
        if recording:
            frame = grab_frame()
            frames.append(frame)
            timestamps.append(time.time() - start_time)
            mask = keys_to_action_mask(pressed)
            actions.append(mask)
        time.sleep(0.02)  # 50Hz

if __name__ == "__main__":
    main()
