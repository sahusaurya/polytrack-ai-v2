# src/reward_shaping.py
def compute_shaped_reward(progress0, progress1, base_step_penalty=-1.0, C=50.0, gamma=0.99, info=None):
    shaping = gamma * C * progress1 - C * progress0
    crash_penalty = -200.0 if (info and info.get('crash', False)) else 0.0
    finish_bonus = 0.0
    if info and info.get('finished', False):
        lap_time = info.get('lap_time', None)
        if lap_time is not None:
            finish_bonus = 1000.0 - float(lap_time)
        else:
            finish_bonus = 500.0
    return base_step_penalty + shaping + crash_penalty + finish_bonus
