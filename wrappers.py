import numpy as np
from gym import Wrapper
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack

class SkipFrame(Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self.skip):
            next_state, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return next_state, total_reward, done, truncated, info
    
def apply_wrappers(env):
    env = SkipFrame(env, skip=4)  # number of frames to apply action to
    env = ResizeObservation(env, shape=84)
    env = GrayScaleObservation(env)
    env = FrameStack(env, num_stack=4, lz4_compress=True)
    return env
