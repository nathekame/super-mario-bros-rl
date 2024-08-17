import torch
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers
from agent import Agent
import os
from utils import get_current_date_time_string
import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Paths and setup
model_path = os.path.join("models")
os.makedirs(model_path, exist_ok=True)

if torch.cuda.is_available():
    print("Using CUDA device: ", torch.cuda.get_device_name(0))
else:
    print("CUDA is available")

ENV_NAME = 'SuperMarioBros-1-1-v3'
SHOULD_TRAIN = True
DISPLAY = True
CKPT_SAVE_INTERVAL = 100
NUM_OF_EPISODES = 50_000
TARGET_UPDATE_INTERVAL = 10_000  # Update target network every 10k steps

try:
    env = gym_super_mario_bros.make(ENV_NAME, apply_api_compatibility=True, render_mode="rgb_array")
    # env = gym_super_mario_bros.make(ENV_NAME, apply_api_compatibility=True, render_mode="human")

    print(env.metadata)
    print(f"{ENV_NAME} is available.")
except gym.error.Error as e:
    print(f"{ENV_NAME} is not available. Error: {e}")
    exit()

# Reset the environment
env.reset()
env = JoypadSpace(env, RIGHT_ONLY)
env = apply_wrappers(env)

# Initialize the agent
agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)
agent.target_update_rate = TARGET_UPDATE_INTERVAL  # Set target update interval

if not SHOULD_TRAIN:
    folder_name = get_current_date_time_string()
    ckpt_name = "model_" + "iter.pt"
    agent.load_model(os.path.join("models", ckpt_name))
    agent.epsilon = 1.0
    agent.eps_decay = 0.9999

# TensorBoard writer
writer = SummaryWriter(log_dir='runs/mario_experiment_1')

total_rewards = []
total_scores = []

for episode in range(NUM_OF_EPISODES):
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        print(info.get('score'))

        if SHOULD_TRAIN:
            agent.store_in_memory(state, action, reward, next_state, done)
            agent.learn()
        
        state = next_state
        total_reward += reward

        episode_score = info.get('score')  # Update with environment score if available

        if SHOULD_TRAIN and (episode + 1) % CKPT_SAVE_INTERVAL == 0:
            print("********************************i want to save the model now********************")
            agent.save_model(os.path.join(model_path, "model_" + str(episode + 1) + "_iter.pt"))

        print(f"Current Episode: {episode + 1}, Episode Reward: {total_reward:.2f}, Episode Score: {episode_score:.2f}")
        print("Total reward:", total_reward, "Epsilon:", agent.epsilon, "Learn step counter:", agent.learn_step_counter)
        print(f"Episode {episode + 1}/{NUM_OF_EPISODES} finished with total reward: {total_reward}")

        total_rewards.append(total_reward)
        total_scores.append(episode_score)

        # Calculate average reward and score
        average_reward = np.mean(total_rewards)
        average_score = np.mean(total_scores)

        # Log metrics to TensorBoard and terminal
        writer.add_scalar('Episode Reward', total_reward, episode)
        writer.add_scalar('Episode Score', episode_score, episode)
        writer.add_scalar('Average Reward', average_reward, episode)
        writer.add_scalar('Average Score', average_score, episode)

        print(f"Episode: {episode}, Episode Reward: {total_reward:.2f}, Episode Score: {episode_score:.2f}")
        print(f"Average Reward: {average_reward:.2f}, Average Score: {average_score:.2f}")

    env.render()
    # env.close()
