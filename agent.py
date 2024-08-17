from tensordict import TensorDict
import torch
import numpy as np
from agent_nn import AgentNN
from disk_replay_buffer import DiskReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import psutil
import threading
import time
import os
import gym

# TensorBoard writer
writer = SummaryWriter(log_dir='runs/mario_experiment_1')

class Agent:
    def __init__(self, input_dims, num_actions):
        self.num_actions = num_actions
        self.learn_step_counter = 0
        self.episode_rewards = []
        self.episode_scores = []

        # Hyperparameters
        self.lr = 0.0005
        self.gamma = 0.95
        self.epsilon = 1.0
        self.eps_decay = 0.9999  # Updated decay rate
        self.eps_min = 0.1
        self.batch_size = 128
        self.sync_network_rate = 10_000  # Updated target network sync rate

        # Networks
        self.online_network = AgentNN(input_dims, num_actions)
        self.target_network = AgentNN(input_dims, num_actions, freeze=True)

        # Optimizer and loss
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=self.lr)
        self.loss = torch.nn.MSELoss()

        # Replay Buffer
        script_dir = os.path.dirname(os.path.abspath(__file__))
        storage_path = os.path.join(script_dir, 'replay_buffer_storage')
        os.makedirs(storage_path, exist_ok=True)

        replay_buffer_capacity = 100_000
        self.replay_buffer = DiskReplayBuffer(directory=storage_path, max_size=replay_buffer_capacity)

        # Start system resource monitoring in a separate thread
        self.resource_monitor_thread = threading.Thread(target=self.log_system_resources, daemon=True)
        self.resource_monitor_thread.start()

    def log_system_resources(self, interval=1):
        while True:
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent
            print(f"CPU Usage: {cpu_usage}%")
            print(f"Memory Usage: {memory_usage}%")
            writer.add_scalar('System/CPU_Usage', cpu_usage, self.learn_step_counter)
            writer.add_scalar('System/Memory_Usage', memory_usage, self.learn_step_counter)
            time.sleep(interval)

    def save_model(self, path):
        torch.save(self.online_network.state_dict(), path)

    def load_model(self, path):
        self.online_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(torch.load(path))

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        observation = torch.tensor(np.array(observation), dtype=torch.float32) \
                            .unsqueeze(0) \
                            .to(self.online_network.device)
        return self.online_network(observation).argmax().item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

    def store_in_memory(self, state, action, reward, next_state, done):
        try:
            state = np.array(state, dtype=np.float32)
            next_state = np.array(next_state, dtype=np.float32)
        except ValueError as e:
            raise ValueError(f"Error converting state or next_state to numpy array: {e}")

        if not np.issubdtype(state.dtype, np.number) or not np.issubdtype(next_state.dtype, np.number):
            raise ValueError("State or next_state contains non-numeric data.")

        if state.shape != next_state.shape:
            raise ValueError("State and next_state must have the same shape.")

        self.replay_buffer.add(TensorDict({
            "state": torch.tensor(np.array(state), dtype=torch.float32),
            "action": torch.tensor(action),
            "reward": torch.tensor(reward),
            "next_state": torch.tensor(np.array(next_state), dtype=torch.float32),
            "done": torch.tensor(done)
            }, batch_size=[]))

    def sync_networks(self):
        if self.learn_step_counter % self.sync_network_rate == 0 and self.learn_step_counter > 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

    def learn(self):
        if len(self.replay_buffer.buffer) < self.batch_size:
            return
        
        self.sync_networks()
        self.optimizer.zero_grad()

        experiences = self.replay_buffer.sample(self.batch_size)

        # Unpack experiences assuming it's a tuple of tensors
        states, actions, rewards, next_states, dones = experiences

        # Move tensors to the device
        states = states.to(self.online_network.device)
        actions = actions.to(self.online_network.device)
        rewards = rewards.to(self.online_network.device)
        next_states = next_states.to(self.online_network.device)
        dones = dones.to(self.online_network.device)

        predicted_q_values = self.online_network(states)
        predicted_q_values = predicted_q_values[range(self.batch_size), actions.squeeze()]

        target_q_values = self.target_network(next_states).max(dim=1)[0]
        target_q_values = rewards + self.gamma * target_q_values * (1 - dones.float())

        loss = self.loss(predicted_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        self.decay_epsilon()

        # Log training progress to TensorBoard
        writer.add_scalar('Loss', loss.item(), self.learn_step_counter)
        writer.add_scalar('Epsilon', self.epsilon, self.learn_step_counter)

        # Print out learning step count and loss
        print("********************************i want to learn now********************")

        print(f"Learning Step: {self.learn_step_counter}, Loss: {loss.item()}")

        return loss.item()

    def train(self, num_episodes):
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            episode_score = 0

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, info = self.env.step(action)
                
                # Update episode reward and score
                episode_reward += reward
                episode_score = info.get('score', episode_score)  # Update with environment score if available
                print(f"Episode: {episode}, Episode Reward: {episode_reward:.2f}, Episode Score: {episode_score:.2f}")

                self.store_in_memory(state, action, reward, next_state, done)
                self.learn()
                state = next_state

            self.episode_rewards.append(episode_reward)
            self.episode_scores.append(episode_score)

            # Calculate average reward and score
            average_reward = np.mean(self.episode_rewards)
            average_score = np.mean(self.episode_scores)

            # Log metrics to TensorBoard and terminal
            writer.add_scalar('Episode Reward', episode_reward, episode)
            writer.add_scalar('Episode Score', episode_score, episode)
            writer.add_scalar('Average Reward', average_reward, episode)
            writer.add_scalar('Average Score', average_score, episode)

            print(f"Episode: {episode}, Episode Reward: {episode_reward:.2f}, Episode Score: {episode_score:.2f}")
            print(f"Average Reward: {average_reward:.2f}, Average Score: {average_score:.2f}")

    def close(self):
        writer.close()
        self.env.close()
