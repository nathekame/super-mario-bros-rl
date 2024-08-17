import dill
import os
import random
from tensordict import TensorDict
import torch

def load_tensor_dict(file_path):
    # Open the file and load the TensorDict using dill
    with open(file_path, 'rb') as f:
        tensor_dict = dill.load(f)
    return tensor_dict

def extract_and_stack(batch_files):
    # Load data from each file and extract tensors
    states_list = []
    actions_list = []
    rewards_list = []
    next_states_list = []
    dones_list = []
    
    for file_path in batch_files:
        tensor_dict = load_tensor_dict(file_path)
        
        # Extract tensors and append to lists
        states_list.append(tensor_dict['state'])
        actions_list.append(tensor_dict['action'])
        rewards_list.append(tensor_dict['reward'])
        next_states_list.append(tensor_dict['next_state'])
        dones_list.append(tensor_dict['done'])
    
    # Stack tensors to form batch tensors
    states = torch.stack(states_list)
    actions = torch.stack(actions_list)
    rewards = torch.stack(rewards_list)
    next_states = torch.stack(next_states_list)
    dones = torch.stack(dones_list)
    
    return states, actions, rewards, next_states, dones

def count_files_in_buffer_storage(folder_path):
    try:
        # List all entries in the folder
        entries = os.listdir(folder_path)
        # Count files (excluding directories)
        file_count = sum(1 for entry in entries if os.path.isfile(os.path.join(folder_path, entry)))
        return file_count
    except FileNotFoundError:
        print("The folder does not exist.")
        return 0
    except PermissionError:
        print("You do not have permission to access this folder.")
        return 0

def get_files_list_in_buffer_storage(folder_path):
    try:
        # List all entries in the folder
        entries = os.listdir(folder_path)
        # Filter out directories and keep only files
        files = [entry for entry in entries if os.path.isfile(os.path.join(folder_path, entry))]
        return files
    except FileNotFoundError:
        print("The folder does not exist.")
        return []
    except PermissionError:
        print("You do not have permission to access this folder.")
        return []

class DiskReplayBuffer:
    def __init__(self, directory, max_size):
        self.directory = directory
        self.max_size = max_size
        self.buffer = []
        self.position = 0  # Initialize position for circular buffer

        if not os.path.exists(directory):
            os.makedirs(directory)

    def add(self, experience):
        pklCount = count_files_in_buffer_storage(self.directory)

        filename = os.path.join(self.directory, f"experience_{len(self.buffer)}.pkl")

        if pklCount < self.max_size:
            with open(filename, "wb") as f:
               dill.dump(experience, f)
            self.buffer.append(filename)
        else:
            oldest_experience_path = self.buffer[self.position]
            os.remove(oldest_experience_path)
            new_filename = os.path.join(self.directory, f"experience_{self.position}.pkl")
            with open(new_filename, "wb") as f:
                dill.dump(experience, f)
            self.buffer[self.position] = new_filename
        self.position = (self.position + 1) % self.max_size

    def sample(self, batch_size):
        batch_files = [os.path.join(self.directory, f) for f in os.listdir(self.directory) if os.path.isfile(os.path.join(self.directory, f))]
        sampled_files = random.sample(batch_files, batch_size)
        return extract_and_stack(sampled_files)
