#!/usr/bin/env python3
from python_gazebo import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import os
import shutil
import math
import random

# ---------------- Global Parameters ----------------

RESTORE_CKPT = True

# ----- SIMULATION PARAMETERS -----
# Reinforcement Learning:
EPISODES = 10000
MAX_STEPS = 10000  # per episode
GAMMA = 0.999      # future reward discount
EXPLORE_MAX = 1.0  # exploration probability
EXPLORE_MIN = 0.1
DECAY_PERCENT = 0.5
DECAY_RATE = 4 / DECAY_PERCENT
LEARN_RATE = 1e-4
STEP_RWD = -0.1
EXIT_RWD = 0

# Camera:
CAM_HEIGHT = 8
CAM_WIDTH = 8
CHANNELS = 3
INPUT_SIZE = CAM_HEIGHT * CAM_WIDTH * CHANNELS

# Batch Memory:
MEM_SIZE = 1000
BATCH_SIZE = 50

# Target Q-Network:
UPDATE_TARGET_EVERY = 1
TAU = 0.1  # target update factor
TRAIN_STEP = 1

SAVE_STEP = 100
CKPT_PATH = './tf_checkpoints/camera'
CKPT_NAME = 'camera_test.pth'  # using PyTorch extension

# Environment (same as before)
ENV_SIZE = 8 * FOOT
AGENT_SIZE = 0.1262
STEP_SIZE = 0.5 * FOOT
EXIT = [0, 4 * FOOT]
EXIT_WIDTH = 0.5

# ---------------- Environment Class (Unchanged) ----------------

class Environment:
    def __init__(self):
        self.jetbot = PythonGazebo(ros_rate=25)

        self.x = 0
        self.y = 0
        self.heading = 0

        self.last_x = 0
        self.last_y = 0
        self.last_heading = 0

        self.action_space = [np.pi/2, 3*np.pi/4, np.pi, -3*np.pi/4,
                             -np.pi/2, -np.pi/4, 0., np.pi/4]

    def reset(self):
        x_target = np.random.uniform(-ENV_SIZE/2 + AGENT_SIZE, ENV_SIZE/2 - AGENT_SIZE)
        y_target = np.random.uniform(-ENV_SIZE/2 + AGENT_SIZE, ENV_SIZE/2 - AGENT_SIZE)
        head_target = self.action_space[np.random.randint(0, 8)]

        if not self.teleport_to(x_target, y_target, head_target):
            print("collision on reset, trying again...")
            return self.reset()

        self.last_x = self.x
        self.last_y = self.y
        self.last_heading = self.heading

        return self.jetbot.get_raw_image()

    def step(self, action):
        image = self.jetbot.get_raw_image()

        head_target = self.action_space[action]
        x_target = self.x + STEP_SIZE * math.cos(head_target)
        y_target = self.y + STEP_SIZE * math.sin(head_target)

        if not self.teleport_to(x_target, y_target, head_target):
            return image, STEP_RWD, False, True

        image = self.jetbot.get_raw_image()

        if self.exited(self.x, self.y):
            print("RESULT: exited")
            return image, EXIT_RWD, True, True

        if self.out_of_bounds(x_target, y_target):
            return image, STEP_RWD, False, False

        return image, STEP_RWD, False, False

    def teleport_to(self, x, y, heading):
        x_result, y_result, head_result = self.jetbot.teleport_to(x, y, heading)

        x_err = abs(x - x_result)
        y_err = abs(y - y_result)
        head_err = abs(normalize(heading - head_result))

        if abs(x_result - self.x) + abs(y_result - self.y) < 0.01:
            print("LAG WARNING: ros_rate may be set too high. (1) ")

        if x_err > 0.01 or y_err > 0.01 or head_err > 0.01:
            x_result, y_result, head_result = self.jetbot.teleport_to(self.x, self.y, self.heading)

            x_err = abs(self.x - x_result)
            y_err = abs(self.y - y_result)
            head_err = abs(normalize(self.heading - head_result))

            if x_err > 0.01 or y_err > 0.01 or head_err > 0.01:
                x_result, y_result, head_result = self.jetbot.teleport_to(self.last_x, self.last_y, self.last_heading)

                x_err = abs(self.x - x_result)
                y_err = abs(self.y - y_result)
                head_err = abs(normalize(self.heading - head_result))

                if x_err > 0.01 or y_err > 0.01 or head_err > 0.01:
                    print(x_err, y_err, head_err)
                    print("ERROR: failure to recover after collision at ({:.2f},{:.2f})".format(self.x, self.y))
                    return False
                else:
                    self.last_x = x_result
                    self.last_y = y_result
                    self.last_heading = head_result

                    self.x = x_result
                    self.y = y_result
                    self.heading = head_result
                    return True
            else:
                return True

        self.last_x = self.x
        self.last_y = self.y
        self.last_heading = self.heading

        self.x = x_result
        self.y = y_result
        self.heading = head_result
        return True

    def exited(self, x, y):
        return y > EXIT[1] and x < EXIT_WIDTH/2 and x > -EXIT_WIDTH/2

    def out_of_bounds(self, x, y):
        return abs(x) > (ENV_SIZE/2) or abs(y) > (ENV_SIZE/2)

    def normalized(self, state):
        return np.array(state) / 255

# ---------------- PyTorch DQN Class ----------------

class DQN(nn.Module):
    def __init__(self, learning_rate=LEARN_RATE, gamma=GAMMA, action_size=8):
        super(DQN, self).__init__()
        self.input_size = INPUT_SIZE
        self.fc1 = nn.Linear(self.input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)
        self.gamma = gamma

    def forward(self, x):
        # x is expected to be of shape (batch, INPUT_SIZE)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        output = self.fc4(x)
        return output

# ---------------- Soft Update Function ----------------

def soft_update(target, source, tau):
    """
    Soft-update target network parameters:
      target = tau * source + (1-tau) * target
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

# ---------------- Checkpoint Functions ----------------

def save_checkpoint(state, filename):
    torch.save(state, filename)

def load_checkpoint(filename, mainQN, targetQN, optimizer):
    checkpoint = torch.load(filename)
    mainQN.load_state_dict(checkpoint['main_state_dict'])
    targetQN.load_state_dict(checkpoint['target_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['episode']

# ---------------- Memory Replay Class (Unchanged) ----------------

class Memory:
    def __init__(self):
        self.buffer = deque(maxlen=MEM_SIZE)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[ii] for ii in idx]

# ---------------- Main Training Loop ----------------

if __name__ == '__main__':
    # Set up device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create environment and memory (unchanged)
    env = Environment()
    memory = Memory()

    # Instantiate networks and optimizer
    mainQN = DQN(learning_rate=LEARN_RATE, gamma=GAMMA, action_size=8).to(device)
    targetQN = DQN(learning_rate=LEARN_RATE, gamma=GAMMA, action_size=8).to(device)
    optimizer = optim.Adam(mainQN.parameters(), lr=LEARN_RATE)
    
    # Initialize target network to have the same weights as main network
    targetQN.load_state_dict(mainQN.state_dict())

    # Checkpoint directory setup
    if not os.path.isdir(CKPT_PATH):
        os.makedirs(CKPT_PATH)
    ckpt_file = os.path.join(CKPT_PATH, CKPT_NAME)
    
    # Restore checkpoint if available and enabled
    start_episode = 1
    if RESTORE_CKPT and os.path.isfile(ckpt_file):
        print("Restoring DQN from checkpoint:", ckpt_file)
        start_episode = load_checkpoint(ckpt_file, mainQN, targetQN, optimizer) + 1
        # Optionally, remove old checkpoints:
        shutil.rmtree(CKPT_PATH)
        os.makedirs(CKPT_PATH)
        # Save a new initial checkpoint
        save_checkpoint({
            'episode': start_episode - 1,
            'main_state_dict': mainQN.state_dict(),
            'target_state_dict': targetQN.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, ckpt_file)
    else:
        print("\nWARNING: No checkpoint found... DQN initialized from scratch\n")

    # Training loop
    for episode in range(start_episode, EPISODES + 1):
        print("\nEPISODE", episode)
        state = env.reset()  # raw image from camera_gazebo
        step = 0

        while step < MAX_STEPS:
            # Compute epsilon for ε-greedy exploration
            epsilon = EXPLORE_MIN + (EXPLORE_MAX - EXPLORE_MIN) * np.exp(-DECAY_RATE * episode / EPISODES)

            # Normalize the state and convert to tensor
            norm_state = env.normalized(state)
            state_tensor = torch.tensor(norm_state, dtype=torch.float32, device=device).unsqueeze(0)  # shape: (1, INPUT_SIZE)

            # Choose action: explore or exploit
            if np.random.rand() < epsilon:
                action = np.random.randint(0, 8)
            else:
                with torch.no_grad():
                    q_values = mainQN(state_tensor)
                # Convert to numpy to allow random tie-breaking among best actions
                q_values_np = q_values.cpu().numpy().flatten()
                max_q = np.max(q_values_np)
                # Select randomly among the actions that equal max_q
                best_actions = np.where(q_values_np == max_q)[0]
                action = int(np.random.choice(best_actions))

            # Take step in the environment
            next_state, reward, exited, terminate = env.step(action)
            memory.add((env.normalized(state), action, reward, env.normalized(next_state), exited))
            step += 1
            state = next_state

            # Training step: only if enough samples have been gathered
            if len(memory.buffer) >= BATCH_SIZE and step % TRAIN_STEP == 0:
                batch = memory.sample(BATCH_SIZE)
                # Unpack the batch
                batch_states = np.array([mem[0] for mem in batch])  # shape: (BATCH_SIZE, INPUT_SIZE)
                batch_actions = np.array([mem[1] for mem in batch])   # shape: (BATCH_SIZE,)
                batch_rewards = np.array([mem[2] for mem in batch], dtype=np.float32)  # shape: (BATCH_SIZE,)
                batch_next_states = np.array([mem[3] for mem in batch])  # shape: (BATCH_SIZE, INPUT_SIZE)
                batch_finish = np.array([1.0 if mem[4] else 0.0 for mem in batch], dtype=np.float32)  # finished flag as float

                # Convert batch data to tensors
                states_tensor = torch.tensor(batch_states, dtype=torch.float32, device=device)
                actions_tensor = torch.tensor(batch_actions, dtype=torch.long, device=device)
                rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32, device=device)
                next_states_tensor = torch.tensor(batch_next_states, dtype=torch.float32, device=device)
                finish_tensor = torch.tensor(batch_finish, dtype=torch.float32, device=device)

                # Compute target Q-values using the target network
                with torch.no_grad():
                    target_q_values = targetQN(next_states_tensor)
                    # For each sample, take the maximum Q-value
                    max_target_q, _ = torch.max(target_q_values, dim=1)
                    # If the state is terminal, set the Q-value to 0 by multiplying with (1 - finish)
                    max_target_q = max_target_q * (1 - finish_tensor)
                    # Compute the target value (TD target)
                    td_target = rewards_tensor + GAMMA * max_target_q

                # Compute the main network Q-values for the current states
                q_values = mainQN(states_tensor)
                # Select the Q-value for the action that was actually taken
                q_value = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

                # Compute loss (MSE)
                loss = F.mse_loss(q_value, td_target)

                # Optimize the main network
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if terminate:
                break

        print(step, "steps")
        if step >= MAX_STEPS:
            print("RESULT: took too many steps")

        # Save checkpoint every SAVE_STEP episodes
        if episode % SAVE_STEP == 0:
            save_checkpoint({
                'episode': episode,
                'main_state_dict': mainQN.state_dict(),
                'target_state_dict': targetQN.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(CKPT_PATH, CKPT_NAME))
            print("Checkpoint saved at episode", episode)

        # Update target network every UPDATE_TARGET_EVERY episodes
        if episode % UPDATE_TARGET_EVERY == 0:
            soft_update(targetQN, mainQN, TAU)

    # Save final checkpoint
    save_checkpoint({
        'episode': EPISODES,
        'main_state_dict': mainQN.state_dict(),
        'target_state_dict': targetQN.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(CKPT_PATH, CKPT_NAME))
    print("Training complete. Final checkpoint saved.")
