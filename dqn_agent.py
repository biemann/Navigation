import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork
from model_pixels import QCNNetwork

import torch
import torch.nn.functional as f
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

BUFFER_SIZE = int(10000)  # replay buffer size
BATCH_SIZE = 32  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 1e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network

# Without pixels:

# BUFFER_SIZE = int(10000)  # replay buffer size
# BATCH_SIZE = 128        # minibatch size
# GAMMA = 0.99            # discount factor
# TAU = 2e-3              # for soft update of target parameters
# LR = 5e-4            # learning rate
# UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def soft_update(local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target

    Params
    ======
        local_model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, name):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.name = name

        # Q-Network
        if self.name == "pixels":
            self.q_network_local = QCNNetwork(action_size, seed).to(device)
            self.q_network_target = QCNNetwork(action_size, seed).to(device)
        else:
            self.q_network_local = QNetwork(state_size, action_size, seed).to(device)
            self.q_network_target = QNetwork(state_size, action_size, seed).to(device)

        self.optimizer = optim.Adam(self.q_network_local.parameters(), lr=LR)
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.5)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, name)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done, alg):
        # Save experience in replay memory
        if self.name == "pixels":
            self.memory.add(state[3], action, reward, next_state[3], done)
        else:
            self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA, alg)

    def act(self, state, eps=0.0):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.q_network_local.eval()
        with torch.no_grad():
            action_values = self.q_network_local(state)
        self.q_network_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma, alg):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        if alg == "dqn":
            # Get max predicted Q values (for next states) from target model
            q_targets_next = self.q_network_target(next_states).detach().max(1)[0].unsqueeze(1)

        else:
            # best action according to the local network:
            best_action_next = self.q_network_local(next_states).detach().max(1)[1].unsqueeze(1)

            # target of the target network, according to that action
            q_targets_next = self.q_network_target(next_states).detach().gather(1, best_action_next)

        # Compute Q targets for current states
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))
        # Get expected Q values from local model
        q_expected = self.q_network_local(states).gather(1, actions)

        # Compute loss (Huber loss)
        loss = f.smooth_l1_loss(q_expected, q_targets)  # mse + clipping in the article

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()  # because we want stochastic gradient ascent and not descent
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        soft_update(self.q_network_local, self.q_network_target, TAU)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, name):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.name = name
        self.aug_states = []
        self.aug_next_states = []

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""

        if self.name == "pixels":
            indices = random.sample(range(len(self.memory)), k=self.batch_size)
            self.aug_states = []
            self.aug_next_states = []
            for i in indices:
                if (i % 300) > 2:
                    aug_state = np.stack(
                        [self.memory[i - 3].state, self.memory[i - 2].state, self.memory[i - 1].state,
                         self.memory[i].state])
                    aug_next_state = np.stack(
                        [self.memory[i - 3].next_state, self.memory[i - 2].next_state, self.memory[i - 1].next_state,
                         self.memory[i].next_state])
                else:
                    aug_state = np.stack([self.memory[i].state for t in range(4)])
                    aug_next_state = np.stack([self.memory[i].next_state for t in range(4)])
                self.aug_states.append(aug_state)
                self.aug_next_states.append(aug_next_state)

            states = torch.from_numpy(np.stack(self.aug_states)).float().to(device)
            next_states = torch.from_numpy(np.stack(self.aug_next_states)).float().to(device)
            experiences = [self.memory[i] for i in indices]
        else:
            experiences = random.sample(self.memory, k=self.batch_size)
            states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
            next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
                device)

        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
