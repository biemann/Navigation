import torch
import torch.nn as nn
import torch.nn.functional as f


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """

        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Network architecture:

        self.n_neurons = 128
        self.dropout_prob = 0.2

        self.fc1 = nn.Linear(state_size, self.n_neurons)
        self.bn1 = nn.BatchNorm1d(self.n_neurons)
        self.do1 = nn.Dropout(self.dropout_prob)
        self.fc2 = nn.Linear(self.n_neurons, self.n_neurons)
        self.bn2 = nn.BatchNorm1d(self.n_neurons)
        self.do2 = nn.Dropout(self.dropout_prob)
        self.fc3 = nn.Linear(self.n_neurons, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.do1(self.bn1(f.selu(self.fc1(state))))
        x = self.do2(self.bn2(f.selu(self.fc2(x))))
        return self.fc3(x)
        pass
