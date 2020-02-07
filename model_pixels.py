import torch
import torch.nn as nn
import torch.nn.functional as f


class QCNNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """

        super(QCNNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Network architecture:
        self.input_dim1 = 16
        self.input_dim2 = 32
        self.neurons = 256

        self.conv1 = nn.Conv2d(4, self.input_dim1, kernel_size=8, stride=4)  # dim_output = 20, paper: 32
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(self.input_dim1, self.input_dim2, kernel_size=4, stride=2)  # dim_output = 9, paper: 64
        # self.bn2 = nn.BatchNorm2d(32)
        # self.conv3 = nn.Conv2d(self.input_dim2, self.input_dim2, kernel_size=3, stride=1)  # dim_output = 7
        # self.bn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(9 * 9 * self.input_dim2, self.neurons)       # paper: 512
        self.fc2 = nn.Linear(self.neurons, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = f.relu(self.conv1(state))
        x = f.relu(self.conv2(x))
        # x = f.relu(self.conv3(x))
        x = f.relu(self.fc1(x.reshape(x.size(0), -1)))
        return self.fc2(x)
        pass
