from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class DQN(nn.Module):

    def __init__(self, h, w, outputs, device):  # TODO types
        super(DQN, self).__init__()
        self._device = device
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(self._device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

    def save(self, path: Path):
        torch.save(self.state_dict(), path)

    @classmethod
    def from_weights(cls, path: Path, h, w, outputs, device) -> "DQN":
        dqn = DQN(h, w, outputs, device)
        weights = torch.load(path)
        dqn.load_state_dict(weights)

        if torch.cuda.is_available():
            dqn.cuda()

        return dqn
