
import math
import random
from itertools import count
from typing import List

import gym
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
from torch import nn
from torch.functional import Tensor

from mouse.agent.network import DQN
from mouse.environment.steinmetz import Steinmetz
from mouse.model import Action, Transition
from mouse.util import ReplayMemory, plot_durations

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

def get_cart_location(env, screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

class Agent:

    def __init__(self) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #  self._env = gym.make('CartPole-v0').unwrapped
        self._env = Steinmetz()
        self._env.reset()

        self._steps_done = 0
        self._memory = ReplayMemory(10000)
        self._episode_durations: List[int] = []

        init_screen = self._get_screen()
        _, _, screen_height, screen_width = init_screen.shape
        print(f"Screen is {screen_width}x{screen_height}")
        self._policy_net = DQN(screen_height, screen_width, self._n_actions, self._device).to(self._device)
        self._target_net = DQN(screen_height, screen_width, self._n_actions, self._device).to(self._device)
        self._optimizer = optim.RMSprop(self._policy_net.parameters())

    def train(self, num_episodes: int = 50):
        for i_episode in range(num_episodes):
            self._env.reset()
            last_screen = self._get_screen()
            current_screen = self._get_screen()
            state: Tensor = current_screen - last_screen

            for t in count():
                action = self._select_action(state)
                _, reward, done, _ = self._env.step(Action(action.item()))
                reward = torch.tensor([reward], device=self._device)

                # Observe new state
                last_screen = current_screen
                current_screen = self._get_screen()
                if not done:
                    next_state = current_screen - last_screen
                else:
                    self._memory.push(state, action, next_state, reward)
                    self._optimize_model()
                    self._episode_durations.append(t + 1)
                    plot_durations(self._episode_durations)
                    break

                # Store the transition in memory
                self._memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self._optimize_model()
            # Update the target network, copying all weights and biases in DQN
            if i_episode % TARGET_UPDATE == 0:
                self._target_net.load_state_dict(self._policy_net.state_dict())

        self._env.render()
        self._env.close()

    @property
    def _n_actions(self) -> int:
        return self._env.action_space.n

    def _optimize_model(self):
        if len(self._memory) < BATCH_SIZE:
            return
        transitions = self._memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self._device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to self._policy_net
        state_action_values = self._policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" self._target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=self._device)
        next_state_values[non_final_mask] = self._target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self._optimizer.zero_grad()
        loss.backward()
        for param in self._policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self._optimizer.step()

    def _get_screen(self):
        screen = self._env.render(mode='rgb_array').transpose((2, 0, 1))

        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return resize(screen).unsqueeze(0)

    def _select_action(self, state: Tensor) -> Tensor:  # TODO parameter types
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self._steps_done / EPS_DECAY)
        self._steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self._policy_net(state).max(1)[1].view(1, 1)
        else:
            n_actions = self._env.action_space.n
            return torch.tensor([[random.randrange(n_actions)]], device=self._device, dtype=torch.long)

