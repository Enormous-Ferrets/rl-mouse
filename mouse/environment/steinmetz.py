import random
from typing import Optional

import gym
import numpy as np
from gym import spaces
from gym.envs.classic_control import rendering
from PIL import Image, ImageDraw

from mouse.model import Action, InvalidMove, Screen, Stimulus


class Steinmetz(gym.Env):
    """Steinmetz Experiment Environment

    Actions: { LEFT, RIGHT }
    """
    metadata = {"render.modes": []}

    def __init__(self) -> None:
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3, 5), dtype=np.float32)

        self._stimulus: Optional[Stimulus] = None

        self.viewer = rendering.SimpleImageViewer()

    def step(self, action: Action):
        assert self.action_space.contains(action.value)

        try:
            self.stimulus.move(action)
            if self.stimulus.is_in_centre():
                reward = 1.0
                done = True
            else:
                reward = 0.0
                done = False
        except InvalidMove:
            reward = -1.0
            done = True

        obs = self._observe()
        info = {}

        return obs, reward, done, info

    def reset(self):
        """Reset the environment

        Sets the stimulus position to a random position
        """
        self._stimulus = random.choice((Stimulus.left, Stimulus.right))()

    def render(self, mode="rgb_array"):
        if mode is not "rgb_array":
            raise NotImplementedError("Only rgb_array is supported so far")

        im = Image.new('RGB', (Screen.WIDTH, Screen.HEIGHT), (255, 255, 255))
        draw = ImageDraw.Draw(im)
        draw.ellipse(self.stimulus.position, fill=self.stimulus.rgb)
        im.save('foo.jpg', quality=95)
        render_data = np.array(im)
        self.viewer.imshow(render_data)

        return render_data

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    @property
    def stimulus(self) -> Stimulus:
        assert self._stimulus is not None
        return self._stimulus

    # Private ###################################

    def _observe(self):  # TODO(rory) - numpy typing
        center_point = self.stimulus.center_point
        return np.array([center_point[0], center_point[1], self.stimulus.contrast])

