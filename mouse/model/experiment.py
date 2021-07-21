import random
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Tuple

#  @dataclass
#  class Transition:
#      state: str
#      action: str
#      next_state: str
#      reward: int

class InvalidMove(Exception):
    """The attempted move was invalid"""

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class Status(Enum):
    IN_GAME = 0
    WIN = 1
    FAIL = -1

class Action(Enum):
    LEFT = 0
    RIGHT = 1


class Side(Enum):
    LEFT = 1
    RIGHT = -1


class Screen:
    WIDTH = 800
    HEIGHT = 300

STIM_DIAMETER = 100
START_OFFSET = 100  # Distance from (e.g.) left side
START_POS_LEFT = (
    (START_OFFSET, int(Screen.HEIGHT/3)),
    (START_OFFSET + STIM_DIAMETER, int(Screen.HEIGHT/3) + STIM_DIAMETER)
)
START_POS_RIGHT = (
    (Screen.WIDTH-START_OFFSET-STIM_DIAMETER, int(Screen.HEIGHT/3)),
    (Screen.WIDTH-START_OFFSET, int(Screen.HEIGHT/3) + STIM_DIAMETER)
)
STEP_SIZE = 40


@dataclass
class Stimulus:
    x0: int
    y0: int

    x1: int
    y1: int

    contrast: float

    def move(self, action: Action):
        if action == Action.LEFT:
            if self.x0 - STEP_SIZE < 0:
                raise InvalidMove()

            self.x0 -= STEP_SIZE
            self.x1 -= STEP_SIZE
        elif action == Action.RIGHT:
            if self.x1 + STEP_SIZE > Screen.WIDTH:
                raise InvalidMove()

            self.x0 += STEP_SIZE
            self.x1 += STEP_SIZE

    def is_in_centre(self) -> bool:
        centre_pos = Screen.WIDTH / 2
        stim_pos = self.x0 + (STIM_DIAMETER / 2)
        distance_from_centre = abs(centre_pos - stim_pos)
        return distance_from_centre <= 50

    @property
    def position(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return ((self.x0, self.y0), (self.x1, self.y1))

    @classmethod
    def left(cls) -> "Stimulus":
        x0, y0 = START_POS_LEFT[0]
        x1, y1 = START_POS_LEFT[1]
        contrast = random.uniform(0, 1)
        return cls(x0, y0, x1, y1, contrast)

    @classmethod
    def right(cls) -> "Stimulus":
        x0, y0 = START_POS_RIGHT[0]
        x1, y1 = START_POS_RIGHT[1]
        contrast = random.uniform(0, 1)
        return cls(x0, y0, x1, y1, contrast)

    @classmethod
    def from_direction(cls, direction: str) -> "Stimulus":
        if direction not in ["left", "right"]:
            raise ValueError(f"Invalid direction {direction}")

        constructor: Optional[Callable[[], Stimulus]] = getattr(cls, direction, None)
        if not constructor:
            raise RuntimeError(f"No constructor found for {direction} direction")

        return constructor()

    @property
    def rgb(self) -> Tuple[int, int, int]:
        val = int(255 * self.contrast)
        return (val, val, val)

    @property
    def center_point(self) -> Tuple[int, int]:
        return (
            int(self.x0 + (STIM_DIAMETER/2)),
            int(self.y0 + (STIM_DIAMETER/2))
        )

    @property
    def side(self) -> Side:
        if ((self.x0, self.y0), (self.x1, self.y1)) == START_POS_LEFT:
            return Side.LEFT
        elif ((self.x0, self.y0), (self.x1, self.y1)) == START_POS_RIGHT:
            return Side.RIGHT
        else:
            raise RuntimeError("Stimulus is not in the starting position")


