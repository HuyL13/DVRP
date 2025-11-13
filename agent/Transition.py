from typing import NamedTuple

import numpy as np


class Transition(NamedTuple):
    obs: np.ndarray
    action: int
    logp: float
    reward: float
    done: float
    value: float

