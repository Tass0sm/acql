"""Table definitions."""

import dataclasses
from typing import Any, Callable, Sequence, Tuple
import warnings

import jax
import jax.numpy as jnp


@dataclasses.dataclass
class Table:
    init: Callable[..., Any]
    apply: Callable[..., Any]
  

def make_q_table(
        observation_space_size: int,
        action_space_size: int
):
    """Creates a q function table."""
  
    # Q function
    class QTable():
        """Q Table."""
        
        def __call__(self, params, obs: jnp.ndarray):
            return jnp.squeeze(params[obs])
  
    q_table = QTable()
  
    def init(key):
        return jnp.zeros((observation_space_size, action_space_size))

    def apply(q_params, obs):
        return q_table(q_params, obs)
  
    return Table(init=init, apply=apply)
