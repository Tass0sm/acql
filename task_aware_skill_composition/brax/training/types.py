"""My Brax training types."""

from typing import NamedTuple

from brax.envs.base import State
from brax.training.acme.types import NestedArray


class MyTransition(NamedTuple):
  """Container for my transition which includes the full environment state."""
  state: State
  observation: NestedArray
  action: NestedArray
  reward: NestedArray
  discount: NestedArray
  next_observation: NestedArray
  next_state: State
  extras: NestedArray = ()  # pytype: disable=annotation-type-mismatch  # jax-ndarray
