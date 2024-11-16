"""Brax hierarchical training types."""

from typing import NamedTuple

from brax.envs.base import State
from brax.training.acme.types import NestedArray


class HierarchicalTransition(NamedTuple):
  """Container for my transition which includes the full environment state."""
  prev_option: NestedArray
  termination: NestedArray
  observation: NestedArray
  option: NestedArray
  action: NestedArray
  reward: NestedArray
  discount: NestedArray
  next_observation: NestedArray
  extras: NestedArray = ()  # pytype: disable=annotation-type-mismatch  # jax-ndarray
