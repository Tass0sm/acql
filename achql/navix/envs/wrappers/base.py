from typing import Union

import jax

from navix import Timestep


class Wrapper:
  """Wraps an environment to allow modular transformations."""

  def __init__(self, env):
    self.env = env

  def reset(self, rng: jax.Array) -> Timestep:
    return self.env.reset(rng)

  def step(self, timestep: Timestep, action: jax.Array) -> Timestep:
    return self.env.step(timestep, action)

  # @property
  # def observation_size(self) -> ObservationSize:
  #   return self.env.observation_size

  # @property
  # def action_size(self) -> int:
  #   return self.env.action_size

  # @property
  # def unwrapped(self) -> Env:
  #   return self.env.unwrapped

  # @property
  # def backend(self) -> str:
  #   return self.unwrapped.backend

  def __getattr__(self, name):
    if name == '__setstate__':
      raise AttributeError(name)
    return getattr(self.env, name)
