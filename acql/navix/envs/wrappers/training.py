"""Wrappers to support parallelized navix training."""

from typing import Callable, Dict, Optional, Tuple

from flax import struct
import jax
from jax import numpy as jp

from navix import Timestep

from acql.navix.envs.wrappers.base import Wrapper


def wrap_for_training(
    env,
    episode_length: int = 100,
    # action_repeat: int = 1,
    # randomization_fn: Optional[
    #     Callable[[System], Tuple[System, System]]
    # ] = None,
):
    """Common wrapper pattern for all training agents.
  
    Args:
      env: environment to be wrapped
      episode_length: length of episode
      action_repeat: how many repeated actions to take per step
      randomization_fn: randomization function that produces a vectorized system
        and in_axes to vmap over
  
    Returns:
      An environment that is wrapped with Episode and AutoReset wrappers.  If the
      environment did not already have batch dimensions, it is additional Vmap
      wrapped.
    """
    env = VmapWrapper(env)
    # env = EpisodeWrapper(env, episode_length, action_repeat)
    # env = AutoResetWrapper(env)
    return env


class VmapWrapper(Wrapper):
    """Vectorizes Brax env."""
  
    def __init__(self, env, batch_size: Optional[int] = None):
        super().__init__(env)
        self.batch_size = batch_size
  
    def reset(self, rng: jax.Array) -> Timestep:
        if self.batch_size is not None:
            rng = jax.random.split(rng, self.batch_size)
        return jax.vmap(self.env.reset)(rng)
  
    def step(self, timestep: Timestep, action: jax.Array) -> Timestep:
        return jax.vmap(self.env.step)(timestep, action)


@struct.dataclass
class EvalMetrics:
    """Dataclass holding evaluation metrics for Brax.
  
    Attributes:
        episode_metrics: Aggregated episode metrics since the beginning of the
          episode.
        active_episodes: Boolean vector tracking which episodes are not done yet.
        episode_steps: Integer vector tracking the number of steps in the episode.
    """
  
    episode_metrics: Dict[str, jax.Array]
    active_episodes: jax.Array
    episode_steps: jax.Array


class EvalWrapper(Wrapper):
  """Brax env with eval metrics."""

  def reset(self, rng: jax.Array) -> Timestep:
      reset_timestep = self.env.reset(rng)
      eval_metrics = EvalMetrics(
          episode_metrics={
              "reward": reset_timestep.reward
          },
          active_episodes=jp.ones_like(reset_timestep.reward),
          episode_steps=jp.zeros_like(reset_timestep.reward),
      )
      reset_timestep.info['eval_metrics'] = eval_metrics
      return reset_timestep
  
  def step(self, timestep: Timestep, action: jax.Array) -> Timestep:
      timestep_metrics = timestep.info['eval_metrics']
      if not isinstance(timestep_metrics, EvalMetrics):
          raise ValueError(
              f'Incorrect type for timestep_metrics: {type(timestep_metrics)}'
          )
      del timestep.info['eval_metrics']
      ntimestep = self.env.step(timestep, action)
      ntimestep_metrics = {
          "reward": ntimestep.reward
      }
      episode_steps = jp.where(
          timestep_metrics.active_episodes,
          ntimestep.t,
          timestep_metrics.episode_steps,
      )
      episode_metrics = jax.tree_util.tree_map(
          lambda a, b: a + b * timestep_metrics.active_episodes,
          timestep_metrics.episode_metrics,
          ntimestep_metrics,
      )
      active_episodes = timestep_metrics.active_episodes * (1 - ntimestep.is_done())
      
      eval_metrics = EvalMetrics(
          episode_metrics=episode_metrics,
          active_episodes=active_episodes,
          episode_steps=episode_steps,
      )
      ntimestep.info['eval_metrics'] = eval_metrics
      return ntimestep


# class DomainRandomizationVmapWrapper(Wrapper):
#     """Wrapper for domain randomization."""

#     def __init__(
#             self,
#             env: Env,
#             randomization_fn: Callable[[System], Tuple[System, System]],
#     ):
#         super().__init__(env)
#         self._sys_v, self._in_axes = randomization_fn(self.sys)

#     def _env_fn(self, sys: System) -> Env:
#         env = self.env
#         env.unwrapped.sys = sys
#         return env

#     def reset(self, rng: jax.Array) -> Timestep:
#         def reset(sys, rng):
#             env = self._env_fn(sys=sys)
#             return env.reset(rng)

#         timestep = jax.vmap(reset, in_axes=[self._in_axes, 0])(self._sys_v, rng)
#         return timestep

#     def step(self, timestep: Timestep, action: jax.Array) -> Timestep:
#         def step(sys, s, a):
#             env = self._env_fn(sys=sys)
#             return env.step(s, a)

#         res = jax.vmap(step, in_axes=[self._in_axes, 0, 0])(
#             self._sys_v, timestep, action
#         )
#         return res
