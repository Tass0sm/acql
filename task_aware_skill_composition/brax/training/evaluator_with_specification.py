"""Evaluator which also checks specification satisfaction."""

import time
from typing import Callable, Sequence, Tuple, Union

from brax import envs
from brax.training.types import Metrics
from brax.training.types import Policy
from brax.training.types import PolicyParams
from brax.training.types import PRNGKey
from brax.training.types import Transition
import jax
import jax.numpy as jnp
import numpy as np

State = Union[envs.State]
Env = Union[envs.Env]


from brax.training.acting import Evaluator, generate_unroll


class EvaluatorWithSpecification(Evaluator):
  """Class to run evaluations."""

  def __init__(
          self,
          eval_env: envs.Env,
          eval_policy_fn: Callable[[PolicyParams], Policy],
          specification: Callable[..., jnp.ndarray],
          state_var,
          num_eval_envs: int,
          episode_length: int,
          action_repeat: int,
          key: PRNGKey
  ):
      self._key = key
      self._eval_walltime = 0.

      eval_env = envs.training.EvalWrapper(eval_env)

      def generate_eval_unroll(policy_params: PolicyParams,
                               key: PRNGKey) -> State:
          reset_keys = jax.random.split(key, num_eval_envs)
          eval_first_state = eval_env.reset(reset_keys)
          return generate_unroll(
              eval_env,
              eval_first_state,
              eval_policy_fn(policy_params),
              key,
              unroll_length=episode_length // action_repeat)

      self._generate_eval_unroll = jax.jit(generate_eval_unroll)
      self._steps_per_unroll = episode_length * num_eval_envs

      self.specification = specification
      self.state_var = state_var

  def run_evaluation(self,
                     policy_params: PolicyParams,
                     training_metrics: Metrics,
                     aggregate_episodes: bool = True) -> Metrics:
    """Run one epoch of evaluation."""
    self._key, unroll_key = jax.random.split(self._key)

    t = time.time()

    eval_state, data = self._generate_eval_unroll(policy_params, unroll_key)
    # put the batch dim first
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

    eval_metrics = eval_state.info['eval_metrics']

    # From data, calculate robustness
    robustness = self.specification({ self.state_var.idx: data.observation,
                                      # "action": data.action
                                     })
    eval_metrics.episode_metrics["robustness"] = robustness

    eval_metrics.active_episodes.block_until_ready()
    epoch_eval_time = time.time() - t
    metrics = {}
    for fn in [np.mean, np.std]:
      suffix = '_std' if fn == np.std else ''
      metrics.update(
          {
              f'eval/episode_{name}{suffix}': (
                  fn(value) if aggregate_episodes else value
              )
              for name, value in eval_metrics.episode_metrics.items()
          }
      )
    metrics['eval/avg_episode_length'] = np.mean(eval_metrics.episode_steps)
    metrics['eval/epoch_eval_time'] = epoch_eval_time
    metrics['eval/sps'] = self._steps_per_unroll / epoch_eval_time
    self._eval_walltime = self._eval_walltime + epoch_eval_time
    metrics = {
        'eval/walltime': self._eval_walltime,
        **training_metrics,
        **metrics
    }

    return metrics  # pytype: disable=bad-return-type  # jax-ndarray
