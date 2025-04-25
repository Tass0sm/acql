"""Evaluator which also checks specification satisfaction."""

import time
from typing import Callable, Sequence, Tuple, Union, Optional

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


from brax.training.acting import Evaluator

from achql.hierarchy.training.acting import semimdp_generate_unroll
from achql.hierarchy.state import OptionState
from achql.hierarchy.option import Option


class HierarchicalEvaluatorWithSpecification(Evaluator):
  """Class to run evaluations."""

  def __init__(
          self,
          eval_env: envs.Env,
          eval_policy_fn: Callable[[PolicyParams], Policy],
          options: Sequence[Option],
          specification: Callable[..., jnp.ndarray],
          state_var,
          num_eval_envs: int,
          episode_length: int,
          action_repeat: int,
          key: PRNGKey,
          return_states: bool = False,
  ):
      self._key = key
      self._eval_walltime = 0.

      self.env = eval_env
      self.has_automaton = hasattr(eval_env, "automaton")
      self.obs_augmented = eval_env.augment_obs if self.has_automaton else None
      self.automaton_states = eval_env.automaton.num_states if self.has_automaton else None

      eval_env = envs.training.EvalWrapper(eval_env)

      def generate_eval_unroll(policy_params: PolicyParams,
                               key: PRNGKey) -> State:
          reset_keys = jax.random.split(key, num_eval_envs)
          eval_first_state = eval_env.reset(reset_keys)
          return semimdp_generate_unroll(
              eval_env,
              eval_first_state,
              eval_policy_fn(policy_params),
              key,
              unroll_length=episode_length // action_repeat,
              return_states=return_states,
          )

      self._generate_eval_unroll = jax.jit(generate_eval_unroll)
      self._steps_per_unroll = episode_length * num_eval_envs

      self._return_states = return_states

      self.specification = specification
      self.state_var = state_var

  def run_evaluation(self,
                     policy_params: PolicyParams,
                     training_metrics: Metrics,
                     aggregate_episodes: bool = True,
                     return_data: bool = False) -> Metrics:
    """Run one epoch of evaluation."""
    self._key, unroll_key = jax.random.split(self._key)

    t = time.time()

    eval_state, all_data = self._generate_eval_unroll(policy_params, unroll_key)

    if self._return_states:
      states, data = all_data
    else:
      data = all_data

    # put the batch dim first
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

    eval_metrics = eval_state.info['eval_metrics']

    if self.has_automaton and self.obs_augmented:
      obs = data.observation[..., :self.env.original_obs_dim]
      # obs = data.observation[..., :-self.automaton_states]
    else:
      obs = data.observation

    ap_param_fields = {}

    if hasattr(self.env, "randomize_goals") and getattr(self.env, "randomize_goals", False):
      for _, ap in self.env.automaton.aps.items():
        i = 32 + ap.info["ap_i"]
        ap_param_fields[i] = eval_state.info["ap_params"][:, 0]

    # From data, calculate robustness
    robustness_traces = jax.vmap(self.specification)({ self.state_var.idx: obs,
                                                       # "action": data.action
                                                      } | ap_param_fields)

    robustness = robustness_traces[..., 0]
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

    metrics['eval/proportion_robustness_over_zero'] = jnp.mean(jnp.where(robustness > 0, 1.0, 0.0))
    metrics['eval/min_episode_robustness'] = jnp.min(robustness)
    metrics['eval/max_episode_robustness'] = jnp.max(robustness)
    metrics['eval/avg_episode_length'] = np.mean(eval_metrics.episode_steps)
    metrics['eval/epoch_eval_time'] = epoch_eval_time
    metrics['eval/sps'] = self._steps_per_unroll / epoch_eval_time
    self._eval_walltime = self._eval_walltime + epoch_eval_time
    metrics = {
        'eval/walltime': self._eval_walltime,
        **training_metrics,
        **metrics
    }

    if return_data:
      return metrics, eval_state, all_data
    else:
      return metrics  # pytype: disable=bad-return-type  # jax-ndarray
