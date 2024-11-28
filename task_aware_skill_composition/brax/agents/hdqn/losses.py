"""Hierarchical DQN losses.
"""
from typing import Any

from brax.training import types
from brax.training.types import Params
from brax.training.types import PRNGKey
import jax
import jax.numpy as jnp

from task_aware_skill_composition.brax.agents.hdqn import networks as hdq_networks


Transition = types.Transition


def make_critic_loss(
        hdq_network: hdq_networks.HDQNetworks,
        reward_scaling: float,
        discounting: float,
):
  """Creates the HDQN losses."""

  option_q_network = hdq_network.option_q_network

  def critic_loss(
          option_q_params: Params,
          normalizer_params: Any,
          target_option_q_params: Params,
          transitions: Transition,
          key: PRNGKey
  ) -> jnp.ndarray:

    # Q(s_t, o_t) for all options
    qs_old = option_q_network.apply(normalizer_params, option_q_params, transitions.observation)
    q_old_action = jax.vmap(lambda x, i: x.at[i].get())(qs_old, transitions.option)

    # Q1(s_t+1, o_t+1)/Q2(s_t+1, o_t+1) for all options
    next_double_qs = option_q_network.apply(normalizer_params, target_option_q_params, transitions.next_observation)

    # Q(s_t+1, o_t+1) for all options
    next_qs = jnp.min(next_double_qs, axis=-1)

    # V(s_t+1) = max_o Q(s_t+1, o) (because pi is argmax Q)
    next_v = next_qs.max(axis=-1)

    # E (s_t, a_t, s_t+1) ~ D [r(s_t, a_t) + gamma * V(s_t+1)]
    target_q = jax.lax.stop_gradient(transitions.reward * reward_scaling +
                                     transitions.discount * discounting * next_v)

    # Q(s_t, a_t) - E[r(s_t, a_t) + gamma * V(s_t+1)]
    q_error = q_old_action - jnp.expand_dims(target_q, -1)

    # Better bootstrapping for truncated episodes.
    truncation = transitions.extras['state_extras']['truncation']
    q_error *= jnp.expand_dims(1 - truncation, -1)

    q_loss = 0.5 * jnp.mean(jnp.square(q_error))
    return q_loss

  return critic_loss
