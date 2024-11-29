"""Hierarchical DCQN losses.
"""
from typing import Any

from brax.training import types
from brax.training.types import Params
from brax.training.types import PRNGKey
import jax
import jax.numpy as jnp

from task_aware_skill_composition.brax.agents.hdcqn import networks as hdcq_networks


Transition = types.Transition


def make_losses(
        hdcq_network: hdcq_networks.HDCQNetworks,
        reward_scaling: float,
        cost_scaling: float,
        cost_budget: float,
        discounting: float,
):
  """Creates the HDCQN losses."""

  option_q_network = hdcq_network.option_q_network
  cost_q_network = hdcq_network.cost_q_network

  def safe_greedy_policy(reward_qs, cost_qs):
    "Finds option with maximal value under cost constraint"
    masked_q = jnp.where(cost_qs < cost_budget, reward_qs, -jnp.inf)
    option = masked_q.argmax(axis=-1)
    q = jax.vmap(lambda x, i: x.at[i].get())(reward_qs, option)
    cq = jax.vmap(lambda x, i: x.at[i].get())(cost_qs, option)
    return q, cq

  # TODO: Check if one should use target network when its not being updated

  def critic_loss(
      option_q_params: Params,
      cost_q_params: Params,
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
    next_double_cqs = cost_q_network.apply(normalizer_params, cost_q_params, transitions.next_observation)

    # Q(s_t+1, o_t+1) for all options
    next_qs = jnp.min(next_double_qs, axis=-1)
    next_cqs = jnp.min(next_double_cqs, axis=-1)

    # V(s_t+1) = max_o Q(s_t+1, o) (because pi is argmax Q)
    next_v, _ = safe_greedy_policy(next_qs, next_cqs)

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

  def cost_critic_loss(
      option_q_params: Params,
      cost_q_params: Params,
      normalizer_params: Any,
      target_cost_q_params: Params,
      transitions: Transition,
      key: PRNGKey
  ) -> jnp.ndarray:

    # Q(s_t, o_t) for all options
    cqs_old = cost_q_network.apply(normalizer_params, cost_q_params, transitions.observation)
    cq_old_action = jax.vmap(lambda x, i: x.at[i].get())(cqs_old, transitions.option)

    # Q1(s_t+1, o_t+1)/Q2(s_t+1, o_t+1) for all options
    next_double_qs = option_q_network.apply(normalizer_params, option_q_params, transitions.next_observation)
    next_double_cqs = cost_q_network.apply(normalizer_params, target_cost_q_params, transitions.next_observation)

    # Q(s_t+1, o_t+1) for all options
    next_qs = jnp.min(next_double_qs, axis=-1)
    next_cqs = jnp.min(next_double_cqs, axis=-1)

    # V(s_t+1) = max_o Q(s_t+1, o) (because pi is argmax Q)
    _, next_cv = safe_greedy_policy(next_qs, next_cqs)

    # E (s_t, a_t, s_t+1) ~ D [r(s_t, a_t) + gamma * V(s_t+1)]
    target_cq = jax.lax.stop_gradient(transitions.extras["state_extras"]["cost"] * cost_scaling +
                                     transitions.discount * discounting * next_cv)

    # Q(s_t, a_t) - E[r(s_t, a_t) + gamma * V(s_t+1)]
    cq_error = cq_old_action - jnp.expand_dims(target_cq, -1)

    # Better bootstrapping for truncated episodes.
    truncation = transitions.extras['state_extras']['truncation']
    cq_error *= jnp.expand_dims(1 - truncation, -1)

    cq_loss = 0.5 * jnp.mean(jnp.square(cq_error))
    return cq_loss

  return critic_loss, cost_critic_loss
