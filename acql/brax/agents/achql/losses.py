"""Hierarchical DCQN losses.
"""
from typing import Any

from brax import envs
from brax.training import types
from brax.training.types import Params
from brax.training.types import PRNGKey
import jax
import jax.numpy as jnp

from acql.brax.agents.achql import networks as achql_networks

from .argmaxes import *


Transition = types.Transition


def make_losses(
        achql_network: achql_networks.ACHQLNetworks,
        env: envs.Env,
        reward_scaling: float,
        cost_scaling: float,
        safety_threshold: float,
        discounting: float,
        use_sum_cost_critic: bool = False,
        actor_type: str = "argmax",
):
  """Creates the ACQL losses."""

  option_q_network = achql_network.option_q_network
  cost_q_network = achql_network.cost_q_network

  def safe_greedy_policy(reward_qs, cost_qs, option_key):
    "Finds option with maximal value under cost constraint"

    if use_sum_cost_critic:
      masked_q = jnp.where(cost_qs < safety_threshold, reward_qs, -999.)
    else:
      masked_q = jnp.where(cost_qs > safety_threshold, reward_qs, -999.)

    if actor_type == "argmax_random":
      option = argmax_with_random_tiebreak(masked_q, option_key, axis=-1)
    elif actor_type == "argmax_safest":
      option = argmax_with_safest_tiebreak(masked_q, cost_qs, axis=-1, use_sum_cost_critic=use_sum_cost_critic)
    elif actor_type == "argmax":
      option = masked_q.argmax(axis=-1)
    else:
      raise NotImplementedError()

    # all_unsafe = jnp.all(masked_q == -999, axis=-1)
    q = jax.vmap(lambda x, i: x.at[i].get())(reward_qs, option)
    # q = jnp.where(all_unsafe, 0, q)
    cq = jax.vmap(lambda x, i: x.at[i].get())(cost_qs, option)
    return q, cq

  # TODO: Check if one should use target network when its not being updated

  def critic_loss(
      option_q_params: Params,
      cost_q_params: Params,
      normalizer_params: Any,
      target_option_q_params: Params,
      target_cost_q_params: Params,
      transitions: Transition,
      key: PRNGKey
  ) -> jnp.ndarray:

    state_obs, goals, aut_state = env.split_obs(transitions.observation)
    _, aut_state_obs = env.split_aut_obs(transitions.observation)

    next_state_obs, next_goals, next_aut_state = env.split_obs(transitions.next_observation)
    _, next_aut_state_obs = env.split_aut_obs(transitions.next_observation)

    # Double Q(s_t, o_t) for all options
    qr_input = transitions.observation
    qs_old = option_q_network.apply(normalizer_params, option_q_params, qr_input)
    q_old_action = jax.vmap(lambda x, i: x.at[i].get())(qs_old, transitions.action)

    # Q1(s_t+1, o_t+1)/Q2(s_t+1, o_t+1) for all options
    next_qr_input = transitions.next_observation
    next_double_qs = option_q_network.apply(normalizer_params, target_option_q_params, next_qr_input)

    next_qc_input = transitions.next_observation
    next_double_cqs = cost_q_network.apply(normalizer_params, target_cost_q_params, next_qc_input)

    # Q(s_t+1, o_t+1) for all options
    next_qs = jnp.min(next_double_qs, axis=-1)

    if use_sum_cost_critic:
      next_cqs = jnp.max(next_double_cqs, axis=-1)
    else:
      next_cqs = jnp.min(next_double_cqs, axis=-1)

    # V(s_t+1) = max_o Q(s_t+1, o) (because pi is argmax Q)
    next_v, _ = safe_greedy_policy(next_qs, next_cqs, key)

    # E (s_t, a_t, s_t+1) ~ D [r(s_t, a_t) + gamma * V(s_t+1)]
    target_q = jax.lax.stop_gradient(transitions.reward * reward_scaling +
                                     transitions.discount * discounting * next_v)

    # Q(s_t, a_t) - E[r(s_t, a_t) + gamma * V(s_t+1)]
    q_error = q_old_action - jnp.expand_dims(target_q, -1)

    # Better bootstrapping for truncated episodes.
    truncation = transitions.extras['state_extras']['truncation']
    q_error *= jnp.expand_dims(1 - truncation, -1)

    q_loss = 0.5 * jnp.mean(jnp.square(q_error))

    if use_sum_cost_critic:
      proportion_unsafe = jnp.where(next_cqs < safety_threshold, 0.0, 1.0).mean()
    else:
      proportion_unsafe = jnp.where(next_cqs > safety_threshold, 0.0, 1.0).mean()

    proportion_transition = transitions.extras["state_extras"]["made_transition"].mean()

    # jax.lax.cond(proportion_transition > 0.0,
    #              lambda trns: jax.debug.breakpoint(),
    #              lambda trns: None,
    #              transitions)

    return q_loss, {
      "target_q": target_q,
      "mean_reward": transitions.reward.mean(),
      "proportion_transition": proportion_transition,
      "proportion_unsafe": proportion_unsafe
    }

  def cost_critic_loss(
      cost_q_params: Params,
      option_q_params: Params,
      normalizer_params: Any,
      target_cost_q_params: Params,
      target_option_q_params: Params,
      transitions: Transition,
      cost_gamma: float,
      key: PRNGKey
  ) -> jnp.ndarray:

    state_obs, goals, aut_state = env.split_obs(transitions.observation)
    _, aut_state_obs = env.split_aut_obs(transitions.observation)

    next_state_obs, next_goals, next_aut_state = env.split_obs(transitions.next_observation)
    _, next_aut_state_obs = env.split_aut_obs(transitions.next_observation)

    # Double Q(s_t, o_t) for all options

    qc_input = transitions.observation
    cqs_old = cost_q_network.apply(normalizer_params, cost_q_params, qc_input)
    cq_old_action = jax.vmap(lambda x, i: x.at[i].get())(cqs_old, transitions.action)

    # Q1(s_t+1, o_t+1)/Q2(s_t+1, o_t+1) for all options
    next_qr_input = transitions.next_observation
    next_double_qs = option_q_network.apply(normalizer_params, target_option_q_params, next_qr_input)

    next_qc_input = transitions.next_observation
    next_double_cqs = cost_q_network.apply(normalizer_params, target_cost_q_params, next_qc_input)

    # Q(s_t+1, o_t+1) for all options
    next_qs = jnp.min(next_double_qs, axis=-1)

    if use_sum_cost_critic:
      next_cqs = jnp.max(next_double_cqs, axis=-1)
    else:
      next_cqs = jnp.min(next_double_cqs, axis=-1)

    # V(s_t+1) = max_o Q(s_t+1, o) (because pi is argmax Q)
    _, next_cv = safe_greedy_policy(next_qs, next_cqs, key)

    # E (s_t, a_t, s_t+1) ~ D [ gamma * min(c(s_t, a_t), V(s_t+1)) + (1-gamma) * c(s_t, a_t) ]

    if use_sum_cost_critic:
      target_cq = jax.lax.stop_gradient(transitions.extras["state_extras"]["cost"] * cost_scaling +
                                        transitions.discount * discounting * next_cv)
    else:
      target_cq = jax.lax.stop_gradient(
        cost_gamma * jnp.min(
          jnp.stack((transitions.extras["state_extras"]["cost"] * cost_scaling,
                     transitions.discount * next_cv), axis=-1),
          axis=-1
        ) + (1 - cost_gamma) * transitions.extras["state_extras"]["cost"]
      )

    # Q(s_t, a_t) - E[r(s_t, a_t) + gamma * V(s_t+1)]
    cq_error = cq_old_action - jnp.expand_dims(target_cq, -1)

    # Better bootstrapping for truncated episodes.
    truncation = transitions.extras['state_extras']['truncation']
    cq_error *= jnp.expand_dims(1 - truncation, -1)

    cq_loss = 0.5 * jnp.mean(jnp.square(cq_error))

    return cq_loss, {
      "target_cq": target_cq,
      "max_cost": transitions.extras["state_extras"]["cost"].max(),
      "mean_cost": transitions.extras["state_extras"]["cost"].mean(),
      "min_cost": transitions.extras["state_extras"]["cost"].min(),
    }

  return critic_loss, cost_critic_loss
