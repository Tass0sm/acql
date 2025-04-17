"""Hierarchical DCQN losses.
"""
from typing import Any

from brax import envs
from brax.training import types
from brax.training.types import Params
from brax.training.types import PRNGKey
import jax
import jax.numpy as jnp

from achql.brax.agents.achql import networks as achql_networks

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
        argmax_type: str = "plain",
):
  """Creates the ACHQL losses."""

  option_q_network = achql_network.option_q_network
  cost_q_network = achql_network.cost_q_network
  q_func_branches = achql_networks.get_compiled_q_function_branches(achql_network, env)

  def safe_greedy_policy(reward_qs, cost_qs, option_key):
    "Finds option with maximal value under cost constraint"

    if use_sum_cost_critic:
      masked_q = jnp.where(cost_qs < safety_threshold, reward_qs, -999.)
    else:
      masked_q = jnp.where(cost_qs > safety_threshold, reward_qs, -999.)

    if argmax_type == "random":
      option = argmax_with_random_tiebreak(masked_q, option_key, axis=-1)
    elif argmax_type == "safest":
      option = argmax_with_safest_tiebreak(masked_q, cost_qs, axis=-1, use_sum_cost_critic=use_sum_cost_critic)
    elif argmax_type == "plain":
      option = masked_q.argmax(axis=-1)
    else:
      raise NotImplementedError()

    q = jax.vmap(lambda x, i: x.at[i].get())(reward_qs, option)
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
    batched_qr = jax.vmap(lambda a_s, o: jax.lax.switch(a_s, q_func_branches, (normalizer_params, option_q_params), o))
    qs_old = batched_qr(aut_state, qr_input)
    q_old_action = jax.vmap(lambda x, i: x.at[i].get())(qs_old, transitions.action)

    # Q1(s_t+1, o_t+1)/Q2(s_t+1, o_t+1) for all options
    next_qr_input = transitions.next_observation
    batched_target_qr = jax.vmap(lambda a_s, o: jax.lax.switch(a_s, q_func_branches, (normalizer_params, target_option_q_params), o))
    next_double_qs = batched_target_qr(next_aut_state, next_qr_input)

    next_qc_input = jnp.concatenate((next_state_obs, next_aut_state_obs), axis=-1)
    cost_observation_size = state_obs.shape[-1]
    trimmed_normalizer_params = jax.tree.map(lambda x: x[..., :cost_observation_size] if x.ndim >= 1 else x, normalizer_params)
    batched_target_qc = jax.vmap(lambda a_s, o: cost_q_network.apply(trimmed_normalizer_params, target_cost_q_params, o))
    next_double_cqs = batched_target_qc(next_aut_state, next_qc_input)

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

    return q_loss, {
      "target_q": target_q,
      "mean_reward": transitions.reward.mean(),
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

    qc_input = jnp.concatenate((state_obs, aut_state_obs), axis=-1)
    cost_observation_size = state_obs.shape[-1]
    trimmed_normalizer_params = jax.tree.map(lambda x: x[..., :cost_observation_size] if x.ndim >= 1 else x, normalizer_params)
    batched_qc = jax.vmap(lambda a_s, o: cost_q_network.apply(trimmed_normalizer_params, cost_q_params, o))
    cqs_old = batched_qc(aut_state, qc_input)
    cq_old_action = jax.vmap(lambda x, i: x.at[i].get())(cqs_old, transitions.action)

    # Q1(s_t+1, o_t+1)/Q2(s_t+1, o_t+1) for all options
    next_qr_input = transitions.next_observation
    batched_target_qr = jax.vmap(lambda a_s, o: jax.lax.switch(a_s, q_func_branches, (normalizer_params, target_option_q_params), o))
    next_double_qs = batched_target_qr(next_aut_state, next_qr_input)

    next_qc_input = jnp.concatenate((next_state_obs, next_aut_state_obs), axis=-1)
    batched_target_qc = jax.vmap(lambda a_s, o: cost_q_network.apply(trimmed_normalizer_params, target_cost_q_params, o))
    next_double_cqs = batched_target_qc(next_aut_state, next_qc_input)

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
      "mean_cost": transitions.extras["state_extras"]["cost"].mean(),
    }

  return critic_loss, cost_critic_loss
