"""Hierarchical DCQN losses.
"""
from typing import Any

from brax import envs
from brax.training import types
from brax.training.types import Params
from brax.training.types import PRNGKey
import jax
import jax.numpy as jnp

from acql.brax.agents.acddpg import networks as acddpg_networks


Transition = types.Transition


def make_losses(
        acddpg_network: acddpg_networks.ACDDPGNetworks,
        env: envs.Env,
        reward_scaling: float,
        cost_scaling: float,
        safety_threshold: float,
        discounting: float,
        use_sum_cost_critic: bool = False,
        actor_type: str = "argmax",
):
  """Creates the ACQL losses."""

  policy_network = acddpg_network.policy_network
  q_network = acddpg_network.q_network
  cost_q_network = acddpg_network.cost_q_network
  parametric_action_distribution = acddpg_network.parametric_action_distribution

  def lambda_loss(
      lambda_multiplier: jnp.ndarray,
      cost_q_params: Params,
      normalizer_params: Any,
      transitions: Transition,
      key: PRNGKey) -> jnp.ndarray:
    """Eq 7 from https://arxiv.org/pdf/2002.08550"""

    # First version based on: https://github.com/ammarhydr/SAC-Lagrangian/blob/232a0772205007068449740e0a0f1b7ddca3d946/SAC_Agent.py#L278
    # log_lambda = jax.nn.softplus(lambda_multiplier)
    # used cost critic

    # Second version based on omnisafe: https://github.com/PKU-Alignment/omnisafe/blob/main/omnisafe/algorithms/off_policy/sac_lag.py#L97
    # uses mean cost from batch

    if use_sum_cost_critic:
      raise NotImplementedError()
    else:
      violation = jnp.mean(transitions.extras["state_extras"]["cost"] - safety_threshold)
    lambda_loss = lambda_multiplier * jax.lax.stop_gradient(violation)

    return jnp.sum(lambda_loss), {
      # "cost_q": jnp.mean(min_cost_q),
      "mean_violation": violation,
    }

  def critic_loss(q_params: Params,
                  policy_params: Params,
                  normalizer_params: Any,
                  target_q_params: Params,
                  target_cost_q_params: Params,
                  transitions: Transition,
                  key: PRNGKey) -> jnp.ndarray:
    q_old_action = q_network.apply(normalizer_params, q_params,
                                   transitions.observation, transitions.action)
    next_dist_params = policy_network.apply(normalizer_params, policy_params,
                                            transitions.next_observation)
    next_action = parametric_action_distribution.sample_no_postprocessing(
      next_dist_params, key)
    next_action = parametric_action_distribution.postprocess(next_action)
    next_q = q_network.apply(normalizer_params, target_q_params,
                             transitions.next_observation, next_action)

    next_qc_input = transitions.next_observation
    next_double_cq = cost_q_network.apply(normalizer_params, target_cost_q_params, next_qc_input, next_action)

    if use_sum_cost_critic:
      next_cq = jnp.max(next_double_cq, axis=-1)
      next_v = jnp.min(next_q, axis=-1)
      # next_v = jnp.where(next_cq < safety_threshold, next_v, 0.0)
    else:
      next_cq = jnp.min(next_double_cq, axis=-1)
      next_v = jnp.min(next_q, axis=-1)
      # next_v = jnp.where(next_cq > safety_threshold, next_v, 0.0)

    target_q = jax.lax.stop_gradient(transitions.reward * reward_scaling + transitions.discount * discounting *next_v)
    q_error = q_old_action - jnp.expand_dims(target_q, -1)

    # Better bootstrapping for truncated episodes.
    truncation = transitions.extras['state_extras']['truncation']
    q_error *= jnp.expand_dims(1 - truncation, -1)

    q_loss = 0.5 * jnp.mean(jnp.square(q_error))

    if use_sum_cost_critic:
      proportion_unsafe = jnp.where(next_cq < safety_threshold, 0.0, 1.0).mean()
    else:
      proportion_unsafe = jnp.where(next_cq > safety_threshold, 0.0, 1.0).mean()

    return q_loss, {
      "target_q": target_q,
      "proportion_unsafe": proportion_unsafe
    }

  def cost_critic_loss(
      cost_q_params: Params,
      policy_params: Params,
      normalizer_params: Any,
      target_cost_q_params: Params,
      transitions: Transition,
      cost_gamma: float,
      key: PRNGKey
  ) -> jnp.ndarray:
    state_obs, goals, aut_state = env.split_obs(transitions.observation)
    _, aut_state_obs = env.split_aut_obs(transitions.observation)

    next_state_obs, next_goals, next_aut_state = env.split_obs(transitions.next_observation)
    _, next_aut_state_obs = env.split_aut_obs(transitions.next_observation)

    qc_input = transitions.observation
    cq_old_action = cost_q_network.apply(normalizer_params, cost_q_params, qc_input, transitions.action)

    next_dist_params = policy_network.apply(normalizer_params, policy_params, transitions.next_observation)
    next_action = parametric_action_distribution.sample_no_postprocessing(next_dist_params, key)
    next_action = parametric_action_distribution.postprocess(next_action)

    next_qc_input = transitions.next_observation
    next_cost_q = cost_q_network.apply(normalizer_params, target_cost_q_params, next_qc_input, next_action)
    # next_cost_v = jnp.min(next_cost_q, axis=-1)

    if use_sum_cost_critic:
      next_cv = jnp.max(next_cost_q, axis=-1)
    else:
      next_cv = jnp.min(next_cost_q, axis=-1)

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

    cq_error = cq_old_action - jnp.expand_dims(target_cq, -1)

    # Better bootstrapping for truncated episodes.
    truncation = transitions.extras['state_extras']['truncation']
    cq_error *= jnp.expand_dims(1 - truncation, -1)

    cq_loss = 0.5 * jnp.mean(jnp.square(cq_error))

    return cq_loss, {
      "target_cq": target_cq,
      "mean_cost": transitions.extras["state_extras"]["cost"].mean(),
      # "min_cost_q_old_action": jnp.min(cost_q_old_action),
      # "mean_cost_q_old_action": jnp.mean(cost_q_old_action),
      # "max_cost_q_old_action": jnp.max(cost_q_old_action),
      # "min_target_cost_q": jnp.min(target_cost_q),
      # "mean_target_cost_q": jnp.mean(target_cost_q),
      # "max_target_cost_q": jnp.max(target_cost_q),
      # "mean_cost_over_batch": jnp.mean(transitions.extras["state_extras"]["cost"])
    }

  def actor_loss(policy_params: Params,
                 normalizer_params: Any,
                 q_params: Params,
                 cost_q_params: Params,
                 lambda_multiplier: jnp.ndarray,
                 transitions: Transition,
                 key: PRNGKey) -> jnp.ndarray:

    state_obs, goals, aut_state = env.split_obs(transitions.observation)
    _, aut_state_obs = env.split_aut_obs(transitions.observation)

    dist_params = policy_network.apply(normalizer_params, policy_params, transitions.observation)

    diff_action_raw = parametric_action_distribution.sample_no_postprocessing(
      dist_params, key)
    diff_action = parametric_action_distribution.postprocess(diff_action_raw)

    truncation_mask = 1 - transitions.extras['state_extras']['truncation']

    # Q(s, pi(s))
    q_action = q_network.apply(normalizer_params, q_params,
                               transitions.observation, diff_action)
    min_q = jnp.min(q_action, axis=-1) * truncation_mask

    # Qc(s, pi(s))
    qc_input = transitions.observation
    cost_q_action = cost_q_network.apply(normalizer_params, cost_q_params, qc_input, diff_action)

    if use_sum_cost_critic:
      raise NotImplementedError()
    else:
      min_cost_q = jnp.min(cost_q_action, axis=-1)

    actor_loss = -min_q + (lambda_multiplier * min_cost_q)
    return jnp.mean(actor_loss), {
      'raw_action_mean': jnp.mean(diff_action_raw),
      'raw_action_std': jnp.std(diff_action_raw),
      'sampled_action_mean': jnp.mean(diff_action),
      'sampled_action_std': jnp.std(diff_action),
    }

  return lambda_loss, critic_loss, cost_critic_loss, actor_loss
