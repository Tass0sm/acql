import functools
import jax
import os
from typing import Optional

from datetime import datetime
from jax import numpy as jnp
import matplotlib.pyplot as plt

import brax

import flax
from brax import envs
from brax.io import model
from brax.io import json
from brax.io import html
from brax.training.agents.sac import train as sac
from jaxgcrl.src import train as crl

from task_aware_skill_composition.brax.agents.ppo import train as ppo
from task_aware_skill_composition.brax.agents.tasc import train as tasc
from task_aware_skill_composition.brax.agents.dscrl import train as dscrl
from task_aware_skill_composition.brax.agents.ppo_with_spec_rewards import train as ppo_spec
from task_aware_skill_composition.brax.agents.sac_lagrangian import train as sac_lagrangian

from task_aware_skill_composition.brax.safety_gymnasium_envs.safety_hopper_velocity_v1 import SafetyHopperVelocityEnv


# tasks
from task_aware_skill_composition.brax.tasks import get_task

import corallab_stl.expression_jax2 as stl


# task = get_task("xy_point", "obstacle")
# env = task.env

# spec = task.lo_spec
# spec_tag = type(task).__name__
# env_tag = type(env).__name__

# jit_reset = jax.jit(env.reset)
# jit_step = jax.jit(env.step)


env = SafetyHopperVelocityEnv(backend="mjx")
obs_var = stl.Var(name="obs", dim=env.observation_size)
spec = stl.STLPredicate(obs_var, lambda s: 999, lower_bound=0.0)
spec_tag = "SafeVelocity"
env_tag = "Hopper"

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)


def progress_fn(num_steps, metrics, df=None):
    print(num_steps)
    print(metrics)


train_fn = functools.partial(sac_lagrangian.train,
                             num_timesteps=10_000_000,
                             reward_scaling=10,
                             cost_scaling=10,
                             num_evals=50,
                             episode_length=1000,
                             normalize_observations=True,
                             action_repeat=1,
                             unroll_length=62,
                             multiplier_num_sgd_steps=1,
                             max_devices_per_host=1,
                             max_replay_size=10000,
                             min_replay_size=1000,
                             use_her=False,
                             discounting=0.97,
                             num_envs=512,
                             batch_size=256)

# train_fn = functools.partial(ppo.train,
#                              num_timesteps=50_000_000,
#                              num_evals=10,
#                              reward_scaling=10,
#                              episode_length=1000,
#                              normalize_observations=True,
#                              action_repeat=1,
#                              unroll_length=5,
#                              num_minibatches=32,
#                              num_updates_per_batch=4,
#                              discounting=0.97,
#                              learning_rate=3e-4,
#                              entropy_cost=1e-2,
#                              # specification_cost=1e-3,
#                              num_envs=2048,
#                              batch_size=512,
#                              seed=1)

# train_fn = functools.partial(crl.train,
#                              num_timesteps=10_000_000,
#                              max_replay_size=10_000,
#                              min_replay_size=1_000,
#                              num_evals=50,
#                              episode_length=1000,
#                              action_repeat=1,
#                              discounting=0.99,
#                              num_envs=512,
#                              batch_size=256,
#                              unroll_length=62,
#                              multiplier_num_sgd_steps=1,
#                              normalize_observations=False,
#                              policy_lr=6e-4,
#                              alpha_lr=3e-4,
#                              critic_lr=3e-4,
#                              contrastive_loss_fn='infonce_backward',
#                              energy_fn='l2',
#                              use_ln=False,
#                              use_c_target=False,
#                              logsumexp_penalty=0.0,
#                              l2_penalty=0.0,
#                              exploration_coef=0.0,
#                              random_goals=0.0,
#                              disable_entropy_actor=False,
#                              eval_env=None,
#                              h_dim=256,
#                              n_hidden=2,
#                              repr_dim=64)

# TRAIN WITH SPEC REWARDS

# from task_aware_skill_composition.brax.envs.wrappers.specification_reward_wrapper import SpecificationRewardWrapper

# spec_wrapped_env = SpecificationRewardWrapper(
#     env,
#     task.lo_spec,
#     task.obs_var,
#     rho_weight=1.0
# )

# make_inference_fn, params, _ = train_fn(
#     environment=spec_wrapped_env,
#     progress_fn=functools.partial(progress_fn),
#     specification=task.lo_spec,
#     state_var=task.obs_var,
# )

# TRAIN WITH AUTOMATON

# from task_aware_skill_composition.brax.envs.wrappers.automaton_wrapper import AutomatonWrapper

# automaton_wrapped_env = AutomatonWrapper(
#     env,
#     spec,
#     task.obs_var,
#     augment_obs = True
# )


# Making an automaton augmented MDP
from task_aware_skill_composition.brax.envs.wrappers.automaton_wrapper import AutomatonWrapper
from task_aware_skill_composition.brax.envs.wrappers.automaton_cost_wrapper import AutomatonCostWrapper

# automaton_wrapped_env = AutomatonWrapper(
#     env,
#     spec,
#     task.obs_var,
#     augment_obs = True
# )

# # Making a CMDP
# cmdp = AutomatonCostWrapper(
#     automaton_wrapped_env
# )

make_inference_fn, params, _ = train_fn(
    environment=env,
    progress_fn=progress_fn,
    specification=spec, # task.lo_spec,
    state_var=obs_var, # task.obs_var,
)

# make_inference_fn, params = training_run(
#     run.info.run_id,
#     env,
#     seed,
#     train_fn=ppo_option_critic.train,
#     hyperparameters=hppo_hps,
#     extras={
#         "options": options,
#         "specification": spec,
#         "state_var": task.obs_var,
#     }
# )

# print(f'time to jit: {times[1] - times[0]}')
# print(f'time to train: {times[-1] - times[1]}')
