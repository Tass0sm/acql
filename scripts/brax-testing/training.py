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

from task_aware_skill_composition.brax.agents.ppo import train as ppo
from task_aware_skill_composition.brax.agents.tasc import train as tasc
from task_aware_skill_composition.brax.agents.dscrl import train as dscrl
from task_aware_skill_composition.brax.agents.ppo_with_spec_rewards import train as ppo_spec

# tasks
from task_aware_skill_composition.brax.tasks import get_task


task = get_task("point", "turn_seq")
env = task.env

spec = task.lo_spec
spec_tag = type(task).__name__
env_tag = type(env).__name__

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

def progress_fn(num_steps, metrics, df=None):
    pass

train_fn = functools.partial(tasc.train,
                             num_timesteps=50_000_000,
                             num_evals=10,
                             reward_scaling=10,
                             episode_length=1000,
                             normalize_observations=True,
                             action_repeat=1,
                             unroll_length=5,
                             num_minibatches=32,
                             num_updates_per_batch=4,
                             discounting=0.97,
                             learning_rate=3e-4,
                             entropy_cost=1e-2,
                             # specification_cost=1e-3,
                             num_envs=2048,
                             batch_size=512,
                             seed=1)

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

from task_aware_skill_composition.brax.envs.wrappers.automaton_wrapper import AutomatonWrapper

automaton_wrapped_env = AutomatonWrapper(
    env,
    spec,
    task.obs_var,
    augment_obs = True
)

make_inference_fn, params, _ = train_fn(
    environment=automaton_wrapped_env,
    progress_fn=functools.partial(progress_fn),
    specification=task.lo_spec,
    state_var=task.obs_var,
)


# print(f'time to jit: {times[1] - times[0]}')
# print(f'time to train: {times[-1] - times[1]}')
