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
from jaxgcrl.src import train as crl

from task_aware_skill_composition.brax.agents.ppo import train as ppo
from task_aware_skill_composition.brax.agents.tasc import train as tasc
from task_aware_skill_composition.brax.agents.dscrl import train as dscrl
from task_aware_skill_composition.brax.agents.sac_her import train as sac_her
from task_aware_skill_composition.brax.agents.ppo_with_spec_rewards import train as ppo_spec
from task_aware_skill_composition.brax.agents.ppo_option_critic import train as ppo_option_critic
from task_aware_skill_composition.brax.agents.hsac import train as soac
from task_aware_skill_composition.brax.agents.hdqn import train as hdqn
from task_aware_skill_composition.brax.agents.hdqn_her import train as hdqn_her
from task_aware_skill_composition.brax.agents.hdqn_automaton_her import train as hdqn_automaton_her
from task_aware_skill_composition.brax.agents.hdcqn_her import train as hdcqn_her
from task_aware_skill_composition.brax.agents.hdcqn_automaton_her import train as hdcqn_automaton_her
from brax.training.agents.apg import train as apg

from task_aware_skill_composition.baselines.reward_machines.crm import train as crm
# from task_aware_skill_composition.baselines.reward_machines.qrm import train as qrm
from task_aware_skill_composition.baselines.reward_machines.hrm import train as hrm
from task_aware_skill_composition.baselines.reward_machines.reward_shaping import train as rm_reward_shaping

# tasks
from task_aware_skill_composition.brax.tasks import get_task
from task_aware_skill_composition.brax.utils import (
    make_cmdp, make_aut_goal_cmdp, make_shaped_reward_mdp, make_shaped_reward_mdp2,
    make_aut_mdp,
    make_reward_machine_mdp,
    make_aut_goal_mdp,
)

from task_aware_skill_composition.visualization.critic import (
    make_plots_for_hdqn,
    make_plots_for_hdcqn,
    make_plots_for_hdcqn_aut,
)


# working with default task

# task = get_task("simple_maze", "subgoal")
# task = get_task("simple_maze", "two_subgoals")
# task = get_task("simple_maze", "branching1")
# task = get_task("simple_maze", "obligation2")

# task = get_task("simple_maze_3d", "two_subgoals")
# task = get_task("simple_maze_3d", "branching1")
# task = get_task("simple_maze_3d", "obligation2")

# task = get_task("ant_maze", "two_subgoals")
# task = get_task("ant_maze", "branching1")
# task = get_task("ant_maze", "obligation2")

# task = get_task("simple_maze_3d", "subgoal")

# task = get_task("ant_maze", "true")
# task = get_task("ant_maze", "umaze_constraint")

task = get_task("ur5e", "reach_shelf")


env = task.env
spec = task.lo_spec
spec_tag = type(task).__name__
env_tag = type(env).__name__

print(spec_tag)
print(env_tag)

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

def progress_fn(num_steps, metrics, **kwargs):
    print(num_steps)
    print(metrics)

# def hdqn_progress_fn(num_steps, metrics, **kwargs):
#     print(f"Logging for {num_steps}")
#     print(metrics)

#     env = kwargs["env"]
#     network = kwargs["network"]
#     params = kwargs["params"]

#     make_plots_for_hdqn(
#       env=env,
#       network=network,
#       params=params,
#       label=f"initial state - {num_steps}",
#     )

# def hdcqn_progress_fn(num_steps, metrics, **kwargs):
#     print(f"Logging for {num_steps}")
#     print(metrics)

#     env = kwargs["env"]
#     make_policy = kwargs["make_policy"]
#     network = kwargs["network"]
#     params = kwargs["params"]

#     make_plots_for_hdcqn_aut(
#       env=env,
#       make_policy=make_policy,
#       network=network,
#       params=params,
#       label=f"initial state 1 - {num_steps}",
#     )

    # make_plots_for_hdcqn_aut(
    #   env=env,
    #   make_policy=make_policy,
    #   network=network,
    #   params=params,
    #   tmp_state_fn=lambda x: x.replace(obs=x.obs.at[4:].set(jnp.array([4.0, 12.0, 0., 0., 1.]))),
    #   label=f"next state 2 - {num_steps}",
    # )    


options = task.get_options()

extras = {
    "options": options,
    "specification": spec,
    "state_var": task.obs_var,
    # "replay_buffer_class_name": "TrajectoryUniformSamplingQueue",
    # "replay_buffer_class_name": "AutomatonTrajectoryUniformSamplingQueue",
}

seed = 0

sac_her_hps = {
    "num_timesteps": 6_000_000,
    "reward_scaling": 1,
    "num_evals": 6,
    "episode_length": 1000,
    "normalize_observations": True,
    "action_repeat": 1,
    "discounting": 0.99,
    # "learning_rate": 3e-4,
    "num_envs": 512,
    "batch_size": 256,
    # "unroll_length": 62,
    "max_devices_per_host": 1,
    "max_replay_size": 10000,
    # 8192, the default, causes the error "TypeError: broadcast_in_dim shape must have every element be nonnegative, got (-2, 50)."
    "min_replay_size": 1000,
    "use_her": True,
}

# train_fn = functools.partial(hdcqn_her.train, **task.hdcqn_her_hps)
# train_fn = functools.partial(hdqn_her.train, **task.hdqn_her_hps)

# hps = task.hdcqn_her_hps # | { "num_timesteps": 2_500_000 }
# train_fn = functools.partial(hdcqn_automaton_her.train, **hps)
# train_fn = functools.partial(rm_reward_shaping.train, **task.rm_reward_shaping_hps)
# train_fn = functools.partial(crm.train, **task.crm_hps)
# train_fn = functools.partial(hrm.train, **task.hrm_hps)
# train_fn = functools.partial(hdqn_her.train, **task.hdqn_her_hps)
# train_fn = functools.partial(hdqn_automaton_her.train, **task.hdqn_her_hps)
# train_fn = functools.partial(hdqn.train, **hdqn_hps)
train_fn = functools.partial(sac_her.train, **sac_her_hps)

make_inference_fn, params, _ = train_fn(
    # environment=make_aut_goal_cmdp(task),
    # environment=make_reward_machine_mdp(task),
    environment=env,
    # progress_fn=hdcqn_progress_fn,
    # progress_fn=hdqn_progress_fn,
    progress_fn=progress_fn,
    seed=seed,
    # eval_env=make_aut_goal_cmdp(task, randomize_goals=False),
    **extras
)
