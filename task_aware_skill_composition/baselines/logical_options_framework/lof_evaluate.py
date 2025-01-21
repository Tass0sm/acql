import functools
import jax
import os
import pandas as pd
import mlflow
import pickle

from datetime import datetime
from jax import numpy as jnp
import matplotlib.pyplot as plt

from IPython.display import HTML, clear_output
import mediapy

import brax

import flax
from brax import envs
from brax.io import model
from brax.io import json
from brax.io import html

# other envs
from task_aware_skill_composition.brax.envs.point import Point
from task_aware_skill_composition.brax.envs.car import Car
from task_aware_skill_composition.brax.envs.drone import Drone
from task_aware_skill_composition.brax.envs.point import Point
from task_aware_skill_composition.brax.envs.doggo import Doggo
from task_aware_skill_composition.brax.envs.xy_point_maze import XYPointMaze

from task_aware_skill_composition.brax.safety_gymnasium_envs.safety_hopper_velocity_v1 import SafetyHopperVelocityEnv

from task_aware_skill_composition.brax.agents.ppo import train as ppo
from task_aware_skill_composition.brax.agents.dscrl import train as dscrl
from task_aware_skill_composition.brax.agents.ppo_with_spec_rewards import train as ppo_spec
from task_aware_skill_composition.brax.agents.hsac import train as hsac

from brax.training.agents.sac import train as sac
from task_aware_skill_composition.brax.agents.sac_lagrangian import train as sac_lagrangian

from task_aware_skill_composition.brax.agents.ddpg import train as ddpg
from task_aware_skill_composition.brax.agents.ddpg_lagrangian import train as ddpg_lagrangian

from task_aware_skill_composition.brax.agents.hdqn import train as hdqn
from task_aware_skill_composition.brax.agents.hdqn_her import train as hdqn_her
from task_aware_skill_composition.brax.agents.hdqn_automaton_her import train as hdqn_automaton_her
from task_aware_skill_composition.brax.agents.hdcqn_her import train as hdcqn_her
from task_aware_skill_composition.brax.agents.hdcqn_automaton_her import train as hdcqn_automaton_her

from task_aware_skill_composition.brax.agents.sac_her import train as sac_her
from task_aware_skill_composition.brax.agents.sac_her_lagrangian import train as sac_her_lagrangian

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.sac import networks as sac_networks
from task_aware_skill_composition.hierarchy.option_critic import networks as oc_networks
from task_aware_skill_composition.brax.agents.hsac import networks as hsac_networks
from task_aware_skill_composition.brax.agents.hdqn import networks as hdq_networks
from task_aware_skill_composition.brax.agents.hdcqn import networks as hdcq_networks
from task_aware_skill_composition.brax.agents.hdqn_automaton_her import networks as hdq_aut_networks
from task_aware_skill_composition.brax.agents.hdcqn_automaton_her import networks as hdcq_aut_networks

from task_aware_skill_composition.baselines.logical_options_framework import train as lof

from jaxgcrl.src import networks as crl_networks
from brax.training.acme import running_statistics

# tasks
from task_aware_skill_composition.brax.tasks import get_task
from task_aware_skill_composition.brax.utils import make_cmdp, make_aut_goal_cmdp, make_shaped_reward_mdp, make_shaped_reward_mdp2, make_aut_goal_mdp

mlflow.set_tracking_uri(f"file:///home/tassos/.local/share/mlflow")

# %% [markdown]
# # Defining STL

# %%
import corallab_stl.expression_jax2 as stl

# %% [markdown]
# # Defining Env/Task

# %%
mlflow.set_experiment("proj2-notebook")

# %%
backend = 'mjx'

task = get_task("ant_maze", "two_subgoals")

env = task.env

spec = task.lo_spec
spec_tag = type(task).__name__
env_tag = type(env).__name__

options = task.get_options()

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

# %%
# env = SafetyHopperVelocityEnv(backend="mjx")
# spec = stl.STLPredicate(stl.Var(name="obs", dim=env.observation_size), lambda s: 999, lower_bound=0.0)
# spec_tag = "SafeVelocity"
# env_tag = "Hopper"

# jit_reset = jax.jit(env.reset)
# jit_step = jax.jit(env.step)

# %%
print(spec_tag)
print(spec)
print(env.observation_size)
print(env.action_size)
alg_suffix = ""

training_run_id = "aa333aa2590b449988df49e8d24d413c"
logged_model_path = f'runs:/{training_run_id}/policy_params'
real_path = mlflow.artifacts.download_artifacts(logged_model_path)
params = model.load_params(real_path)
# normalizer_params, policy_params = model.load_params(real_path)
# normalizer_params, policy_params, crl_critic_params = model.load_params(real_path)

run = mlflow.get_run(run_id=training_run_id)
if run.data.params["normalize_observations"] == "True":
    normalize_fn = running_statistics.normalize
    cost_normalize_fn = running_statistics.normalize
else:
    normalize_fn = lambda x, y: x
    cost_normalize_fn = lambda x, y: x


make_policy = lof.make_inference_fn(prod_mdp, )

# ## Or from the most recent run from the current session

# make_policy = make_inference_fn

# inference_fn = make_policy(params, deterministic=True)
# jit_inference_fn = jax.jit(inference_fn)

from task_aware_skill_composition.visualization import hierarchy
from task_aware_skill_composition.visualization import flat

def visualize_loaded_policy_rollout():
    inference_fn = make_policy(params, deterministic=True)
    jit_inference_fn = jax.jit(inference_fn)

    rollout, opt_traj, action = hierarchy.get_rollout(aut_goal_cmdp, jit_inference_fn, options, n_steps=100, seed=4, render_every=1)

    mediapy.write_video(f'./hierarchical-rollout.mp4',
                        env.render(
                            [s.pipeline_state for s in rollout],
                            camera='overview'
                        ), fps=1.0 / env.dt)


def visualize_option_rollouts():
    for i in range(0, 1):
        inference_fn = jax.jit(options[i].inference)
    
        rollout, actions = flat.get_rollout(env, inference_fn, n_steps=50, render_every=1, seed=1)
    
        mediapy.write_video(f'./option{i}-rollout.mp4',
                            env.render(
                                [s.pipeline_state for s in rollout],
                                camera='overview'
                            ), fps=1.0 / env.dt)


visualize_loaded_policy_rollout()

### With Options

# rollout, opt_traj, action = get_rollout(make_shaped_reward_mdp2(task, 1), jit_inference_fn, options, n_steps=300, seed=0, render_every=1)
# rollout, opt_traj, action = get_rollout(aut_goal_mdp, jit_inference_fn, options, n_steps=300, seed=0, render_every=1)
# rollout, opt_traj, action = get_rollout(aut_goal_cmdp, jit_inference_fn, options, n_steps=300, seed=1, render_every=1)
# rollout, opt_traj, action = get_rollout(make_cmdp(task), jit_inference_fn, options, n_steps=300, seed=3, render_every=1)
# rollout, opt_traj, action = get_rollout(env, jit_inference_fn, options, n_steps=600, seed=4, render_every=1)

# %%
# mediapy.show_video(

# stats

# obs_traj = jnp.stack([state.obs for state in rollout])
# position_traj = jnp.stack([state.obs[:2] for state in rollout])
# reward_traj = jnp.stack([state.reward for state in rollout])

# automata_state_traj = jnp.stack([state.info["automata_state"] for state in rollout])
# made_transition_traj = jnp.stack([state.info["made_transition"] for state in rollout])


# option_traj = jnp.stack(options)

# from jax.numpy.linalg import norm
# print(norm(position_traj - jnp.array([2.0, 0.0]), axis=1))
# print(delta_good_edge_satisfaction_state_traj)
# automaton_wrapped_env.automaton.one_hot_decode(observations[:, -3:])
