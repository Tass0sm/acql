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
mlflow.set_experiment("proj2-hierarchy-comparison")

# %%
backend = 'mjx'

# task = get_task("simple_maze", "branching1")
# task = get_task("simple_maze", "subgoal")
# task = get_task("simple_maze_3d", "subgoal")
# task = get_task("ant_maze", "true")
# task = get_task("ur5e", "reach")
task = get_task("ur5e", "translate")

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

# training_run_id = "ea6fc1f281494ab4a184d425787a3a08"
# logged_model_path = f'runs:/{training_run_id}/policy_params'
# real_path = mlflow.artifacts.download_artifacts(logged_model_path)
# params = model.load_params(real_path)
# # normalizer_params, policy_params = model.load_params(real_path)
# # normalizer_params, policy_params, crl_critic_params = model.load_params(real_path)

# run = mlflow.get_run(run_id=training_run_id)
# if run.data.params["normalize_observations"] == "True":
#     normalize = running_statistics.normalize
#     cost_normalize_fn = running_statistics.normalize
# else:
#     normalize_fn = lambda x, y: x
#     cost_normalize_fn = lambda x, y: x

# # Making the network
# ppo_network = ppo_networks.make_ppo_networks(
#       state.obs.shape[0],
#       env.action_size,
#       preprocess_observations_fn=normalize
# )
# make_policy = ppo_networks.make_inference_fn(ppo_network)

# # Making the network
# oc_network = oc_networks.make_option_critic_networks(
#       state.obs.shape[0],
#       env.action_size,
#       options=options,
#       preprocess_observations_fn=normalize,
# )
# make_policy = oc_networks.make_inference_fn(oc_network)

# # Making the network
# hsac_network = hsac_networks.make_soac_networks(
#       env.observation_size,
#       env.action_size,
#       options=options,
#       preprocess_observations_fn=normalize,
# )
# make_policy = hsac_networks.make_inference_fn(hsac_network)

# options = task.get_hard_coded_options()
# hdq_network = hdq_networks.make_hdq_networks(
#       env.observation_size,
#       env.action_size,
#       options=options,
#       preprocess_observations_fn=normalize,
# )
# make_policy = hdq_networks.make_inference_fn(hdq_network)

# aut_goal_mdp = make_aut_goal_mdp(task)
# options = task.get_hard_coded_options()
# hdq_aut_network = hdq_networks.make_hdq_networks(
#     aut_goal_mdp.observation_size,
#     aut_goal_mdp.action_size,
#     options=options,
#     preprocess_observations_fn=normalize,
# )
# make_policy = hdq_aut_networks.make_inference_fn(hdq_aut_network, aut_goal_mdp)

# hdcq_network = hdcq_networks.make_hdcq_networks(
#       env.observation_size,
#       env.action_size,
#       options=options,
#       preprocess_observations_fn=normalize,
# )
# make_policy = hdcq_networks.make_inference_fn(hdcq_network, task.hdcqn_her_hps["cost_budget"])

# aut_goal_cmdp = make_aut_goal_cmdp(task)
# hdcq_aut_network = hdcq_aut_networks.make_hdcq_networks(
#       aut_goal_cmdp.observation_size,
#       aut_goal_cmdp.cost_observation_size,
#       aut_goal_cmdp.automaton.n_states,
#       aut_goal_cmdp.action_size,
#       options=options,
#       preprocess_observations_fn=normalize,
# )
# # make_option_policy = hdcq_aut_networks.make_option_inference_fn(hdcq_aut_network, aut_goal_cmdp, task.hdcqn_her_hps["safety_minimum"])
# make_policy = hdcq_aut_networks.make_inference_fn(hdcq_aut_network, aut_goal_cmdp, task.hdcqn_her_hps["safety_minimum"])

# Making the network
# sac_network = sac_networks.make_sac_networks(
#       env.observation_size,
#       env.action_size,
#       preprocess_observations_fn=normalize,
# )
# make_policy = sac_networks.make_inference_fn(sac_network)

# # Making the network
# crl_network = crl_networks.make_crl_networks(
#     env=env,
#     observation_size=env.observation_size,
#     action_size=env.action_size,
#     # repr_dim=run.data.params["repr_dim"],
#     preprocess_observations_fn=normalize,
#     hidden_layer_sizes=[int(run.data.params["h_dim"])] * int(run.data.params["n_hidden"]),
#     use_ln=bool(run.data.params["use_ln"]),
# )
# make_policy = crl_networks.make_inference_fn(crl_network)

# %% [markdown]
# ## Or from the most recent run from the current session

# %%
# make_policy = make_inference_fn

# # %%
# inference_fn = make_policy(params, deterministic=True)
# jit_inference_fn = jax.jit(inference_fn)

jit_inference_fn = jax.jit(options[0].inference)

### Without Options

from task_aware_skill_composition.visualization.flat import get_rollout
rollout, actions = get_rollout(env, jit_inference_fn, n_steps=300, render_every=1, seed=1)

### With Options

# from task_aware_skill_composition.visualization.hierarchy import get_rollout
# rollout, opt_traj, action = get_rollout(make_shaped_reward_mdp2(task, 1), jit_inference_fn, options, n_steps=300, seed=0, render_every=1)
# rollout, opt_traj, action = get_rollout(aut_goal_mdp, jit_inference_fn, options, n_steps=300, seed=0, render_every=1)
# rollout, opt_traj, action = get_rollout(aut_goal_cmdp, jit_inference_fn, options, n_steps=300, seed=1, render_every=1)
# rollout, opt_traj, action = get_rollout(make_cmdp(task), jit_inference_fn, options, n_steps=300, seed=3, render_every=1)
# rollout, opt_traj, action = get_rollout(env, jit_inference_fn, options, n_steps=600, seed=4, render_every=1)

# %%
# mediapy.show_video(
mediapy.write_video('./option0-rollout.mp4',
                    env.render(
                        [s.pipeline_state for s in rollout],
                        camera='overview'
                    ), fps=1.0 / env.dt
)

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
