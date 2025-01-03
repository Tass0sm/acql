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


import corallab_stl.expression_jax2 as stl

# tasks
from task_aware_skill_composition.brax.tasks import get_task
from task_aware_skill_composition.brax.cmdp import make_cmdp, make_aut_goal_cmdp
from task_aware_skill_composition.brax.reward_shaping import make_shaped_reward_mdp, make_shaped_reward_mdp2, make_aut_goal_mdp

mlflow.set_tracking_uri(f"file:///home/tassos/.local/share/mlflow")
mlflow.set_experiment("proj2-hierarchy-comparison")

# %%
backend = 'mjx'

# task = get_task("simple_maze", "umaze_constraint")
# task = get_task("simple_maze_3d", "subgoal")
task = get_task("ant_maze", "true")
env = task.env

spec = task.lo_spec
spec_tag = type(task).__name__
env_tag = type(env).__name__

options = task.get_options()

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

# %%
print(spec_tag)
print(spec)
print(env.observation_size)
print(env.action_size)
alg_suffix = ""

# %%
from task_aware_skill_composition.brax.envs.wrappers.automaton_wrapper import AutomatonWrapper
from task_aware_skill_composition.brax.envs.wrappers.automaton_transition_rewards_wrapper import AutomatonTransitionRewardsWrapper
from task_aware_skill_composition.brax.envs.wrappers.automaton_cost_wrapper import AutomatonCostWrapper

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


# %%
training_run_id = "47c7d22f481f47a5a58ad3873b865b78"
logged_model_path = f'runs:/{training_run_id}/policy_params'
real_path = mlflow.artifacts.download_artifacts(logged_model_path)
params = model.load_params(real_path)

# %%
run = mlflow.get_run(run_id=training_run_id)
if run.data.params["normalize_observations"] == "True":
    normalize = running_statistics.normalize
    cost_normalize_fn = running_statistics.normalize
else:
    normalize_fn = lambda x, y: x
    cost_normalize_fn = lambda x, y: x


aut_goal_cmdp = make_aut_goal_cmdp(task)
hdcq_aut_network = hdcq_aut_networks.make_hdcq_networks(
      aut_goal_cmdp.observation_size,
      aut_goal_cmdp.cost_observation_size,
      aut_goal_cmdp.automaton.n_states,
      aut_goal_cmdp.action_size,
      options=options,
      preprocess_observations_fn=normalize,
      preprocess_cost_observations_fn=cost_normalize_fn,
)
make_policy = hdcq_aut_networks.make_option_inference_fn(hdcq_aut_network, aut_goal_cmdp, task.hdcqn_her_hps["safety_minimum"])

from task_aware_skill_composition.brax.agents.hdcqn_automaton_her import networks as hdcq_networks
from task_aware_skill_composition.visualization.critic import make_plots_for_hdcqn, make_plots_for_hdcqn_aut

plot1, plot2 = make_plots_for_hdcqn_aut(
        aut_goal_cmdp,
        make_policy,
        hdcq_aut_network,
        params,
        "test",
        # tmp_state_fn=lambda x: x.replace(obs=x.obs.at[4:6].set(jnp.array([4., 12.]))),
        save_and_close=True,
)
