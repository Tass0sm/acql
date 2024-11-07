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

# tasks
from task_aware_skill_composition.brax.tasks import get_task


backend = 'mjx'

# env = Car(backend=backend)
# env = Drone(backend=backend)
# env = Point(backend=backend)
# env = Doggo(backend=backend)
# env = envs.get_environment(env_name="hopper", backend=backend)
# env = AntMaze(backend=backend)
# env_tag = type(env).__name__

task = get_task("point", "turn_seq")
env = task.env

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

# reset the environment
rng = jax.random.PRNGKey(0)
state = jit_reset(rng)

# from brax.training.agents.ppo import networks as ppo_networks
# from brax.training.acme import running_statistics

# training_run_id = "9a1bae66199d4e4aba59f14b333f2e5a"
# logged_model_path = f'runs:/{training_run_id}/policy_params'
# real_path = mlflow.artifacts.download_artifacts(logged_model_path)
# params = model.load_params(real_path)

# run = mlflow.get_run(run_id=training_run_id)
# if run.data.params["normalize_observations"] == "True":
#     normalize = running_statistics.normalize
# else:
#     normalize = lambda x, y: x

# # Making the network
# ppo_network = ppo_networks.make_ppo_networks(
#       state.obs.shape[0],
#       env.action_size,
#       preprocess_observations_fn=normalize
# )
# make_policy = ppo_networks.make_inference_fn(ppo_network)

# inference_fn = make_policy(params)
# jit_inference_fn = jax.jit(inference_fn)

# # grab a trajectory acting according to the policy function
# rollout = [state]
# n_steps = 50
# render_every = 1

# for i in range(n_steps):
#     act_rng, rng = jax.random.split(rng)
#     ctrl, _ = jit_inference_fn(state.obs, act_rng)
#     state = jit_step(state, ctrl)

#     if state.done:
#         break

#     # print(type(state.pipeline_state))

#     rollout.append(state)

# mediapy.show_video(
#     env.render(
#         [s.pipeline_state for s in rollout],
#         camera='overview'
#     ), fps=1.0 / env.dt
# )
