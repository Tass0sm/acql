import functools
import jax
import os
import pandas as pd
import mlflow
import pickle

from datetime import datetime
from jax import numpy as jnp
import matplotlib.pyplot as plt

import mediapy

import brax

import flax
from brax import envs
from brax.io import model
from brax.io import json
from brax.io import html

from achql.brax.agents.hdcqn_automaton_her import networks as hdcq_aut_networks
from brax.training.acme import running_statistics

# tasks
from achql.brax.tasks import get_task
from achql.brax.utils import make_cmdp, make_aut_goal_cmdp, make_shaped_reward_mdp, make_shaped_reward_mdp2, make_aut_goal_mdp

from achql.visualization import hierarchy
from achql.visualization import flat

import achql.stl.expression_jax2 as stl

mlflow.set_tracking_uri(f"file:///home/tassos/.local/share/mlflow")
mlflow.set_experiment("proj2-ur5e")


if __name__ == "__main__":
    task = get_task("ur5e", "scalability_test")
    
    env = task.env
    
    spec = task.lo_spec
    spec_tag = type(task).__name__
    env_tag = type(env).__name__
    
    options = task.get_options()
    
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    
    print(spec_tag)
    print(spec)
    print(env.observation_size)
    print(env.action_size)
    
    training_run_id = "449b832fb5df4b7d9740c22ed27ed671"
    logged_model_path = f'runs:/{training_run_id}/policy_params'
    real_path = mlflow.artifacts.download_artifacts(logged_model_path)
    params = model.load_params(real_path)
    
    run = mlflow.get_run(run_id=training_run_id)
    if run.data.params["normalize_observations"] == "True":
        normalize_fn = running_statistics.normalize
        cost_normalize_fn = running_statistics.normalize
    else:
        normalize_fn = lambda x, y: x
        cost_normalize_fn = lambda x, y: x
    
    aut_goal_cmdp = make_aut_goal_cmdp(task, margin=0.0, randomize_goals=False)
    hdcq_aut_network = hdcq_aut_networks.make_hdcq_networks(
          aut_goal_cmdp.observation_size,
          aut_goal_cmdp.cost_observation_size,
          aut_goal_cmdp.automaton.n_states,
          aut_goal_cmdp.action_size,
          options=options,
          preprocess_observations_fn=normalize_fn,
    )

    make_policy = hdcq_aut_networks.make_inference_fn(hdcq_aut_network, aut_goal_cmdp, task.hdcqn_her_hps["safety_minimum"])
    
    inference_fn = make_policy(params, deterministic=True)
    jit_inference_fn = jax.jit(inference_fn)
    
    rollout, opt_traj, action = hierarchy.get_rollout(aut_goal_cmdp, jit_inference_fn, options, n_steps=100, seed=3, render_every=1)

    mediapy.write_video(f'./hierarchical-rollout.mp4',
        env.render(
            [s.pipeline_state for s in rollout],
            camera='overview'
        ), fps=1.0 / env.dt
    )

    obs_traj = jnp.stack([state.obs for state in rollout])
    rho = spec({ task.obs_var.idx: obs_traj })

    print(f"rho = {rho}")
