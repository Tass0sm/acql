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


# tasks
from task_aware_skill_composition.brax.tasks import get_task
from task_aware_skill_composition.brax.cmdp import make_cmdp, make_aut_goal_cmdp
from task_aware_skill_composition.brax.reward_shaping import make_shaped_reward_mdp, make_shaped_reward_mdp2, make_aut_goal_mdp

mlflow.set_tracking_uri(f"file:///home/tassos/.local/share/mlflow")

# %% [markdown]
# # Defining STL

# %%
import corallab_stl.expression_jax2 as stl
# import corallab_stl.expression_jax2_extra as stl_extra

# state_var = stl.Var("state", idx=0, dim=37, y_velocity=14)
# # moderate_y_vel = stl_extra.InBox(state_var.y_velocity, jnp.array([0.4]), jnp.array([25.0]))
# # always_moderate_y_vel = stl.STLUntimedAlways(moderate_y_vel)
# # eventually_always_moderate_y_vel = stl.STLUntimedEventually(always_moderate_y_vel)

# true = stl.STLPredicate(state_var, lambda s: 999, lower_bound=0.0)

# high_y_vel = stl.STLPredicate(state_var.y_velocity, lambda s: s[0], lower_bound=1.0)
# always_high_y_vel = stl.STLUntimedAlways(high_y_vel)
# eventually_high_y_vel = stl.STLUntimedEventually(high_y_vel)
# eventually_always_high_y_vel = stl.STLUntimedEventually(always_high_y_vel)

# state_var = stl.Var("state", idx=0, dim=29, position=(0, 2))

# def in_box_pred(x_min, x_max):
#     dim = x_min.shape[0]
#     neg_eye = -jnp.eye(dim)
#     pos_eye = jnp.eye(dim)
#     A = jnp.vstack((neg_eye, pos_eye))
#     b = jnp.concatenate((-x_min, x_max))
        
#     def in_box(s):
#         x = b - jnp.matmul(A, s)
#         return jnp.min(x, axis=-1)

#     return in_box

# in_region_1 = stl.STLPredicate(
#     state_var.position,
#     in_box_pred(jnp.array([3.0, 3.0]), jnp.array([4.0, 4.0])),
#     0.1
# )

# in_region_2 = stl.STLPredicate(
#     state_var.position,
#     in_box_pred(jnp.array([-4.0, 3.0]), jnp.array([-3.0, 4.0])),
#     0.1
# )

# in_region_3 = stl.STLPredicate(
#     state_var.position,
#     in_box_pred(jnp.array([1.5, 1.5]), jnp.array([2.5, 2.5])),
#     0.0
# )

# phi = stl.STLAnd(
#     stl.STLUntimedEventually(
#         stl.STLAnd(
#             in_region_1,
#             stl.STLNext(stl.STLUntimedEventually(in_region_2)),
#         )
#     ),
#     stl.STLUntimedAlways(stl.STLNegation(in_region_3))
# )

# phi({ state_var.idx: jnp.array([[0.0, 0.0],
#                                 [3.0, 0.0],
#                                 [3.5, 3.0],
#                                 [3.5, 3.5],
#                                 [-3.5, 3.5]]) })

# in_region_1 = stl.STLPredicate(state_var.position, lambda s: s

# high_y_vel = stl.STLPredicate(state_var.y_velocity, lambda s: s[0], lower_bound=1.0)
# always_high_y_vel = stl.STLUntimedAlways(high_y_vel)
# eventually_always_high_y_vel = stl.STLUntimedEventually(high_y_vel)

# %% [markdown]
# # Defining Env/Task

# %%
mlflow.set_experiment("proj2-hierarchy-comparison")

# %%
backend = 'mjx'

# env = Car(backend=backend)
# env = Drone(backend=backend)
# env = Point(backend=backend)
# env = Doggo(backend=backend)

# env = envs.get_environment(env_name="reacher", backend=backend)
# env = AntMaze(backend=backend)

# task = get_task("simple_maze", "subgoal")
task = get_task("simple_maze_3d", "subgoal")
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

# %% [markdown]
# ## Making the Env Automaton Wrapped

# %%
from task_aware_skill_composition.brax.envs.wrappers.automaton_wrapper import AutomatonWrapper
from task_aware_skill_composition.brax.envs.wrappers.automaton_transition_rewards_wrapper import AutomatonTransitionRewardsWrapper
from task_aware_skill_composition.brax.envs.wrappers.automaton_cost_wrapper import AutomatonCostWrapper

cmdp = make_cmdp(task)
cmdp.automaton.automaton

# automaton_wrapped_env = AutomatonWrapper(
#     env,
#     spec,
#     task.obs_var,
#     augment_obs = True
# )

# automaton_wrapped_env = AutomatonCostWrapper(
#     automaton_wrapped_env
# )

# automaton_wrapped_env = AutomatonTransitionRewardsWrapper(
#     automaton_wrapped_env
# )

# alg_suffix = "_with_aut0"
# env = automaton_wrapped_env

# automaton_wrapped_env.automaton.automaton

# %%
# print(automaton_wrapped_env.observation_size)
# print(automaton_wrapped_env.action_size)
# alg_suffix = ""

# %% [markdown]
# # Plotting Utils

# %%
# xdata, ydata = [], []
# times = [datetime.now()]

def progress_fn(num_steps, metrics, df=None):
    print(f"Logging for {num_steps}")
    mlflow.log_metrics(metrics, step=num_steps)
    
    # df.loc[num_steps] = {
    #     "time": datetime.now(),
    #     "eval/episode_reward": metrics['eval/episode_reward'],
    #     "eval/episode_robustness": metrics['eval/episode_robustness'] if 'eval/episode_robustness' in metrics else None, 
    #     "training/specification_loss": metrics['training/specification_loss'] if 'training/specification_loss' in metrics else None, 
    # }

# %% [markdown]
# # Training

# %%
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


def training_run(run_id, env, seed, train_fn=ppo.train, hyperparameters={}, extras={}):
    hyperparameters = {
        **hyperparameters,
        "seed": seed,
    }

    mlflow.log_params(hyperparameters)

    train_fn = functools.partial(train_fn, **hyperparameters)
    
    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=functools.partial(progress_fn),
        seed=seed,
        **extras
    )

    with mlflow.MlflowClient()._log_artifact_helper(run_id, f'policy_params') as tmp_path:
        model.save_params(tmp_path, params)

    return make_inference_fn, params

# %% [markdown]
# ## Training with HDQN

# %%
from task_aware_skill_composition.hierarchy.xy_point.load import load_xy_point_options, load_hard_coded_xy_point_options
# from task_aware_skill_composition.hierarchy.ant.load import load_ant_options

# def adapter(x):
#     return x[..., 2:-2]

# options = load_hard_coded_xy_point_options()
# options = load_hard_coded_xy_point_options(termination_prob = 0.25)
# options = load_ant_options(adapter=adapter)
# options = load_xy_point_options() # adapter=adapter)

# %% [markdown]
# # Visualizing Policy

# %% [markdown]
# ## Loading from a previous run

# %%
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
training_run_id = "1f5dcc52a7404aabb122a265ad861ee7"
logged_model_path = f'runs:/{training_run_id}/policy_params'
real_path = mlflow.artifacts.download_artifacts(logged_model_path)
params = model.load_params(real_path)
# normalizer_params, policy_params = model.load_params(real_path)
# normalizer_params, policy_params, crl_critic_params = model.load_params(real_path)

# %%
run = mlflow.get_run(run_id=training_run_id)
if run.data.params["normalize_observations"] == "True":
    normalize = running_statistics.normalize
else:
    normalize = lambda x, y: x

# %%
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
# make_policy = hdcq_aut_networks.make_option_inference_fn(hdcq_aut_network, aut_goal_cmdp, task.hdcqn_her_hps["safety_minimum"])
# make_policy = hdcq_aut_networks.make_inference_fn(hdcq_aut_network, aut_goal_cmdp, task.hdcqn_her_hps["safety_minimum"])

# # Making the network
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

# %%
inference_fn = make_policy(params, deterministic=True)
jit_inference_fn = jax.jit(inference_fn)

# %% [markdown]
# ### Without Options

# # %%
# from task_aware_skill_composition.visualization.flat import get_rollout

# rollout, actions = get_rollout(env, jit_inference_fn, n_steps=800, render_every=1, seed=1)

# # %%
# mediapy.write_video(
#     "./safe_hopper.mp4",
#     env.render(
#         [s.pipeline_state for s in rollout],
#         camera='track'
#     ), fps=1.0 / env.dt
# )

# mediapy.show_video(
#     env.render(
#         [s.pipeline_state for s in rollout],
#         camera='track'
#     ), fps=1.0 / env.dt
# )

# %% [markdown]
# ### With Options

# %%
from task_aware_skill_composition.visualization.hierarchy import get_rollout

# rollout, opt_traj, action = get_rollout(make_shaped_reward_mdp2(task, 1), jit_inference_fn, options, n_steps=300, seed=0, render_every=1)
rollout, opt_traj, action = get_rollout(aut_goal_mdp, jit_inference_fn, options, n_steps=300, seed=0, render_every=1)
# rollout, opt_traj, action = get_rollout(aut_goal_cmdp, jit_inference_fn, options, n_steps=300, seed=0, render_every=1)
# rollout, opt_traj, action = get_rollout(make_cmdp(task), jit_inference_fn, options, n_steps=300, seed=3, render_every=1)
# rollout, opt_traj, action = get_rollout(env, jit_inference_fn, options, n_steps=600, seed=4, render_every=1)

# %%
# mediapy.show_video(
mediapy.write_video('./rollout.mp4',
                    env.render(
                        [s.pipeline_state for s in rollout],
                        camera='overview'
                    ), fps=1.0 / env.dt
                    )


# stats

obs_traj = jnp.stack([state.obs for state in rollout])
position_traj = jnp.stack([state.obs[:2] for state in rollout])
reward_traj = jnp.stack([state.reward for state in rollout])

automata_state_traj = jnp.stack([state.info["automata_state"] for state in rollout])
made_transition_traj = jnp.stack([state.info["made_transition"] for state in rollout])


# option_traj = jnp.stack(options)

# from jax.numpy.linalg import norm
# print(norm(position_traj - jnp.array([2.0, 0.0]), axis=1))
# print(delta_good_edge_satisfaction_state_traj)
# automaton_wrapped_env.automaton.one_hot_decode(observations[:, -3:])
