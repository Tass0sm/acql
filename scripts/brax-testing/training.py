import functools
import jax
import os

from datetime import datetime
from jax import numpy as jp
import matplotlib.pyplot as plt

import brax

import flax
from brax import envs
from brax.io import model
from brax.io import json
from brax.io import html
# from brax.training.agents.ppo import train as ppo
# from brax.training.agents.sac import train as sac

from task_aware_skill_composition.brax.agents.dscrl import train as dscrl
from task_aware_skill_composition.brax.agents.vpg import train as vpg


#@title Load Env { run: "auto" }

env_name = 'ant'  # @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
backend = 'positional'  # @param ['generalized', 'positional', 'spring']

env = envs.get_environment(env_name=env_name,
                           backend=backend)
state = jax.jit(env.reset)(rng=jax.random.PRNGKey(seed=0))

#@title Training

# We determined some reasonable hyperparameters offline and share them here.
train_fn = functools.partial(dscrl.train,
                             num_timesteps=50_000_000,
                             # num_evals=10,
                             # reward_scaling=10,
                             episode_length=1000,
                             # normalize_observations=True,
                             # action_repeat=1,
                             # unroll_length=5,
                             # num_minibatches=32,
                             # num_updates_per_batch=4,
                             # discounting=0.97,
                             # learning_rate=3e-4,
                             # entropy_cost=1e-2,
                             # num_envs=4096,
                             # batch_size=2048,
                             # seed=1
                             )



max_y = {'ant': 8000, 'halfcheetah': 8000, 'hopper': 2500, 'humanoid': 13000, 'humanoidstandup': 75_000, 'reacher': 5, 'walker2d': 5000, 'pusher': 0}[env_name]
min_y = {'reacher': -100, 'pusher': -150}.get(env_name, 0)

xdata, ydata = [], []
times = [datetime.now()]

def progress(num_steps, metrics):
  times.append(datetime.now())
  xdata.append(num_steps)
  ydata.append(metrics['eval/episode_reward'])
  # clear_output(wait=True)
  plt.xlim([0, train_fn.keywords['num_timesteps']])
  plt.ylim([min_y, max_y])
  plt.xlabel('# environment steps')
  plt.ylabel('reward per episode')
  plt.plot(xdata, ydata)
  plt.show()

make_inference_fn, params, _ = train_fn(
  environment=env,
  specification=
  progress_fn=progress
)

print(f'time to jit: {times[1] - times[0]}')
print(f'time to train: {times[-1] - times[1]}')
