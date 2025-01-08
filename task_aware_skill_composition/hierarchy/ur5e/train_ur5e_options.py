import functools
import jax
import os
from typing import Optional
from importlib.resources import files

from datetime import datetime
from jax import numpy as jnp
import matplotlib.pyplot as plt

import brax

import flax
from brax import envs
from brax.io import model
from brax.io import json
from brax.io import html
from brax.training.agents.ppo import train as ppo

from task_aware_skill_composition.brax.envs.manipulation.ur5e_translate import UR5eTranslate

from . import options as options_module


ppo_hps = {
    "num_timesteps": 6_000_000,
    "num_evals": 4,
    "reward_scaling": 10,
    "episode_length": 1000,
    "normalize_observations": True,
    "action_repeat": 1,
    "unroll_length": 5,
    "num_minibatches": 32,
    "num_updates_per_batch": 4,
    "discounting": 0.97,
    "learning_rate": 3e-4,
    "entropy_cost": 1e-2,
    "num_envs": 4096,
    "batch_size": 2048,
}


def progress_fn(num_steps, metrics):
    print(f"Finished {num_steps}")
    print(metrics)

def train_ur5e_options(seed=0):

    target_velocity_vecs = {
        "up": jnp.array([0.0, 0.0, 1.0]),
        "forward": jnp.array([0.0, 1.0, 0.0]),
        "right": jnp.array([1.0, 0.0, 0.0]),
        "left": jnp.array([-1.0, 0.0, 0.0]),
        "backward": jnp.array([0.0, -1.0, 0.0]),
        "down": jnp.array([0.0, 0.0, -1.0]),
    }

    train_fn = functools.partial(ppo.train, seed=seed, **ppo_hps)

    for name, target_vec in target_velocity_vecs.items():
        option_mdp = UR5eTranslate(
            target_velocity_vec=target_vec,
            backend="mjx"
        )

        jit_reset = jax.jit(option_mdp.reset)
        jit_step = jax.jit(option_mdp.step)

        rng = jax.random.PRNGKey(seed)
        state = jit_reset(rng)

        make_inference_fn, params, _ = train_fn(
            environment=option_mdp,
            progress_fn=progress_fn,
        )
        option_param_file = files(options_module) / (name + ".pkl")
        model.save_params(option_param_file, params)
        print(f"Saved to {option_param_file}")


if __name__ == "__main__":
    train_ur5e_options()
