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


def get_rollout(env, flat_policy, n_steps=200, render_every=1, seed=0):
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # reset the environment
    rng = jax.random.PRNGKey(seed)
    state = jit_reset(rng)

    rollout = [state]
    actions = []

    for i in range(n_steps):
        act_rng, rng = jax.random.split(rng)
        obs = jnp.expand_dims(state.obs, 0)

        ctrl, _ = flat_policy(obs, act_rng)
        ctrl = ctrl.squeeze(0)

        state = jit_step(state, ctrl)

        if state.done:
            break

        actions.append(ctrl)
        rollout.append(state)

    return rollout, jnp.stack(actions)
