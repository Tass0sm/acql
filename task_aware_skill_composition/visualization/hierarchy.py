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

from task_aware_skill_composition.hierarchy.state import OptionState


def get_rollout(env, hierarchical_policy, n_steps=200, render_every=1):
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # reset the environment
    rng = jax.random.PRNGKey(1)
    state = jit_reset(rng)

    initial_option_state = OptionState(option=jnp.ones((1,), dtype=jnp.int32) * 0,
                                       option_beta=jnp.ones((1,), dtype=jnp.int32) * 1)
    option_state = initial_option_state

    rollout = [state]
    options = []
    actions = []

    for i in range(n_steps):
        act_rng, rng = jax.random.split(rng)
        obs = jnp.expand_dims(state.obs, 0)

        ctrl, extras = hierarchical_policy(obs, option_state, act_rng)
        ctrl = ctrl.squeeze(0)

        state = jit_step(state, ctrl)

        if state.done:
            break

        option = extras["option"]
        # termination = jnp.zeros_like(option_state.option_beta)
        # termination = jax.random.bernoulli(act_rng, p=jnp.float32(0.2), shape=option_state.option_beta.shape).astype(jnp.int32)
        termination = jnp.ones_like(option_state.option_beta)
        option_state = OptionState(option, termination)

        options.append(option)
        actions.append(ctrl)
        rollout.append(state)

    return rollout, options, actions
