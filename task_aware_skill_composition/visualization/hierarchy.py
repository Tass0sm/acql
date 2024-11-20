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


def get_rollout(env, hierarchical_policy, options, n_steps=200, render_every=1, seed=0):
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # reset the environment
    rng = jax.random.PRNGKey(seed)
    state = jit_reset(rng)

    initial_option_state = OptionState(option=jnp.ones((1,), dtype=jnp.int32) * 0,
                                       option_beta=jnp.ones((1,), dtype=jnp.int32) * 1)
    option_state = initial_option_state

    rollout = [state]
    rollout_options = []
    actions = []

    for i in range(n_steps):
        act_rng, rng = jax.random.split(rng)
        obs = jnp.expand_dims(state.obs, 0)

        ctrl, extras = hierarchical_policy(obs, option_state, act_rng)
        ctrl = ctrl.squeeze(0)

        state = jit_step(state, ctrl)

        if state.done:
            break

        # TODO: Figure out centralized way to handle the option state / termination...
        option = extras["option"]

        def get_beta(s_t, o_t, beta_key):
            return jax.lax.switch(o_t, [o.termination for o in options], s_t, beta_key)

        # calculate b_t+1 from s_t+1, o_t
        beta_keys = jax.random.split(rng, option.shape[0])
        termination = jax.vmap(get_beta)(jnp.expand_dims(state.obs, 0), option, beta_keys)

        option_state = OptionState(option, termination)

        rollout_options.append(option)
        actions.append(ctrl)
        rollout.append(state)

    return rollout, jnp.concatenate(rollout_options), jnp.stack(actions)
