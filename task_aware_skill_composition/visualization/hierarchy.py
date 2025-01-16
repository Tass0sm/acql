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
from task_aware_skill_composition.hierarchy.option import Option, FixedLengthTerminationPolicy


def get_rollout(env, hierarchical_policy, options, n_steps=200, render_every=1, seed=0):
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    if hasattr(env, "automaton") and env.augment_obs:
        def augment_adapter(option):
            old_adapter = option.obs_adapter
            new_adapter = lambda x: old_adapter(env.original_obs(x))
            option.obs_adapter = new_adapter

        for o in options:
            augment_adapter(o)

    # reset the environment
    rng = jax.random.PRNGKey(seed)
    state = jit_reset(rng)

    initial_option_state = OptionState(option=jnp.ones((1,), dtype=jnp.int32) * 0,
                                       option_beta=jnp.ones((1,), dtype=jnp.int32) * 1)
    option_state = initial_option_state

    rollout = [state]
    rollout_options = []
    actions = []

    # TODO: Figure out centralized way to handle the option state / termination...
    # Assuming that all options have same fixed length
    if isinstance(options[0].termination_policy, FixedLengthTerminationPolicy):
        length = options[0].termination_policy.t

        def get_beta(i, s_t, o_t, beta_key):
            return jnp.where(i % length, 1, 0)
    else:
        def get_beta(i, s_t, o_t, beta_key):
            return jax.lax.switch(o_t, [o.termination for o in options], s_t, beta_key)


    for i in range(n_steps):
        act_rng, rng = jax.random.split(rng)
        obs = jnp.expand_dims(state.obs, 0)

        ctrl, extras = hierarchical_policy(obs, option_state, act_rng)
        ctrl = ctrl.squeeze(0)

        state = jit_step(state, ctrl)

        if state.done:
            break

        option = extras["option"]

        # calculate b_t+1 from s_t+1, o_t
        beta_keys = jax.random.split(rng, option.shape[0])
        termination = jax.vmap(get_beta)(jnp.expand_dims(i, 0), jnp.expand_dims(state.obs, 0), option, beta_keys)

        option_state = OptionState(option, termination)

        rollout_options.append(option)
        actions.append(ctrl)
        rollout.append(state)

    return rollout, jnp.concatenate(rollout_options), jnp.stack(actions)
