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


def get_rollout(env, hierarchical_policy, logical_options, n_steps=200, render_every=1, seed=0):
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # if hasattr(env, "automaton") and env.augment_obs:
    #     extra_adapter = env.original_obs
    # else:
    #     extra_adapter = lambda x: x

    # reset the environment
    rng = jax.random.PRNGKey(seed)
    state = jit_reset(rng)

    initial_option_state = OptionState(option=jnp.ones((1,), dtype=jnp.int32) * 0,
                                       option_beta=jnp.ones((1,), dtype=jnp.int32) * 1)
    option_state = initial_option_state

    rollout = [state]
    rollout_logical_options = []
    actions = []

    def get_beta(i, s_t, o_t, beta_key):
        return jax.lax.switch(o_t, [o.termination for o in logical_options], s_t, beta_key)

    for i in range(n_steps):
        act_rng, rng = jax.random.split(rng)
        obs = jnp.expand_dims(state.obs, 0)

        ctrl, extras = hierarchical_policy(obs, option_state, act_rng)
        ctrl = ctrl.squeeze(0)

        state = jit_step(state, ctrl)

        if state.done:
            break

        logical_option = extras["logical_option"]

        # calculate b_t+1 from s_t+1, o_t
        beta_keys = jax.random.split(rng, logical_option.shape[0])
        termination = jax.vmap(get_beta)(jnp.expand_dims(i, 0), jnp.expand_dims(state.obs, 0), logical_option, beta_keys)

        option_state = OptionState(logical_option, termination)

        rollout_logical_options.append(logical_option)
        actions.append(ctrl)
        rollout.append(state)

    return rollout, jnp.concatenate(rollout_logical_options), jnp.stack(actions)


def get_rollout2(env, hierarchical_policy, logical_options, options, n_steps=200, render_every=1, seed=0):
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # if hasattr(env, "automaton") and env.augment_obs:
    #     extra_adapter = env.original_obs
    # else:
    #     extra_adapter = lambda x: x

    # reset the environment
    rng = jax.random.PRNGKey(seed)
    state = jit_reset(rng)

    initial_option_state = OptionState(option=jnp.ones((1,), dtype=jnp.int32) * 0,
                                       option_beta=jnp.ones((1,), dtype=jnp.int32) * 1)
    option_state = initial_option_state

    rollout = [state]
    rollout_options = []
    actions = []

    def get_beta(i, s_t, o_t, beta_key):
        return jax.lax.switch(o_t, [o.termination for o in logical_options], s_t, beta_key)

    def get_mid_beta(j, s_t, o_t, beta_rng):
        if isinstance(options[0].termination_policy, FixedLengthTerminationPolicy):
            return j == options[0].termination_policy.t
        else:
            return jax.lax.switch(o_t, [o.termination for o in options], s_t, beta_rng)

    # def inner_while_cond(x):
    #     j, s_t, o_t, rng = x
    #     beta_t = jax.lax.switch(o_t, [o.termination for o in self.options], s_t.obs, key)
    #     return jnp.logical_and(beta_t != 1, iters < self.max_option_iters)

    # def inner_while_body(x):
    #     j, s_t, o_t, rng = x

    #     low_rng, low_beta_rng, rng = jax.random.split(rng, 3)
    #     a_t, _ = jax.lax.switch(o_t, [o.inference for o in options], obs, low_rng)
    #     a_t = a_t.squeeze(0)

    #     state = jit_step(state, a_t)
    #     obs = jnp.expand_dims(state.obs, 0)

    #     if state.done:
    #         break

    #     rollout_options.append(low_option_t)
    #     actions.append(a_t)
    #     rollout.append(state)

    #     # logical_option = extras["logical_option"]

    #     # calculate b_t+1 from s_t+1, o_t
    #     return j+1, state, rng

    #     beta_t = jax.lax.switch(o_t, [o.termination for o in self.options], s_t.obs, key)
    #     return jnp.logical_and(beta_t != 1, iters < self.max_option_iters)


    t = 0
    while t < n_steps:
        hi_rng, rng = jax.random.split(rng)
        obs = jnp.expand_dims(state.obs, 0)

        low_option_t, extras = hierarchical_policy(obs, option_state, hi_rng)
        low_option_t = low_option_t.squeeze(0)

        # mid_option_t, _ = jax.lax.switch(logical_option, [o.inference for o in logical_options], obs, mid_rng)
        mid_beta = 0
        j = 0
        while mid_beta != 1:
            low_rng, low_beta_rng, rng = jax.random.split(rng, 3)
            a_t, _ = jax.lax.switch(low_option_t, [o.inference for o in options], obs, low_rng)
            a_t = a_t.squeeze(0)

            state = jit_step(state, a_t)
            obs = jnp.expand_dims(state.obs, 0)

            if state.done:
                break
    
            rollout_options.append(low_option_t)
            actions.append(a_t)
            rollout.append(state)

            # logical_option = extras["logical_option"]
    
            # calculate b_t+1 from s_t+1, o_t
            j += 1

            mid_beta = get_mid_beta(j, obs, low_option_t, low_beta_rng)

        logical_option = extras["logical_option"]

        # calculate b_t+1 from s_t+1, o_t
        beta_keys = jax.random.split(rng, logical_option.shape[0])
        termination = jax.vmap(get_beta)(jnp.expand_dims(t, 0), jnp.expand_dims(state.obs, 0), logical_option, beta_keys)

        option_state = OptionState(logical_option, termination)

        t += j

        # breakpoint()

    return rollout, jnp.stack(rollout_options), jnp.stack(actions)


def get_rollout3(env, hierarchical_policy, logical_options, options, n_steps=200, render_every=1, seed=0):
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # if hasattr(env, "automaton") and env.augment_obs:
    #     extra_adapter = env.original_obs
    # else:
    #     extra_adapter = lambda x: x

    # reset the environment
    rng = jax.random.PRNGKey(seed)
    state = jit_reset(rng)

    initial_option_state = OptionState(option=jnp.ones((1,), dtype=jnp.int32) * 0,
                                       option_beta=jnp.ones((1,), dtype=jnp.int32) * 1)
    option_state = initial_option_state

    rollout = [state]
    rollout_options = []
    actions = []

    def get_beta(i, s_t, o_t, beta_key):
        return jax.lax.switch(o_t, [o.termination for o in logical_options], s_t, beta_key)

    if hasattr(options[0].termination_policy, "t"):
        inner_scan_length = options[0].termination_policy.t
    else:
        inner_scan_length = 1

    def inner_scan_body(x, unused):
        j, s_t, o_t, rng = x

        low_rng, low_beta_rng, rng = jax.random.split(rng, 3)
        a_t, _ = jax.lax.switch(o_t, [o.inference for o in options], s_t.obs, low_rng)
        # a_t = a_t.squeeze(0)

        s_t = jit_step(s_t, a_t)

        return (j+1, s_t, o_t, rng), s_t

    t = 0
    while t < n_steps:
        hi_rng, rng = jax.random.split(rng)
        obs = jnp.expand_dims(state.obs, 0)

        low_option_t, extras = hierarchical_policy(obs, option_state, hi_rng)
        low_option_t = low_option_t.squeeze(0)

        (j, state, _, rng), s_ts = jax.lax.scan(inner_scan_body, (0, state, low_option_t, rng), length=inner_scan_length)

        # rollout_options.append(low_option_t)
        # actions.append(a_t)
        rollout.extend([ jax.tree.map(lambda x: x[i], s_ts) for i in range(inner_scan_length) ])

        logical_option = extras["logical_option"]

        # calculate b_t+1 from s_t+1, o_t
        beta_keys = jax.random.split(rng, logical_option.shape[0])
        termination = jax.vmap(get_beta)(jnp.expand_dims(t, 0), jnp.expand_dims(state.obs, 0), logical_option, beta_keys)

        option_state = OptionState(logical_option, termination)

        t += j

        # breakpoint()

    # return rollout, jnp.stack(rollout_options), jnp.stack(actions)
    return rollout, None, None
