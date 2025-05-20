import functools
import jax
import os
from typing import Optional, Callable, Tuple
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
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.acme import running_statistics
from brax.training import types
from brax.training.types import PRNGKey

from achql.hierarchy.ur5e import options
from achql.hierarchy.option import Option, BernoulliTerminationPolicy


def load_ur5e_options(
        termination_prob: float = 0.3,
        termination_policy = None,
        adapter: Optional[Callable] = None
):

    # In training environment, not in goal conditioned maze env
    UR5E_TRANSLATE_OBS_SIZE = 12
    UR5E_TRANSLATE_ACTION_SIZE = 5

    options_l = []

    for name in ["up", # "forward", "right", "left", "backward", "down"
                 ]:
        option_param_file = files(options) / (name + ".pkl")
        params = model.load_params(option_param_file)

        # Making the network
        ppo_network = ppo_networks.make_ppo_networks(
            UR5E_TRANSLATE_OBS_SIZE,
            UR5E_TRANSLATE_ACTION_SIZE,
            # Options were trained with "normalize observations"
            preprocess_observations_fn=running_statistics.normalize
        )

        make_policy = ppo_networks.make_inference_fn(ppo_network)
        inference_fn = make_policy(params)

        if termination_policy is None:
            termination_policy = BernoulliTerminationPolicy(termination_prob)

        options_l.append(
            Option(name, ppo_network, params, inference_fn,
                   termination_policy=termination_policy,
                   adapter=adapter)
        )


    return options_l


def load_ur5e_eef_options(
        termination_prob: float = 1.0,
        termination_policy = None,
        adapter: Optional[Callable] = None,
        ctrl_scale: float = 1.0,
):

    options_l = []

    ctrl_dict = {
        "up": jnp.array([0.0, 0.0, 1.0]),
        "forward": jnp.array([0.0, 1.0, 0.0]),
        "right": jnp.array([1.0, 0.0, 0.0]),
        "left": jnp.array([-1.0, 0.0, 0.0]),
        "backward": jnp.array([0.0, -1.0, 0.0]),
        "down": jnp.array([0.0, 0.0, -1.0]),
    }

    def hard_policy(
            ctrl,
            observations: types.Observation,
            key_sample: PRNGKey
    ) -> Tuple[types.Action, types.Extra]:
        return ctrl * ctrl_scale, {
            'log_prob': 0.0,
            'raw_action': ctrl
        }

    if termination_policy is None:
        termination_policy = BernoulliTerminationPolicy(termination_prob)

    for name, ctrl in ctrl_dict.items():
        print(name, ctrl)
        options_l.append(
            Option(
                name, None, None, functools.partial(hard_policy, ctrl),
                termination_policy=termination_policy,
                adapter=adapter
            )
        )

    return options_l


def load_ur5e_eef_binpick_options(
        termination_prob: float = 1.0,
        termination_policy = None,
        adapter: Optional[Callable] = None,
        ctrl_scale: float = 0.5,
):

    options_l = []

    ctrl_dict = {
        "up": jnp.array([0.0, 0.0, 1.0, 0.0]),
        "forward": jnp.array([0.0, 1.0, 0.0, 0.0]),
        "right": jnp.array([1.0, 0.0, 0.0, 0.0]),
        "left": jnp.array([-1.0, 0.0, 0.0, 0.0]),
        "backward": jnp.array([0.0, -1.0, 0.0, 0.0]),
        "down": jnp.array([0.0, 0.0, -1.0, 0.0]),
        "grasp": jnp.array([0.0, 0.0, 0.0, 1.0]),
        "release": jnp.array([0.0, 0.0, 0.0, -1.0]),
    }

    def hard_policy(
            ctrl,
            observations: types.Observation,
            key_sample: PRNGKey
    ) -> Tuple[types.Action, types.Extra]:
        return ctrl * ctrl_scale, {
            'log_prob': 0.0,
            'raw_action': ctrl
        }

    if termination_policy is None:
        termination_policy = BernoulliTerminationPolicy(termination_prob)

    for name, ctrl in ctrl_dict.items():
        print(name, ctrl)
        options_l.append(
            Option(
                name, None, None, functools.partial(hard_policy, ctrl),
                termination_policy=termination_policy,
                adapter=adapter
            )
        )

    return options_l
