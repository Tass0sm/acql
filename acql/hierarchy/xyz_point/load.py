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

from achql.brax.envs.xy_point import XYPoint
from achql.hierarchy.xy_point import options
from achql.hierarchy.option import Option, BernoulliTerminationPolicy


# def load_xyz_point_options(adapter: Optional[Callable] = None):

#     # In training environment, not in goal conditioned maze env
#     XY_POINT_OBS_SIZE = 6
#     XY_POINT_ACTION_SIZE = 3

#     options_l = []

#     for name in ["up", "right", "left", "down"]:
#         option_param_file = files(options) / (name + ".pkl")
#         params = model.load_params(option_param_file)

#         # Making the network
#         ppo_network = ppo_networks.make_ppo_networks(
#             XY_POINT_OBS_SIZE,
#             XY_POINT_ACTION_SIZE,
#             # Options were trained with "normalize observations"
#             preprocess_observations_fn=running_statistics.normalize
#         )

#         make_policy = ppo_networks.make_inference_fn(ppo_network)
#         inference_fn = make_policy(params)

#         options_l.append(Option(name, ppo_network, params, inference_fn, adapter=adapter))

#     return options_l


def load_hard_coded_xyz_point_options(
        termination_prob: float = 1.0,
        termination_policy = None,
        adapter: Optional[Callable] = None,
):

    # In training environment, not in goal conditioned maze env
    XY_POINT_OBS_SIZE = 6
    XY_POINT_ACTION_SIZE = 3

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
        return ctrl, {
            'log_prob': 0.0,
            'raw_action': ctrl
        }

    for name, ctrl in ctrl_dict.items():

        if termination_policy is None:
            termination_policy = BernoulliTerminationPolicy(termination_prob)

        print(name, ctrl)
        options_l.append(
            Option(
                name, None, None, functools.partial(hard_policy, ctrl),
                termination_policy=termination_policy,
                adapter=adapter
            )
        )

    return options_l
