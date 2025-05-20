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

from acql.brax.envs.xy_point import XYPoint
from acql.hierarchy.xy_point import options
from acql.hierarchy.option import Option, FixedLengthTerminationPolicy


def load_hard_coded_x_point_options(k: int = 1):

    options_l = []

    ctrl_dict = {
        "left": jnp.array([-1.0]),
        "right": jnp.array([1.0]),
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
        print(name, ctrl)
        options_l.append(
            Option(
                name, None, None, functools.partial(hard_policy, ctrl),
                termination_policy=FixedLengthTerminationPolicy(k)
            )
        )

    return options_l
