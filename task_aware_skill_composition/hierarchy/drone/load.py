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

from task_aware_skill_composition.hierarchy.option import Option, BernoulliTerminationPolicy


def load_hard_coded_drone_options(
        termination_prob: float = 1.0,
        adapter: Optional[Callable] = None,
):

    # In training environment, not in goal conditioned maze env
    DRONE_OBS_SIZE = None
    DRONE_ACTION_SIZE = 4

    options_l = []

    ctrl_dict = {
        "up":            jnp.array([0.50000,  0.0,  0.0,  0.0]),
        "hover":         jnp.array([0.26487,  0.0,  0.0,  0.0]),
        "x-moment-plus": jnp.array([0.26487,  0.1,  0.0,  0.0]),
        "x-moment-neg":  jnp.array([0.26487, -0.1,  0.0,  0.0]),
        "y-moment-plus": jnp.array([0.26487,  0.0,  0.1,  0.0]),
        "y-moment-neg":  jnp.array([0.26487,  0.0, -0.1,  0.0]),
        "z-moment-plus": jnp.array([0.26487,  0.0,  0.0,  0.1]),
        "z-moment-neg":  jnp.array([0.26487,  0.0,  0.0, -0.1]),
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
                termination_policy=BernoulliTerminationPolicy(termination_prob),
                adapter=adapter
            )
        )

    return options_l
