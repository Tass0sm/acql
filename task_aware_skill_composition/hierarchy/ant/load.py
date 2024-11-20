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

from task_aware_skill_composition.brax.envs.ant import Ant
from task_aware_skill_composition.hierarchy.ant import options
from task_aware_skill_composition.hierarchy.option import Option, BernoulliTerminationPolicy


def load_ant_options(
        termination_prob: float = 0.2,
        adapter: Optional[Callable] = None
):

    # In training environment, not in goal conditioned maze env
    ANT_OBS_SIZE = 27
    ANT_ACTION_SIZE = 8

    options_l = []

    for name in ["up", "right", "left", "down"]:
        option_param_file = files(options) / (name + ".pkl")
        params = model.load_params(option_param_file)

        # Making the network
        ppo_network = ppo_networks.make_ppo_networks(
            ANT_OBS_SIZE,
            ANT_ACTION_SIZE,
            # Options were trained with "normalize observations"
            preprocess_observations_fn=running_statistics.normalize
        )

        make_policy = ppo_networks.make_inference_fn(ppo_network)
        inference_fn = make_policy(params)

        options_l.append(
            Option(name, ppo_network, params, inference_fn,
                   termination_policy=BernoulliTerminationPolicy(termination_prob),
                   adapter=adapter)
        )


    return options_l
