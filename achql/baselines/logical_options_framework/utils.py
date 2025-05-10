import functools
import time
from typing import Any, Callable, Optional, Tuple, Union, Sequence

import mlflow

from absl import logging
from brax import base
from brax import envs
from brax.envs import wrappers
from brax.io import model
from brax.training import acting
from brax.training import gradients
from brax.training import pmap
from brax.training import replay_buffers
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.types import Params
from brax.training.types import PRNGKey
import flax
import jax
import jax.numpy as jnp
import optax
import numpy as np

from achql.hierarchy.training.evaluator import HierarchicalEvaluatorWithSpecification
from achql.hierarchy.option import Option
from achql.hierarchy.envs.options_wrapper import OptionsWrapper
from achql.hierarchy.state import OptionState

from achql.brax.agents.hdqn import networks as hdq_networks
from achql.brax.envs.wrappers.automaton_wrapper import JaxAutomaton, AutomatonWrapper
from achql.baselines.logical_options_framework.lof_wrapper import LOFWrapper

from achql.stl import get_spot_formula_and_aps, make_just_liveness_automaton, get_outgoing_conditions
import achql.stl as stl

from achql.baselines.logical_options_framework.train import partition, load_logical_options


def get_logical_option_run_ids(task_name, seed):

    # Order matters and corresponds to the AP numbers for each task.
    lo_run_ids_dict = {
        ("SimpleMazeObligationConstraint1", 0): ["fd1f5a509fbf4d91a5d28a46d8d2ccd2"],
        ("SimpleMazeObligationConstraint1", 1): ["9fabb95db7aa46d3962d5dadab1cbec1"],
        ("SimpleMazeObligationConstraint1", 2): ["a997e94974aa4f8289f049f794fa1676"],
        ("SimpleMazeBranching1", 0): ["da500b0db98443fc917c7787aed938c1", "a7f6a0f9566043bdae312aaa8f1f6cce"],
        ("SimpleMazeBranching1", 1): ["2f241710354c4c0fb309f75c659f2161", "9f4df64a1ec340b48730bc2a389351cd"],
        ("SimpleMazeBranching1", 2): ["b00099d25a734f5dbf45afa1e2423cbf", "544890cbd9f34024a1161bd51f1539d3"],
        ("SimpleMazeTwoSubgoals", 0): ["54121566101c41d1b368ebebb10eaf5e", "c6f81218e0994bdea570221e5e37896b"],
        ("SimpleMazeTwoSubgoals", 1): ["d9cfefe273744f12aa371265d1e2944d", "c570bf3774694dc2a7e9627aec81231a"],
        ("SimpleMazeTwoSubgoals", 2): ["b37c26d8bc2645b9a710e18f735d8b25", "36055cb35c3b4966992a627f7aeb5dda"],

        ("SimpleMazeUntil2", 0): ["50a47e077f8d435d8aa90404d68f4bed", "2161a861f62d4439b4f439466617534c"],

        ("SimpleMaze3DTwoSubgoals", 0): ["def0481071e1485e942a37eb6ae3d123", "496bc0f8e01b47c39602b77837946d0b"],
        ("SimpleMaze3DTwoSubgoals", 1): ["c15732b4d5444f91b7c97d38b70f95a8", "8168d6f1a1af404594bd4df0f843529d"],
        ("SimpleMaze3DTwoSubgoals", 2): ["7a2825234a2d4b2c8ecf10db3734c449", "1998df693fe84e329891edf347c2009e"],
        ("SimpleMaze3DBranching1", 0): ["a25fa8d1e2354578987466711d0bd359", "16373bf412e347e6b2598c6951773a9a"],
        ("SimpleMaze3DBranching1", 1): ["7502e6ff1b064bf49024e751fa3a1bdc", "e2f91f424e8245f8a9626ca5f8922c83"],
        ("SimpleMaze3DBranching1", 2): ["a57e925d9ac04674bd8fb6daa97ce5c8", "b2bc156475e44e6cb4ed184a6ba66dea"],
        ("SimpleMaze3DObligationConstraint2", 0): ["c087bdc1373440cf9965774ad4aee2c4"],
        ("SimpleMaze3DObligationConstraint2", 1): ["db42593655be47fbae570418bda2ebe4"],
        ("SimpleMaze3DObligationConstraint2", 2): ["dc9cb40588dd4596b02d59a533745ad7"],
        # ("SimpleMaze3DObligationConstraint3", 0): ["7699e861c6544772882c7c2ec806b577"],

        ("AntMazeTwoSubgoals", 0): ["731ee5b9391249cd98add1b8049cab4c", "90739cab34244620b6fc70bbabc504de"],
        ("AntMazeTwoSubgoals", 1): ["2d5ee614fde54ae2adb69bb5e6a53c83", "7db38d2d1afe49b880e45f35e84b1add"],
        ("AntMazeTwoSubgoals", 2): ["85433bb2436041bd930781fc595a5d42", "7e655e0ef8fa41a1b81aad65d223b1e2"],
        ("AntMazeBranching1", 0): ["99a314ccad384a8fbb8729329f3fa4ff", "4bc3c3f314304061adb4b16ee8fb2e7b"],
        ("AntMazeBranching1", 1): ["c30f3f0db7334a8d91a4e1274922a23e", "ee35b1c020114df0a9a2180f5e7961ff"],
        ("AntMazeBranching1", 2): ["90bc2965466d48d7998be863525cdc08", "1136c41b700945c589f57bef0190924f"],
        ("AntMazeObligationConstraint3", 0): ["ceadd5cc14da408c86bf0e2f4fa2c179"],
        ("AntMazeObligationConstraint3", 1): ["3046f9b8b8154a55b80a27dda8383836"],
        ("AntMazeObligationConstraint3", 2): ["e4184c8092a84fe697519675b3c1539b"],
    }

    return lo_run_ids_dict.get((task_name, seed), None)


def get_logical_options(
    task,
    logical_option_run_ids,
    options,
):

    environment = task.env
    specification = task.lo_spec
    state_var = task.obs_var
    
    full_automaton = JaxAutomaton(specification, state_var)
    liveness_automaton = make_just_liveness_automaton(full_automaton.automaton)
    plain_out_conditions = get_outgoing_conditions(liveness_automaton)

    no_goal_aps, goal_aps = partition(lambda _, v: "goal" in v.info, full_automaton.aps)

    max_param_dim = max([ap.info["goal"].size for ap in goal_aps.values()])
    automaton_ap_params = jnp.zeros((full_automaton.n_aps, max_param_dim))
    for k, ap in full_automaton.aps.items():
        if k in goal_aps:
            dim = ap.info["goal"].size
            automaton_ap_params = automaton_ap_params.at[k, :dim].set(ap.info["goal"])
        elif ap.default_params is not None:
            dim = ap.default_params.size
            automaton_ap_params = automaton_ap_params.at[k, :dim].set(ap.default_params)

    # Train or load options

    logical_options = load_logical_options(
        logical_option_run_ids,
        environment,
        no_goal_aps,
        goal_aps,
        full_automaton,
        automaton_ap_params,
        state_var,
        options=options,
    )

    # S
    start_states = (
        environment.possible_starts.tolist() +
        [ap.info["goal"].tolist() for ap in goal_aps.values()]
    )
    start_states_array = jnp.array(start_states)

    return logical_options, start_states_array
