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
        # ("SimpleMazeObligationConstraint1", 0): ["fd1f5a509fbf4d91a5d28a46d8d2ccd2"],
        # ("SimpleMazeObligationConstraint1", 1): ["9fabb95db7aa46d3962d5dadab1cbec1"],
        # ("SimpleMazeObligationConstraint1", 2): ["a997e94974aa4f8289f049f794fa1676"],
        # ("SimpleMazeBranching1", 0): ["da500b0db98443fc917c7787aed938c1", "a7f6a0f9566043bdae312aaa8f1f6cce"],
        # ("SimpleMazeBranching1", 1): ["2f241710354c4c0fb309f75c659f2161", "9f4df64a1ec340b48730bc2a389351cd"],
        # ("SimpleMazeBranching1", 2): ["b00099d25a734f5dbf45afa1e2423cbf", "544890cbd9f34024a1161bd51f1539d3"],
        # ("SimpleMazeTwoSubgoals", 0): ["54121566101c41d1b368ebebb10eaf5e", "c6f81218e0994bdea570221e5e37896b"],
        # ("SimpleMazeTwoSubgoals", 1): ["d9cfefe273744f12aa371265d1e2944d", "c570bf3774694dc2a7e9627aec81231a"],
        # ("SimpleMazeTwoSubgoals", 2): ["b37c26d8bc2645b9a710e18f735d8b25", "36055cb35c3b4966992a627f7aeb5dda"],
        # ("SimpleMazeUntil2", 0): ["50a47e077f8d435d8aa90404d68f4bed", "2161a861f62d4439b4f439466617534c"],

        # ("SimpleMazeLoopWithObs", 0): ["0307b27d6d5241abb0bdeda7d277e3d8", "7a20996353484f7aaab22fe9464941f9"],
        # ("SimpleMazeLoopWithObs", 1): ["aa33459fd44941ada12e681e0617b1f3", "d1bb96c20e9c43aa8be4a2eae734ff3d"],
        # ("SimpleMazeLoopWithObs", 2): ["ce73a3d09e09482fba19a1e4ee63498b", "5e50dbf13d9e42558ee921b51be0a8b1"],
        # ("SimpleMazeLoopWithObs", 3): ["1f8be0fa1dff44debf8e37ab01ff8664", "3c8dc5a023f24832b9540bc4a00634e6"],
        # ("SimpleMazeLoopWithObs", 4): ["56530129fc1c4c49a703be53bcd53605", "3f54899feaf64caf8f1772a44d1393ac"],

        ("SimpleMaze3DTwoSubgoals", 0): ["da4a8293ee952eb49fa88b", "2bd8801324e04f5fb7e45737a2c0229e"],
        ("SimpleMaze3DTwoSubgoals", 1): ["5045ec9f4ed6a8150f8395", "4fd36e121cc448a7bbdf8811263f67c6"],
        ("SimpleMaze3DTwoSubgoals", 2): ["314f34bdc87c424f8de757", "f4e8f14327b84491b84162b86afa5f20"],
        ("SimpleMaze3DTwoSubgoals", 3): ["1c4e8ab835c9a1ccf0cdae", "d91501b6de0f41c18009c444d416d9c9"],
        ("SimpleMaze3DTwoSubgoals", 4): ["1d4dbe89dc5c6b5a3733eb", "461f9ea00be54f108f8d21fbcccf8af4"],

        ("SimpleMaze3DBranching1", 0): ["1244e32a027e47d2885cbca", "cec61ba8c18d4b3fb92abccb4461779b"],
        ("SimpleMaze3DBranching1", 1): ["c474a0da539c1de4c7c8566", "d7d622abff134a3eabc2ea714ef44ddc"],
        ("SimpleMaze3DBranching1", 2): ["18442a8a6c385ea63034400", "36653217238d4632bb87d664458bb8dc"],
        ("SimpleMaze3DBranching1", 3): ["13d4757aa44ece18f44f222", "dd2214da609c451987bce1d50df5ec8b"],
        ("SimpleMaze3DBranching1", 4): ["e2e463f8b9401e391e7ac42", "bc70d482cf9948718b4cef8a89213435"],

        ("SimpleMaze3DObligationConstraint2", 0): ["e81556962e474392b691ce3cc8250a50"],
        ("SimpleMaze3DObligationConstraint2", 1): ["f7c31bbb524f4e42b369351620783278"],
        ("SimpleMaze3DObligationConstraint2", 2): ["a387b811301b4f21a89181b38d13e4a2"],
        ("SimpleMaze3DObligationConstraint2", 3): ["7dd1298b45ed472986c99f42b6033fb5"],
        ("SimpleMaze3DObligationConstraint2", 4): ["e3173c889c2d41e4a4c2de53ace92866"],

        ("SimpleMaze3DUntil2", 0): ["79e7d4d52c6d4b95a7b66199156379bf", "731bf0f33f2a408aacc6f6c249cc0ec6"],
        ("SimpleMaze3DUntil2", 1): ["28ac7a7afd704944b5a056d6986d3c20", "f56fd9aeb3e24e30a871879c78ef391a"],
        ("SimpleMaze3DUntil2", 2): ["ffd68e5ee8464e19825292b209752e0e", "35f13eb87248442c87f47afe4cda1b57"],
        ("SimpleMaze3DUntil2", 3): ["4cbd862b5c4740fbb37d13828bbea5e8", "669a826c8c78452fb59069221aac0cb0"],
        ("SimpleMaze3DUntil2", 4): ["d107fef260c14f058c2444bf5433f3d5", "5c5bebb499a44498bc04afb4b36505e5"],


        ("SimpleMaze3DTwoSubgoals", 0): ["025cbdf778da4a8293ee952eb49fa88b", "2bd8801324e04f5fb7e45737a2c0229e"],
        ("SimpleMaze3DTwoSubgoals", 1): ["b432b1950a5045ec9f4ed6a8150f8395", "4fd36e121cc448a7bbdf8811263f67c6"],
        ("SimpleMaze3DTwoSubgoals", 2): ["9a2ee03de6314f34bdc87c424f8de757", "f4e8f14327b84491b84162b86afa5f20"],
        ("SimpleMaze3DTwoSubgoals", 3): ["6241e8456c1c4e8ab835c9a1ccf0cdae", "d91501b6de0f41c18009c444d416d9c9"],
        ("SimpleMaze3DTwoSubgoals", 4): ["08b44f2e871d4dbe89dc5c6b5a3733eb", "461f9ea00be54f108f8d21fbcccf8af4"],

        ("SimpleMaze3DBranching1", 0): ["e9ac0d8961244e32a027e47d2885cbca", "cec61ba8c18d4b3fb92abccb4461779b"],
        ("SimpleMaze3DBranching1", 1): ["1de2a17d1c474a0da539c1de4c7c8566", "d7d622abff134a3eabc2ea714ef44ddc"],
        ("SimpleMaze3DBranching1", 2): ["eaeda7be618442a8a6c385ea63034400", "36653217238d4632bb87d664458bb8dc"],
        ("SimpleMaze3DBranching1", 3): ["5224c60df13d4757aa44ece18f44f222", "dd2214da609c451987bce1d50df5ec8b"],
        ("SimpleMaze3DBranching1", 4): ["8dabf00e5e2e463f8b9401e391e7ac42", "bc70d482cf9948718b4cef8a89213435"],

        ("SimpleMaze3DObligationConstraint2", 0): ["e81556962e474392b691ce3cc8250a50"],
        ("SimpleMaze3DObligationConstraint2", 1): ["f7c31bbb524f4e42b369351620783278"],
        ("SimpleMaze3DObligationConstraint2", 2): ["a387b811301b4f21a89181b38d13e4a2"],
        ("SimpleMaze3DObligationConstraint2", 3): ["7dd1298b45ed472986c99f42b6033fb5"],
        ("SimpleMaze3DObligationConstraint2", 4): ["e3173c889c2d41e4a4c2de53ace92866"],

        # ("SimpleMaze3DLoopWithObs", 0): ["7927312a1ca149bf914f941754351bdf", "9ce829843de646a594833b0b43b77897"],
        # ("SimpleMaze3DLoopWithObs", 1): ["bc2a4c1282e34e5ea42467d7336a23fd", "43b6823cc52a495cb48fc1c34985e9d4"],
        # ("SimpleMaze3DLoopWithObs", 2): ["fb93d90eec37495d99e37f21ea18300c", "3e8f190e0ce545cdb0fbfb83788e1f79"],
        # ("SimpleMaze3DLoopWithObs", 3): ["e24dc28a756f49eaa148d6e320ccfb46", "486424d7f91645b291cfd18c06e2d6e9"],
        # ("SimpleMaze3DLoopWithObs", 4): ["a62708127f0e4828916463ca78413d17", "ba29e952fc6e41e0bbc4697bdc9d400e"],


        # ("AntMazeTwoSubgoals", 0): ["731ee5b9391249cd98add1b8049cab4c", "90739cab34244620b6fc70bbabc504de"],
        # ("AntMazeTwoSubgoals", 1): ["2d5ee614fde54ae2adb69bb5e6a53c83", "7db38d2d1afe49b880e45f35e84b1add"],
        # ("AntMazeTwoSubgoals", 2): ["85433bb2436041bd930781fc595a5d42", "7e655e0ef8fa41a1b81aad65d223b1e2"],
        # ("AntMazeBranching1", 0): ["99a314ccad384a8fbb8729329f3fa4ff", "4bc3c3f314304061adb4b16ee8fb2e7b"],
        # ("AntMazeBranching1", 1): ["c30f3f0db7334a8d91a4e1274922a23e", "ee35b1c020114df0a9a2180f5e7961ff"],
        # ("AntMazeBranching1", 2): ["90bc2965466d48d7998be863525cdc08", "1136c41b700945c589f57bef0190924f"],
        # ("AntMazeObligationConstraint3", 0): ["ceadd5cc14da408c86bf0e2f4fa2c179"],
        # ("AntMazeObligationConstraint3", 1): ["3046f9b8b8154a55b80a27dda8383836"],
        # ("AntMazeObligationConstraint3", 2): ["e4184c8092a84fe697519675b3c1539b"],

        ("AntMazeTwoSubgoals", 0): ["1d4f758e7d3f41ed8146172965af150a", "89d7a70826e049a2b7ed774e7a91dd4d"],
        ("AntMazeTwoSubgoals", 1): ["487d3c75b79b438da56f1d668382f43a", "b3808e78027a45c79e22d02f31e998fb"],
        ("AntMazeTwoSubgoals", 2): ["3c7432860bb8462ba34d3e0c4ca8dc91", "a1ee1a3f69bb4b9f8c84f936ad55b2d0"],
        ("AntMazeTwoSubgoals", 3): ["6ce5c53879a84682a19ba93ce93143b8", "675097ecfd91442cb586597dd0f69e4c"],
        ("AntMazeTwoSubgoals", 4): ["c4e5c338c0a748609bc9b3d7fe459464", "7810feb8aadb430abdf20fb68f375e97"],

        ("AntMazeBranching1", 0): ["d633db311de74e9c9c3439ab1c121717", "7bfef5cb9cdf4017bc15952895dcdfdb"],
        ("AntMazeBranching1", 1): ["eed2918b4d6442b9a066d1ed8ee28bfd", "d757fb5d938d496f9a1fa118e8fea42f"],
        ("AntMazeBranching1", 2): ["25495c4d69104af19d895a4c3d126743", "639b41dc13004acf9bd10edbcef1b518"],
        ("AntMazeBranching1", 3): ["639b41dc13004acf9bd10edbcef1b518", "2bc73531cc5e46be88b9e75a04dcbc24"],
        ("AntMazeBranching1", 4): ["4a2a475e37764c658e40cd924d821406", "8287f386452c4f4e8b14528f9eeb6577"],

        ("AntMazeObligationConstraint3", 0): ["c173c35d9a91409e8a8b434467d7137e"],
        ("AntMazeObligationConstraint3", 1): ["84d73d81467749a5abd32eff49377dbc"],
        ("AntMazeObligationConstraint3", 2): ["5f5e9237c13e432fa440dbc500226154"],
        ("AntMazeObligationConstraint3", 3): ["fb2d5d60e342463ebe8fe4e1344129bd"],
        ("AntMazeObligationConstraint3", 4): ["0f59115d06534bf593f8c1a5fdac7827"],

        ("AntMazeUntil1", 0): ["d6284d3b9b184c559ab3238872689d23", "b61be1051c9e41af931607f932768622"],
        ("AntMazeUntil1", 1): ["8fccd691a71941aaa2376a2837f10593", "59bab8c205ee4a2485631a385a935d64"],
        ("AntMazeUntil1", 2): ["f1b804d332f14dfbb1e929e65cd6a480", "b1e63f02e8124a95a1edaddc4485e236"],
        ("AntMazeUntil1", 3): ["7a5bbbeeabf648f98c66e56d66af3da0", "ab42d32fff20488081ee40caac3a397f"],
        ("AntMazeUntil1", 4): ["3e0ed8cd0f6840828354ef8a6315256f", "5cca0460519145a5bb3e11bd91ed01a7"],

        # ("AntMazeLoopWithObs", 0): ["875fda3294a74904b40435136861c136", "8ee9569ec2f847898775b28a86ac1211"],
        # ("AntMazeLoopWithObs", 1): ["49a762fab7db4b4f97b374b084202234", "6bfc406bcde64c24b2fcf5219b083d91"],
        # ("AntMazeLoopWithObs", 2): ["c2ab45f7a60741b1b706641321388100", "d4886c5223734cf9b5f0e24d05cf759a"],
        # ("AntMazeLoopWithObs", 3): ["d4886c5223734cf9b5f0e24d05cf759a", "189cbcf29d87468d8d14b1522807afc7"],
        # ("AntMazeLoopWithObs", 4): ["32b38fee2a7049658e3c0759ab99934f", "c8121ffe4cf741319bf44ca51f399342"],



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
