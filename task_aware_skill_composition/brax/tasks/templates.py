from functools import reduce
from typing import List, Tuple

import jax
from jax.numpy.linalg import norm

from corallab_stl.var import Var
import corallab_stl.expression_jax2 as stl


def inside_circle(
        var: Var,
        center: jax.Array,
        radius: float
) -> stl.STLFormula:

    def f(s):
        return -norm(s - center) + radius

    return stl.STLPredicate(var, f, 0.0)


def outside_circle(
        var: Var,
        center: jax.Array,
        radius: float
) -> stl.STLFormula:

    def f(s):
        return norm(s - center) - radius

    return stl.STLPredicate(var, f, 0.0)


def sequence(
        var: Var,
        goals: jax.Array,
        seq_times: List[Tuple],
        close_radius: float = 0.3,
) -> stl.STLFormula:
    """
    Sequential goal specification
    :param var: variable used by expression
    :param goals: goals in sequence
    :param seq_times: time slots for reach each goal
    :param close_radius: how close is close
    :return: stl.STLFormula
    """
    predicates = [inside_circle(wp, close_radius) for wp in goals]
    eventually_exps = [stl.STLTimedEventually(p, *ts) for p, ts in zip(predicates, seq_times)]
    sequence_exp = reduce(lambda a, b: stl.STLAnd(a, b), eventually_exps)
    return sequence_exp
