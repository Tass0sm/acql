from functools import reduce
from typing import List, Tuple

import jax
import jax.numpy as jnp
from jax.numpy.linalg import norm

from acql.stl import Var
import acql.stl as stl


def true_exp(var: Var):
    true = stl.STLPredicate(var, lambda s, _: 999, lower_bound=0.0, info={"true": True})
    return true


def inside_circle(
        var: Var,
        center: jax.Array,
        radius: float,
        has_goal: bool = False,
) -> stl.STLFormula:

    def f(s, c):
        return -norm(s - c) + radius

    info = {"name": f"inside_circle ({center}, {radius})"}
    if has_goal:
        info["goal"] = center
    return stl.STLPredicate(var, f, 0.0, default_params=center, info=info)


def outside_circle(
        var: Var,
        center: jax.Array,
        radius: float
) -> stl.STLFormula:

    def f(s, c):
        return norm(s - c) - radius

    return stl.STLPredicate(var, f, 0.0, default_params=center, info={"name": f"outside_circle ({center}, {radius})"})


def inside_box(
        var: Var,
        corner1: jax.Array,
        corner2: jax.Array,
        has_goal: bool = False
) -> stl.STLFormula:

    neg_eye = -jnp.eye(var.dim)
    pos_eye = jnp.eye(var.dim)
    A = jnp.vstack((neg_eye, pos_eye))
    b = jnp.concatenate((-corner1, corner2))
    # using formula for testing point is inside polytope.

    def f(s, unused):
        return (b - A @ s).min(axis=-1)

    info = {"name": f"inside_box ({corner1}, {corner2})"}
    if has_goal:
        info["goal"] = corner1+(corner2-corner1)/2
    return stl.STLPredicate(var, f, 0.0, info=info)


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
