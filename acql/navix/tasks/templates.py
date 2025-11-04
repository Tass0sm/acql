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

    def f(timestep, c):
        pos = timestep.state.entities["player"].position
        return -norm(pos - c) + radius

    info = {"name": f"inside_circle ({center}, {radius})"}
    if has_goal:
        info["goal"] = center
    return stl.STLPredicate(var, f, 0.0, default_params=center, info=info)


def inside_box(
        var: Var,
        corner1: jax.Array,
        corner2: jax.Array,
        has_goal: bool = False
) -> stl.STLFormula:

    neg_eye = -jnp.eye(2)
    pos_eye = jnp.eye(2)
    A = jnp.vstack((neg_eye, pos_eye))
    b = jnp.concatenate((-corner1, corner2))
    # using formula for testing point is inside polytope.

    def f(timestep, unused):
        pos = timestep.state.entities["player"].position.squeeze()
        return (b - A @ pos).min(axis=-1)

    info = {"name": f"inside_box ({corner1}, {corner2})"}
    if has_goal:
        info["goal"] = corner1+(corner2-corner1)/2
    return stl.STLPredicate(var, f, 0.0, info=info)
