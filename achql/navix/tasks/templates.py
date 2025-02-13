from functools import reduce
from typing import List, Tuple

import jax
import jax.numpy as jnp
from jax.numpy.linalg import norm
import einops

from achql.stl.var import Var
import achql.stl.expression_jax2 as stl


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
