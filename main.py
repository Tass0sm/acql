import functools

import jax

from achql.brax.agents.achql import train as achql
from achql.brax.tasks import get_task
from achql.brax.utils import make_aut_goal_cmdp

task = get_task("ant_maze", "two_subgoals")
spec = task.lo_spec

def progress_fn(num_steps, metrics, **kwargs):
    print(num_steps)
    print(metrics)

options = task.get_options()

extras = {
    "options": options,
    "specification": spec,
    "state_var": task.obs_var,
    "eval_environment": make_aut_goal_cmdp(task, randomize_goals=False, margin=0.0),
}

seed = 0
train_fn = functools.partial(achql.train, **task.hdcqn_her_hps)

make_inference_fn, params, _ = train_fn(
    environment=make_aut_goal_cmdp(task),
    progress_fn=progress_fn,
    seed=seed,
    **extras
)
