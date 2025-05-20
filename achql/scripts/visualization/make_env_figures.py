import jax

from achql.tasks import get_task

import mediapy

for env_name, task_name, seed in [
        ("UR5ePushHard", "ObligationConstraint", 0),
]:
    task = get_task(env_name, task_name)
    env_tag = type(task.env).__name__
    env = task.env

    rng = jax.random.PRNGKey(seed)
    state = env.reset(rng)

    mediapy.write_image(
        f'./env-images/{env_tag}_overview.png',
        env.render(state.pipeline_state,
                   height=2*240,
                   width=2*240,
                   camera='overview')
    )

# working with default task

# task = get_task("simple_maze", "subgoal")
# task = get_task("simple_maze", "two_subgoals")
# task = get_task("simple_maze", "branching1")
# task = get_task("simple_maze", "obligation2")

# task = get_task("simple_maze_3d", "two_subgoals")
# task = get_task("simple_maze_3d", "branching1")
# task = get_task("simple_maze_3d", "obligation2")

# task = get_task("ant_maze", "two_subgoals")
# task = get_task("ant_maze", "branching1")
# task = get_task("ant_maze", "obligation2")

# task = get_task("simple_maze_3d", "subgoal")
# task = get_task("drone_maze", "true")
# task = get_task("turtlebot", "true")

# task = get_task("ant_maze", "true")
# task = get_task("ant_maze", "umaze_constraint")

# task = get_task("panda", "reach")
# task = get_task("ur5e", "reach")
