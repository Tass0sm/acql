import jax
import jax.numpy as jnp

import mediapy

from brax import envs

from task_aware_skill_composition.brax.envs.car import Car
from task_aware_skill_composition.brax.envs.drone import Drone


backend = 'mjx'  # @param ['generalized', 'positional', 'spring']

# env_name = "hopper"
# eval_env = envs.get_environment(env_name=env_name,
#                                 backend=backend)

eval_env = Car(backend=backend)
# eval_env = Drone(backend=backend)

jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)

# initialize the state
rng = jax.random.PRNGKey(0)
state = jit_reset(rng)

# grab a trajectory
rollout = [state.pipeline_state]
n_steps = 100
render_every = 1

ctrl = jnp.ones(eval_env.action_size) * 1.5
print(f"Made ctrl = {ctrl}")

for i in range(n_steps):
  state = jit_step(state, ctrl)

  if state.done:
      break

  rollout.append(state.pipeline_state)

mediapy.write_video(
    "./test_render.mp4",
    eval_env.render(rollout[::render_every], camera='track'),
    fps=1.0 / eval_env.dt / render_every
)
