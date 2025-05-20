import jax
import jax.numpy as jnp
import numpy as np

from gymnasium.spaces import Space
from gymnasium import spaces

from brax import base
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.envs.ant import Ant as BraxAnt
from brax.io import mjcf
from etils import epath
import mujoco

from achql.brax.envs.base import GoalConditionedEnv, load_and_configure_xml


class Ant(GoalConditionedEnv):
    def __init__(
            self,
            ctrl_cost_weight=0.5,
            use_contact_forces=False,
            contact_cost_weight=5e-4,
            healthy_reward=1.0,
            terminate_when_unhealthy=False,
            healthy_z_range=(0.2, 1.0),
            contact_force_range=(-1.0, 1.0),
            reset_noise_scale=0.1,
            exclude_current_positions_from_observation=False,
            backend='generalized',
            target_velocity_vec=jnp.array([1.0, 0.0, 0.0]),
            wall_config=None,
            **kwargs,
    ):
        path = epath.resource_path('brax') / 'envs/assets/ant.xml'
        xml = load_and_configure_xml(path, wall_config=wall_config)
        sys = mjcf.loads(xml)

        n_frames = 5

        if backend in ['spring', 'positional']:
            sys = sys.tree_replace({'opt.timestep': 0.005})
            n_frames = 10

        if backend == 'mjx':
            sys = sys.tree_replace({
                    'opt.solver': mujoco.mjtSolver.mjSOL_NEWTON,
                    'opt.disableflags': mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    'opt.iterations': 1,
                    'opt.ls_iterations': 4,
            })

        if backend == 'positional':
            # TODO: does the same actuator strength work as in spring
            sys = sys.replace(
                    actuator=sys.actuator.replace(
                            gear=200 * jnp.ones_like(sys.actuator.gear)
                    )
            )

        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self._target_velocity_vec = target_velocity_vec

        self._ctrl_cost_weight = ctrl_cost_weight
        self._use_contact_forces = use_contact_forces
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
                exclude_current_positions_from_observation
        )

        if self._use_contact_forces:
            raise NotImplementedError('use_contact_forces not implemented.')

    @property
    def observation_space(self) -> Space:
        return spaces.Box(low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32)

    @property
    def action_space(self) -> Space:
        return spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    def reset(self, rng: jax.Array) -> State:
        return BraxAnt.reset(self, rng)

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        assert pipeline_state0 is not None
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt
        forward_reward = jnp.dot(velocity, self._target_velocity_vec)

        min_z, max_z = self._healthy_z_range
        is_healthy = jnp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        is_healthy = jnp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy
        ctrl_cost = self._ctrl_cost_weight * jnp.sum(jnp.square(action))
        contact_cost = 0.0

        obs = self._get_obs(pipeline_state)
        reward = forward_reward + healthy_reward - ctrl_cost - contact_cost
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        state.metrics.update(
                reward_forward=forward_reward,
                reward_survive=healthy_reward,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                x_position=pipeline_state.x.pos[0, 0],
                y_position=pipeline_state.x.pos[0, 1],
                distance_from_origin=math.safe_norm(pipeline_state.x.pos[0]),
                x_velocity=velocity[0],
                y_velocity=velocity[1],
                forward_reward=forward_reward,
        )
        return state.replace(
                pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        return BraxAnt._get_obs(self, pipeline_state)
