import jax
import numpy as np

from gymnasium.spaces import Space
from gymnasium import spaces

from brax import base
from brax.envs.base import PipelineEnv, State
from brax.envs.ant import Ant as BraxAnt
from brax.io import mjcf
from etils import epath
import mujoco

from task_aware_skill_composition.brax.envs.base import GoalConditionedEnv, load_and_configure_xml


class Ant(GoalConditionedEnv):
    def __init__(
            self,
            ctrl_cost_weight=0.5,
            use_contact_forces=False,
            contact_cost_weight=5e-4,
            healthy_reward=1.0,
            terminate_when_unhealthy=False,
            healthy_z_range=(0.0, 20.0),
            contact_force_range=(-1.0, 1.0),
            reset_noise_scale=0.1,
            exclude_current_positions_from_observation=False,
            backend='generalized',
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
                            gear=200 * jp.ones_like(sys.actuator.gear)
                    )
            )

        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

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
        return BraxAnt.step(self, state, action)

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        return BraxAnt._get_obs(self, pipeline_state)
