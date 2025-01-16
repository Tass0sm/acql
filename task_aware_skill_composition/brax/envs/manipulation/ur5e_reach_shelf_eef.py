from brax import base
from brax.envs.base import State
from brax.io import mjcf
import mujoco
import jax
from jax import numpy as jnp

from task_aware_skill_composition.brax.envs.manipulation.ur5e_reach import UR5eReach


"""
Reach: Move end of arm to random goal.
- Observation space: 13-dim obs + 3-dim goal.
- Action space:      4-dim, each element in [-1, 1], corresponding to target angles for joints 1, 2, 4, 6.

See _get_obs() and ArmEnvs._convert_action() for details.
"""
class UR5eReachShelfEEF(UR5eReach):
    def _get_xml_path(self):
        return "/home/tassos/phd/research/second-project/task-aware-skill-composition/task_aware_skill_composition/brax/envs/assets/ur5e_reach_shelf_eef.xml"

    @property
    def action_size(self) -> int:
        return 3
    
    # See ArmEnvs._set_environment_attributes for descriptions of attributes
    def _set_environment_attributes(self):
        self.env_name = "ur5e_reach_shelf_EEF"
        self.episode_length = 100

        self.pos_indices = jnp.array([0, 1, 2]) # End-effector position
        self.goal_indices = jnp.array([6, 7, 8]) # End-effector position
        self.completion_goal_indices = jnp.array([0, 1, 2]) # Identical
        self.state_dim = 6
        self.goal_dist = 0.1

        self.eef_noise_scale = 0.2
        # self.goal_noise_scale = 0.2

        # Upper Right, Upper Left, Lower Left, Lower Right
        self.possible_goals = jnp.array([[0.225, 0.5, 0.5],
                                         [-0.225, 0.5, 0.5],
                                         [-0.225, 0.5, 0.1],
                                         [0.225, 0.5, 0.1]])

    def _get_initial_state(self, rng):
        rng, subkey = jax.random.split(rng)

        target_q = self.sys.init_q[:7] # 3 pos + 4 quat
        eef_q_default = jnp.array([0, 0.1, 0.3]) # Start closer to the relevant area
        eef_q = eef_q_default + self.eef_noise_scale * jax.random.uniform(subkey, [self.sys.q_size() - 7], minval=-1)

        q = jnp.concatenate([target_q] + [eef_q])
        qd = jnp.zeros([self.sys.qd_size()])
        return q, qd

    def _get_initial_goal(self, pipeline_state: base.State, rng):
        """
        Generate goals in a box. x: [-0.2, 0.2], y: [0.3, 0.7], z: [0.0, 0.8]
        """
        # goal = jnp.array([0, 0.5, 0.4]) + self.goal_noise_scale * jax.random.uniform(rng, [3], minval=-1)
        # return goal

        # idx = jax.random.randint(rng, (1,), 0, len(self.possible_goals))
        # return self.possible_goals[idx][0]

        return self._random_target(rng)

    def _get_obs(self, pipeline_state: base.State, goal: jax.Array, timestep) -> jax.Array:
        """
        Observation space (9-dim)
         - End of arm (6-dim): position and velocity
        Note q is 10-dim: 7-dim goal position/angle, 3-dim eef position
         
        Goal space (3-dim): position of end of arm
        """

        eef_index = 1 # Goal is at index 0, then link 0 is at index 1.
        eef_x_pos = pipeline_state.x.pos[eef_index]
        eef_xd_vel = pipeline_state.xd.vel[eef_index]

        return jnp.concatenate([eef_x_pos] + [eef_xd_vel] + [goal])

    def _random_target(self, rng: jax.Array) -> jax.Array:
        """Returns a random target location chosen from possibilities specified in the maze layout."""
        idx = jax.random.randint(rng, (1,), 0, len(self.possible_goals))
        return jnp.array(self.possible_goals[idx])[0]
