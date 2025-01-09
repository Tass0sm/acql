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
class UR5eReachShelf(UR5eReach):
    def _get_xml_path(self):
        return "/home/tassos/phd/research/second-project/task-aware-skill-composition/task_aware_skill_composition/brax/envs/assets/ur5e_reach_shelf.xml"
    
    # See ArmEnvs._set_environment_attributes for descriptions of attributes
    def _set_environment_attributes(self):
        self.env_name = "ur5e_reach_shelf"
        self.episode_length = 100

        self.pos_indices = jnp.array([6, 7, 8]) # End-effector position
        self.goal_indices = jnp.array([12, 13, 14]) # End-effector position
        self.completion_goal_indices = jnp.array([6, 7, 8]) # Identical
        self.state_dim = 12
        self.goal_dist = 0.1

        self.arm_noise_scale = 0
        self.goal_noise_scale = 0.2
