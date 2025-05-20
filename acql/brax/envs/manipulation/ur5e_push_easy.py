import os
from brax import base
from brax.envs.base import State
from brax.io import mjcf
import mujoco
import jax
from jax import numpy as jnp

from acql.brax.envs.manipulation.ur5e_envs import UR5eEnvs


"""
Push-Easy: Move a cube from a random location on the blue region to a random goal on the adjacent red region. The regions are very small.
- Observation space: 18-dim obs + 3-dim goal.
- Action space:      5-dim, each element in [-1, 1], corresponding to target angles for joints 1, 2, 4, 6, and finger closedness.

See _get_obs() and ArmEnvs._convert_action() for details.
"""
class UR5ePushEasy(UR5eEnvs):
    def _get_xml_path(self):
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'assets', "ur5e_push_easy.xml")

    @property
    def action_size(self) -> int:
        return 5 # Override default (actuator count)
    
    # See ArmEnvs._set_environment_attributes for descriptions of attributes
    def _set_environment_attributes(self):
        self.env_name = "ur5e_push_easy"
        self.episode_length = 150
        
        self.pos_indices = jnp.array([0, 1, 2]) # Cube position
        self.goal_indices = jnp.array([15, 16, 17]) # Cube position
        self.completion_goal_indices = jnp.array([0, 1, 2]) # Identical
        self.state_dim = 15
        self.goal_dist = 0.1
        self.goal_reach_thresh = 0.1

        self.arm_noise_scale = 0
        self.cube_noise_scale = 0.1
        self.goal_noise_scale = 0.1
        
    def _get_initial_state(self, rng):
        rng, subkey1, subkey2 = jax.random.split(rng, 3)
        cube_q_xy = self.sys.init_q[:2] + self.cube_noise_scale * jax.random.uniform(subkey1, [2], minval=-1)
        cube_q_remaining = self.sys.init_q[2:7]
        target_q = self.sys.init_q[7:14]

        arm_q_default = jnp.array([1.5708, -2.2000, 1.9000, -1.3830, -1.5700,  0.0000]) # Start closer to the relevant area
        arm_q = arm_q_default + self.arm_noise_scale * jax.random.uniform(subkey2, [self.sys.q_size() - 14], minval=-1)
        
        q = jnp.concatenate([cube_q_xy] + [cube_q_remaining] + [target_q] + [arm_q])
        qd = jnp.zeros([self.sys.qd_size()])
        return q, qd
        
    def _get_initial_goal(self, pipeline_state: base.State, rng):
        rng, subkey = jax.random.split(rng)
        cube_goal_pos = jnp.array([0.1, 0.4, 0.03]) + jnp.array([self.goal_noise_scale, self.goal_noise_scale, 0]) * jax.random.uniform(subkey, [3], minval=-1)
        return cube_goal_pos

    def _random_target(self, rng):
        rng, subkey = jax.random.split(rng)
        cube_goal_pos = jnp.array([0.1, 0.4, 0.03]) + jnp.array([self.goal_noise_scale, self.goal_noise_scale, 0]) * jax.random.uniform(subkey, [3], minval=-1)
        return cube_goal_pos
        
    def _compute_goal_completion(self, obs, goal):
        # Goal occupancy: is the cube close enough to the goal? 
        current_cube_pos = obs[self.pos_indices]
        goal_pos = goal[:3]
        dist = jnp.linalg.norm(current_cube_pos - goal_pos)

        success = jnp.array(dist < self.goal_dist, dtype=float)
        success_easy = jnp.array(dist < 0.3, dtype=float)
        success_hard = jnp.array(dist < 0.03, dtype=float)
        return success, success_easy, success_hard
    
    def reached_goal(self, obs: jax.Array, threshold: float = 0.1):
        # Goal occupancy: is the end of the arm close enough to the goal?
        eef_pos = obs[..., self.pos_indices]
        goal_eef_pos = obs[..., self.goal_indices]
        dist = jnp.linalg.norm(eef_pos - goal_eef_pos)
        reached = jnp.array(dist < threshold, dtype=float)
        return reached

    def _update_goal_visualization(self, pipeline_state: base.State, goal: jax.Array) -> base.State:
        updated_q = pipeline_state.q.at[7:10].set(goal[:3]) # Only set the position, not orientation
        updated_pipeline_state = pipeline_state.replace(qpos=updated_q)
        return updated_pipeline_state

    def _get_obs(self, pipeline_state: base.State, goal: jax.Array, timestep) -> jax.Array:
        """
        Observation space (15-dim)
         - q_subset (9-dim): 3-dim cube position, 6-dim joint angles
         - End-effector (6-dim): position and velocity
        Note q is 20-dim: 7-dim cube position/angle, 7-dim goal marker position/angle, 6-dim joint angles
         
        Goal space (3-dim): position of cube
        """
        q_indices = jnp.array([0, 1, 2, 14, 15, 16, 17, 18, 19])
        q_subset = pipeline_state.q[q_indices]
        
        eef_index = 7 # Cube is 0, goal marker is 1, then links 1-6 are indices 2-7. The end-effector (eef) base is merged with link 7, so we say link 7 index = eef index.
        eef_x_pos = pipeline_state.x.pos[eef_index]
        eef_xd_vel = pipeline_state.xd.vel[eef_index]

        return jnp.concatenate([q_subset] + [eef_x_pos] + [eef_xd_vel] + [goal])

    def _get_arm_angles(self, pipeline_state: base.State) -> jax.Array:
        q_indices = jnp.arange(14, 20)
        return pipeline_state.q[q_indices]
