from brax import base
from brax.envs.base import State
from brax.io import mjcf
import mujoco
import jax
from jax import numpy as jnp

from achql.brax.envs.manipulation.ur5e_reach import UR5eReach


"""
Reach: Move end of arm to random goal.
- Observation space: 13-dim obs + 3-dim goal.
- Action space:      4-dim, each element in [-1, 1], corresponding to target angles for joints 1, 2, 4, 6.

See _get_obs() and ArmEnvs._convert_action() for details.
"""
class UR5eReachShelf(UR5eReach):
    def _get_xml_path(self):
        return "/home/tassos/phd/research/second-project/task-aware-skill-composition/achql/brax/envs/assets/ur5e_reach_shelf.xml"
    
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

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""

        # Run mujoco step
        pipeline_state0 = state.pipeline_state
        if "EEF" in self.env_name:
            action = self._convert_action_to_actuator_input_EEF(pipeline_state0, action)
        else:
            arm_angles = self._get_arm_angles(pipeline_state0)
            action = self._convert_action_to_actuator_input_joint_angle(action, arm_angles, delta_control=True)

        pipeline_state = self.pipeline_step(pipeline_state0, action)

        # Compute variables for state update, including observation and goal/reward
        timestep = state.info["timestep"] + 1 / self.episode_length
        obs = self._get_obs(pipeline_state, state.info["goal"], timestep)

        success, success_easy, success_hard = self._compute_goal_completion(obs, state.info["goal"])
        state.metrics.update(success=success, success_easy=success_easy, success_hard=success_hard)

        reward = success
        done = 0.0
        info = {**state.info, "timestep": timestep}

        new_state = state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done, info=info)
        return new_state
