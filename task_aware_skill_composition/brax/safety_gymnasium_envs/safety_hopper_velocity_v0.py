"""Hopper environment with a safety constraint on velocity."""

from typing import Tuple

from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp

from brax.envs.hopper import Hopper as BraxHopperEnv

# from safety_gymnasium.utils.task_utils import add_velocity_marker, clear_viewer


class SafetyHopperVelocityEnv(BraxHopperEnv):
    """Hopper environment with a safety constraint on velocity."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._velocity_threshold = 0.37315
        # self.model.light(0).castshadow = False

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)
    
        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(
            rng2, (self.sys.qd_size(),), minval=low, maxval=hi
        )
    
        pipeline_state = self.pipeline_init(qpos, qvel)
    
        obs = self._get_obs(pipeline_state)
        reward, done, zero = jp.zeros(3)
        metrics = {
            'reward_forward': zero,
            'reward_ctrl': zero,
            'reward_healthy': zero,
            'x_position': zero,
            'x_velocity': zero,
        }

        return State(pipeline_state, obs, reward, done, metrics, { "cost": zero })
    
    def step(self, state: State, action: jax.Array) -> State:
        """Runs one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        assert pipeline_state0  is not None
        pipeline_state = self.pipeline_step(pipeline_state0, action)
        
        x_velocity = (
            pipeline_state.x.pos[0, 0] - pipeline_state0.x.pos[0, 0]
        ) / self.dt

        forward_reward = self._forward_reward_weight * x_velocity
        
        z, angle = pipeline_state.x.pos[0, 2], pipeline_state.q[2]
        state_vec = jp.concatenate([pipeline_state.q[2:], pipeline_state.qd])
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range
        min_state, max_state = self._healthy_state_range
        is_healthy = jp.all(
            jp.logical_and(min_state < state_vec, state_vec < max_state)
        )
        is_healthy &= jp.logical_and(min_z < z, z < max_z)
        is_healthy &= jp.logical_and(min_angle < angle, angle < max_angle)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy
            
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
        
        obs = self._get_obs(pipeline_state)
        reward = forward_reward + healthy_reward - ctrl_cost
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        state.metrics.update(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_healthy=healthy_reward,
            x_position=pipeline_state.x.pos[0, 0],
            x_velocity=x_velocity,
        )

        # COMPUTING THE CMDP COST (Distinct from ctrl_cost)
        cost = jp.where(x_velocity > self._velocity_threshold, 1.0, 0.0)
        state.info.update(
            cost=cost
        )
        
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )
