from typing import Callable

from brax.envs.base import Env, State, Wrapper

import jax
from jax import numpy as jnp


class LOFWrapper(Wrapper):
    """Modifies environment reward to be R_epsilon(s), which is R_e(s) + R_s(T_P_S(s))
    R_e is not defined in the paper.
    R_s is a safety reward function .
    """

    def __init__(
            self,
            env: Env,
            state_var,
            subgoal_proposition: Callable[..., jnp.ndarray],
            safety_reward_function: Callable[..., jnp.ndarray],
            distance_cost: bool = False,
            strip_goal_obs: bool = True,
    ):
        super().__init__(env)

        self.possible_starts_and_goals = jnp.concatenate((self.possible_starts, self.possible_goals), axis=0)

        self.state_var = state_var
        self.subgoal_proposition = subgoal_proposition
        self.subgoal = subgoal_proposition.info["goal"]
        self.safety_reward_function = safety_reward_function

        self.strip_goal_obs = strip_goal_obs
        self.original_obs_dim = env.observation_size

        # Set up masks for obs manipulation...
        mask = jnp.zeros(self.original_obs_dim, dtype=bool)
        mask = mask.at[env.goal_indices].set(True)
        self.non_goal_indices = ~mask

        self.distance_cost = distance_cost

    def original_obs(self, obs):
        # takes an obs from this mdp, and returns an obs from the original
        # underlying mdp

        if self.strip_goal_obs:
            dummy_goal = jnp.zeros_like(self.goal_indices)
            obs = jnp.insert(obs, self.goal_indices, dummy_goal, axis=-1)

        return obs

    def stripped_obs(self, obs):
        return obs[..., self.non_goal_indices]

    def reset(self, rng: jax.Array) -> State:
        start_rng, rng = jax.random.split(rng)

        state = self.env.reset(rng)

        # get original pipeline_state
        pipeline_state0 = state.pipeline_state

        # override the start position distribution to properly train the logical option policy
        start = self._random_start(start_rng)
        q = pipeline_state0.q.at[:self.pos_indices.size].set(start)
        qd = pipeline_state0.qd

        # set pipeline state with new q
        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)
        state.replace(pipeline_state=pipeline_state, obs=obs)

        # R_E
        if self.distance_cost:
            position = state.obs[self.pos_indices]
            base_reward = -jnp.linalg.norm(self.subgoal - position)
        else:
            base_reward = -1

        # R_S
        safety_reward = 0

        # R_Epsilon (for option env)
        new_reward = base_reward + safety_reward
        state = state.replace(reward=new_reward)

        if self.strip_goal_obs:
            new_obs = state.obs[..., self.non_goal_indices]
            state = state.replace(obs=new_obs)

        if "dist" in state.metrics:
            del state.metrics["dist"]
        if "success" in state.metrics:
            del state.metrics["success"]
        if "success_easy" in state.metrics:
            del state.metrics["success_easy"]
        if "success_hard" in state.metrics:
            del state.metrics["success_hard"]
        if "forward_reward" in state.metrics:
            del state.metrics["forward_reward"]
        if "reward_contact" in state.metrics:
            del state.metrics["reward_contact"]
        if "reward_ctrl" in state.metrics:
            del state.metrics["reward_ctrl"]
        if "reward_forward" in state.metrics:
            del state.metrics["reward_forward"]
        if "reward_survive" in state.metrics:
            del state.metrics["reward_survive"]

        state.metrics.update(
            proposition_success=0.0,
        )

        return state

    def step(self, state: State, action: jax.Array) -> State:
        nstate = self.env.step(state, action)

        ##########
        # calculate reward to promote fast termination while staying safe.
        ##########

        # R_E
        if self.distance_cost:
            position = nstate.obs[self.pos_indices]
            base_reward = -jnp.linalg.norm(self.subgoal - position)
        else:
            base_reward = -1

        # R_S
        safety_reward = self.safety_reward_function(action, nstate.obs)

        # R_Epsilon (for option env)
        new_reward = base_reward + safety_reward
        nstate = nstate.replace(reward=new_reward)

        ##########
        # terminate when subgoal proposition is true
        ##########

        one = jnp.ones_like(nstate.done)
        subgoal_satisfaction = self.subgoal_proposition({ self.state_var.idx: jnp.expand_dims(nstate.obs, 0) }).squeeze()
        done = jnp.where(subgoal_satisfaction > 0, one, nstate.done)
        nstate = nstate.replace(done=done)

        if self.strip_goal_obs:
            new_obs = nstate.obs[..., self.non_goal_indices]
            nstate = nstate.replace(obs=new_obs)


        if "dist" in state.metrics:
            del state.metrics["dist"]
        if "success" in state.metrics:
            del state.metrics["success"]
        if "success_easy" in state.metrics:
            del state.metrics["success_easy"]
        if "success_hard" in state.metrics:
            del state.metrics["success_hard"]
        if "forward_reward" in state.metrics:
            del state.metrics["forward_reward"]
        if "reward_contact" in state.metrics:
            del state.metrics["reward_contact"]
        if "reward_ctrl" in state.metrics:
            del state.metrics["reward_ctrl"]
        if "reward_forward" in state.metrics:
            del state.metrics["reward_forward"]
        if "reward_survive" in state.metrics:
            del state.metrics["reward_survive"]

        nstate.metrics.update(
            proposition_success=jnp.where(subgoal_satisfaction > 0, 1.0, 0.0)
        )

        return nstate

    @property
    def observation_size(self) -> int:
        if not self.strip_goal_obs:
            return self.env.observation_size
        else:
            return self.env.observation_size - self.env.goal_indices.size

    def _random_start(self, rng: jax.Array) -> jax.Array:
        idx = jax.random.randint(rng, (1,), 0, len(self.possible_starts_and_goals))
        return jnp.array(self.possible_starts_and_goals[idx])[0]
