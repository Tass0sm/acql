import jax
import jax.numpy as jnp

from brax.envs.base import Env, State, Wrapper

from acql.brax.envs.wrappers.automaton_wrapper import AutomatonWrapper


class AutomatonGoalWrapper(Wrapper):
    """Wrap product mdp to add the automaton state as part of the goal input

    G^A = G x Q
    Pg^A = Pg x {accepting_state}

    This is mainly intended to be used with off policy algorithms using HER,
    so the reward is not modified at all (for now).
    """

    def __init__(
            self,
            env: AutomatonWrapper,
    ):
        super().__init__(env)

        assert isinstance(env, AutomatonWrapper), "Input to this wrapper must be a Product MDP"

        original_obs_dim = env.observation_size

        if self.automaton.n_states > 1:
            # compute new indices for adding goal conditioning
            aut_state_indices = jnp.arange(original_obs_dim - self.automaton.n_states, original_obs_dim)
            self.pos_indices = jnp.concatenate((env.pos_indices, aut_state_indices))

            aut_state_goal_indices = jnp.arange(original_obs_dim, original_obs_dim + self.automaton.n_states)
            self.goal_indices = jnp.concatenate((env.goal_indices, aut_state_goal_indices))

    def _original_obs_and_goal(self, obs: jax.Array):
        # remove the new automaton goal appended to the observation
        if self.automaton.n_states > 1:
            aut_goal = obs[..., -self.automaton.n_states:]
            obs = obs[..., :-self.automaton.n_states]
            return obs, aut_goal
        else:
            aut_goal = obs[..., -0:]
            breakpoint()
            return obs, aut_goal

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)

        # sample the automaton state goal (its just the accepting state) and
        # concatenate it to the observation
        sampled_aut_goal = self.env.automaton.one_hot_encode(self.env.accepting_state)
        new_obs = jnp.concatenate((state.obs, sampled_aut_goal))
        state = state.replace(obs=new_obs)

        return state

    def step(self, state: State, action: jax.Array) -> State:

        orig_obs, aut_goal = self._original_obs_and_goal(state.obs)

        state = state.replace(obs=orig_obs)

        nstate = self.env.step(state, action)

        new_obs = jnp.concatenate((nstate.obs, aut_goal), axis=-1)
        nstate = nstate.replace(obs=new_obs)

        # In this environment, success is determined not only by reaching the
        # goal position, but also the goal automaton state
        success_easy = self.reached_goal(new_obs, threshold=2.0)
        success = self.reached_goal(new_obs, threshold=self.goal_dist)
        nstate.metrics.update(
            success_easy=success_easy,
            success=success,
        )

        return nstate

    def _split(self, x: jax.Array):
        non_aut = x[..., :-self.automaton.n_states]
        aut = x[..., -self.automaton.n_states:]
        return non_aut, aut

    def reached_goal(self, obs: jax.Array, threshold: float = 2.0):
        if self.automaton.n_states > 1:
            pos_non_aut, pos_aut = self._split(obs[..., self.pos_indices])
            goal_non_aut, goal_aut = self._split(obs[..., self.goal_indices])
            non_aut_dist = jnp.linalg.norm(pos_non_aut - goal_non_aut, axis=-1)
            aut_dist = jnp.all(pos_aut == goal_aut, axis=-1).astype(float)
            reached = jnp.array(non_aut_dist < threshold, dtype=float) * aut_dist
            return reached
        else:
            pos_non_aut = obs[..., self.pos_indices]
            goal_non_aut = obs[..., self.goal_indices]
            non_aut_dist = jnp.linalg.norm(pos_non_aut - goal_non_aut, axis=-1)
            reached = jnp.array(non_aut_dist < threshold, dtype=float)
            return reached

    @property
    def observation_size(self) -> int:
        if self.augment_obs and not self.strip_goal_obs:
            return self.env.observation_size + (self.automaton.n_states if self.automaton.n_states > 1 else 0)
        elif self.augment_obs:
            return self.env.observation_size - self.env.goal_indices.size + (self.automaton.n_states if self.automaton.n_states > 1 else 0)
        elif self.strip_goal_obs:
            return self.env.observation_size - self.env.goal_indices.size
        else:
            return self.env.observation_size
