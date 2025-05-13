"""Wrapper for creating Semi-MDP with set of options and existing environment."""

from typing import Sequence, Callable, Dict, Optional, Tuple

from brax.base import System
from brax.envs.base import Env, State, Wrapper
from brax.training.types import PRNGKey
from flax import struct, nnx
import jax
from jax import numpy as jnp

from achql.hierarchy.option import Option, FixedLengthTerminationPolicy
from achql.brax.envs.wrappers.automaton_multi_goal_conditioned_wrapper import AutomatonMultiGoalConditionedWrapper
from achql.brax.envs.wrappers.automaton_goal_conditioned_wrapper import AutomatonGoalConditionedWrapper
from achql.brax.envs.wrappers.automaton_reward_machine_wrapper import AutomatonRewardMachineWrapper
from achql.brax.envs.wrappers.automaton_wrapper import AutomatonWrapper
from achql.brax.utils import recursive_is_instance


class OptionsWrapper(Wrapper):
    """Adds to environment information necessary for options framework"""

    def __init__(
            self,
            env: Env,
            options: Sequence[Option],
            discounting: float,
            takes_key = True,
            use_sum_cost_critic = False,
    ):
        super().__init__(env)

        self.options = options
        self.takes_key = takes_key

        if recursive_is_instance(env, AutomatonMultiGoalConditionedWrapper):
            self.extra_adapter = lambda x: x[..., :env.no_goal_obs_dim]
        elif recursive_is_instance(env, AutomatonGoalConditionedWrapper):
            self.extra_adapter = lambda x: x[..., :env.no_goal_obs_dim]
        elif recursive_is_instance(env, AutomatonRewardMachineWrapper):
            self.extra_adapter = lambda x: x[..., :env.state_dim]
        elif recursive_is_instance(env, AutomatonWrapper) and env.augment_obs:
            self.extra_adapter = lambda x: x[..., :env.state_dim]
        # elif hasattr(env, "automaton") and hasattr(env, "strip_goal_obs") and env.strip_goal_obs and env.augment_obs:
        #     self.extra_adapter = env.original_obs
        # elif hasattr(env, "automaton") and env.augment_obs:
        #     self.extra_adapter = lambda x: x[..., :env.original_obs_dim]
        #     # self.extra_adapter = env.no_automaton_obs
        else:
            self.extra_adapter = lambda x: x

        self.num_options = len(options)
        self.discounting = discounting

        self.use_sum_cost_critic = use_sum_cost_critic

        # we need this to prevent infinite loops if an option never decides to terminate.
        self.max_option_iters = 100

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info['low_steps'] = jnp.zeros(rng.shape[:-1])
        if 'cost' not in state.info:
            state.info['cost'] = 0.0
        return state

    # def step(self, state: State, option: jax.Array, key: PRNGKey) -> State:
    def step(self, state: State, option_and_key: Tuple[jax.Array, PRNGKey]) -> State:
        """Turning the MDP into a Semi-MDP requries passing a PRNG key to the
        step function. To stay compatible with the existing brax training
        wrappers, this key is passed in a tuple with the option (which was
        originally the action).

        """

        if self.takes_key:
            o_t, key = option_and_key
        else:
            o_t = option_and_key
            key = jax.random.PRNGKey(0)

        if isinstance(self.options[0].termination_policy, FixedLengthTerminationPolicy):
            length = self.options[0].termination_policy.t

            def for_body(unused, x):
                s_t, iters, r_t, c_t, key = x
                key, inf_key = jax.random.split(key)
                a_t, _ = jax.lax.switch(o_t, [o.inference for o in self.options], self.extra_adapter(s_t.obs), inf_key)
                s_t1 = self.env.step(s_t, a_t)
                r_t1 = r_t + jnp.pow(self.discounting, iters) * s_t1.reward

                if self.use_sum_cost_critic:
                    c_t1 = c_t + jnp.pow(self.discounting, iters) * s_t1.info["cost"]
                else:
                    c_t1 = jnp.minimum(c_t, s_t1.info.get("cost", 0.0))

                return (s_t1, iters+1, r_t1, c_t1, key)

            if self.use_sum_cost_critic:
                cost_identity = 0.0
            else:
                # positive infinity is the identity element for the min operation
                cost_identity = jnp.inf

            # run body <length> times
            fstate, k, freward, fcost, _ = jax.lax.fori_loop(0, length, for_body, (state, 0, 0, cost_identity, key))

        else:
            def while_cond(x):
                s_t, iters, _, _, key = x
                beta_t = jax.lax.switch(o_t, [o.termination for o in self.options], self.extra_adapter(s_t.obs), key)
                return jnp.logical_and(beta_t != 1, iters < self.max_option_iters)

            def while_body(x):
                s_t, iters, r_t, c_t, key = x
                key, inf_key = jax.random.split(key)
                a_t, _ = jax.lax.switch(o_t, [o.inference for o in self.options], self.extra_adapter(s_t.obs), inf_key)
                s_t1 = self.env.step(s_t, a_t)
                r_t1 = r_t + jnp.pow(self.discounting, iters) * s_t1.reward

                if self.use_sum_cost_critic:
                    c_t1 = c_t + jnp.pow(self.discounting, iters) * s_t1.info["cost"]
                else:
                    c_t1 = jnp.minimum(c_t, s_t1.info.get("cost", 0.0))

                return (s_t1, iters+1, r_t1, c_t1, key)
    
            if self.use_sum_cost_critic:
                cost_identity = 0.0
            else:
                cost_identity = jnp.inf

            # run body at least once
            nstate, steps, reward, cost, nkey = while_body((state, 0, 0, cost_identity, key))
    
            # then continue running until terminated
            fstate, k, freward, fcost, _ = jax.lax.while_loop(while_cond, while_body, (nstate, steps, reward, cost, nkey))
    
        # update cost/reward of final state with the option history reward
        fstate.info.update(cost=fcost)
        fstate = fstate.replace(reward=freward)

        # update state info
        fstate.info['low_steps'] = state.info['low_steps'] + k

        return fstate
