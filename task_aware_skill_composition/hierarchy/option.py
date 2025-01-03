from typing import Optional, Callable

import jax
import jax.numpy as jnp

from brax.training import types
from brax.training.types import PRNGKey


class BernoulliTerminationPolicy:
    def __init__(self, p):
        self.p = p

    def __call__(self, s_t, key):
        # shape=option_state.option_beta.shape
        termination = jax.random.bernoulli(key, p=self.p).astype(jnp.int32)
        return termination


class FixedLengthTerminationPolicy:
    def __init__(self, t):
        self.t = t

    def __call__(self, s_t, key):
        raise NotImplementedError

class Option:
    def __init__(
            self,
            name,
            networks,
            params,
            inference_fn,
            termination_policy: Callable = BernoulliTerminationPolicy(0.2),
            adapter: Optional[Callable] = None,
    ):
        self.name = name

        # pi_o
        self.networks = networks
        self.params = params
        self._inference_fn = inference_fn

        # beta_o
        self.termination_policy = termination_policy

        # The option may only consider a subset of the observation/state space
        if adapter is not None:
            self.obs_adapter = adapter
        else:
            self.obs_adapter = lambda x: x

    def inference(
            self,
            observations: types.Observation,
            key_sample: PRNGKey
    ):
        return self._inference_fn(self.obs_adapter(observations), key_sample)

    def termination(
            self,
            s_t,
            key
    ):
        return self.termination_policy(s_t, key)

    def log_prob(self, state, action):
        logits = self.networks.policy_network.apply(*self.params, state)
        return self.networks.parametric_action_distribution.log_prob(logits, action)
