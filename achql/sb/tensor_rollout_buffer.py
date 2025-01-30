# import warnings
# from abc import ABC, abstractmethod
# from typing import Any, Dict, Generator, List, Optional, Tuple, Union
from typing import Optional

import numpy as np
import torch as th
from gymnasium import spaces

# from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
#     DictReplayBufferSamples,
#     DictRolloutBufferSamples,
#     ReplayBufferSamples,
    RolloutBufferSamples,
)
# from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.buffers import RolloutBuffer


"""Common aliases for type hints"""

from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, NamedTuple, Optional, Protocol, SupportsFloat, Tuple, Union

import gymnasium as gym
import numpy as np
import torch as th

# Avoid circular imports, we use type hint as string to avoid it too
if TYPE_CHECKING:
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import VecEnv

GymEnv = Union[gym.Env, "VecEnv"]
GymObs = Union[Tuple, Dict[str, Any], np.ndarray, int]
GymResetReturn = Tuple[GymObs, Dict]
AtariResetReturn = Tuple[np.ndarray, Dict[str, Any]]
GymStepReturn = Tuple[GymObs, float, bool, bool, Dict]
AtariStepReturn = Tuple[np.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]
TensorDict = Dict[str, th.Tensor]
OptimizerStateDict = Dict[str, Any]
MaybeCallback = Union[None, Callable, List["BaseCallback"], "BaseCallback"]
PyTorchObs = Union[th.Tensor, TensorDict]

# A schedule takes the remaining progress as input
# and outputs a scalar (e.g. learning rate, clip range, ...)
Schedule = Callable[[float], float]


class TensorRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    actions_with_grad: th.Tensor
    old_values: th.Tensor
    old_values_with_grad: th.Tensor
    old_log_prob: th.Tensor
    old_log_prob_with_grad: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class TensorRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer using tensors retaining gradients
    """

    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.actions = th.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=th.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        self.log_probs = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super(RolloutBuffer, self).reset()

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """

        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()  # type: ignore[assignment]
        values_np = self.values.clone().detach().cpu().numpy()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = values_np[step + 1]

            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - values_np[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + values_np

    def add(
        self,
        obs: np.ndarray,
        action: th.Tensor,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = action
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().flatten()
        self.log_probs[self.pos] = log_prob.clone()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    # def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
    #     assert self.full, ""
    #     indices = np.random.permutation(self.buffer_size * self.n_envs)
    #     # Prepare the data
    #     if not self.generator_ready:
    #         _tensor_names = [
    #             "observations",
    #             "actions",
    #             "values",
    #             "log_probs",
    #             "advantages",
    #             "returns",
    #         ]

    #         for tensor in _tensor_names:
    #             self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
    #         self.generator_ready = True

    #     # Return everything, don't create minibatches
    #     if batch_size is None:
    #         batch_size = self.buffer_size * self.n_envs

    #     start_idx = 0
    #     while start_idx < self.buffer_size * self.n_envs:
    #         yield self._get_samples(indices[start_idx : start_idx + batch_size])
    #         start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> TensorRolloutBufferSamples:

        data = (
            self.to_torch(self.observations[batch_inds]),
            self.actions[batch_inds].clone().detach(),
            self.actions[batch_inds],
            self.values[batch_inds].clone().detach().flatten(),
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].clone().detach().flatten(),
            self.log_probs[batch_inds].flatten(),
            self.to_torch(self.advantages[batch_inds].flatten()),
            self.to_torch(self.returns[batch_inds].flatten()),
        )
        return TensorRolloutBufferSamples(*tuple(data))
