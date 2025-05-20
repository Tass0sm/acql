# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deep Deterministic Policy Gradient training.

See: https://arxiv.org/pdf/1812.05905.pdf
"""

import functools
import logging
import time
from typing import Any, Callable, NamedTuple, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from brax import base, envs
from brax.io import model
from brax.training import gradients, pmap, types
from brax.training.acme import running_statistics, specs
from brax.training.acme.types import NestedArray
from brax.training.replay_buffers_test import jit_wrap
from brax.training.types import Params, Policy, PRNGKey
from brax.v1 import envs as envs_v1
from flax.struct import dataclass

from jaxgcrl.envs.wrappers import TrajectoryIdWrapper
from jaxgcrl.utils.evaluator import Evaluator
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue

from acql.brax.training.evaluator_with_specification import EvaluatorWithSpecification

from acql.brax.agents.acddpg import losses as acddpg_losses
from acql.brax.agents.acddpg import networks


# from absl import logging
# from brax import base
# from brax import envs
# from brax.envs import wrappers
# from brax.io import model
# from brax.training import acting
# from brax.training import gradients
# from brax.training import pmap
# from brax.training.replay_buffers_test import jit_wrap
# from brax.training import types
# from brax.training.acme import specs
# from brax.training.types import Params
# from brax.training.types import PRNGKey
# import flax
# import jax
# import jax.numpy as jnp
# import optax
# import numpy as np

# from jaxgcrl.envs.wrappers import TrajectoryIdWrapper

# from acql.brax.training.acme import running_statistics
# from acql.brax.her import replay_buffers

# from acql.brax.agents.acddpg import losses as acddpg_losses
# from acql.brax.agents.acddpg import networks as acddpg_networks



Metrics = types.Metrics
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]
State = Union[envs.State, envs_v1.State]


class Transition(NamedTuple):
    """Container for a transition."""

    observation: NestedArray
    next_observation: NestedArray
    action: NestedArray
    reward: NestedArray
    discount: NestedArray
    extras: NestedArray = ()  # pytype: disable=annotation-type-mismatch  # jax-ndarray


def actor_step(
    env: Env,
    env_state: State,
    policy: Policy,
    key: PRNGKey,
    extra_fields: Sequence[str] = (),
) -> Tuple[State, Transition]:
    """Collect data."""
    actions, policy_extras = policy(env_state.obs, key)
    nstate = env.step(env_state, actions)
    state_extras = {x: nstate.info[x] for x in extra_fields}
    return nstate, Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=env_state.obs,
        action=actions,
        reward=nstate.reward,
        discount=1 - nstate.done,
        next_observation=nstate.obs,
        extras={"policy_extras": policy_extras, "state_extras": state_extras},
    )


InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]

ReplayBufferState = Any

_PMAP_AXIS_NAME = "i"


@functools.partial(jax.jit, static_argnames=["config", "env"])
def flatten_batch(config, env, transition: Transition, sample_key: PRNGKey) -> Transition:
    if config.use_her:
        # Find truncation indexes if present
        seq_len = transition.observation.shape[0]
        arrangement = jnp.arange(seq_len)
        is_future_mask = jnp.array(arrangement[:, None] < arrangement[None], dtype=jnp.float32)
        single_trajectories = jnp.concatenate(
            [transition.extras["state_extras"]["traj_id"][:, jnp.newaxis].T] * seq_len,
            axis=0,
        )

        # final_step_mask.shape == (seq_len, seq_len)
        final_step_mask = (
            is_future_mask * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len) * 1e-5
        )
        final_step_mask = jnp.logical_and(
            final_step_mask,
            transition.extras["state_extras"]["truncation"][None, :],
        )
        non_zero_columns = jnp.nonzero(final_step_mask, size=seq_len)[1]

        # If final state is not present use original goal (i.e. don't change anything)
        new_goals_idx = jnp.where(non_zero_columns == 0, arrangement, non_zero_columns)
        binary_mask = jnp.logical_and(non_zero_columns, non_zero_columns)

        new_goals = (
            binary_mask[:, None] * transition.observation[new_goals_idx][:, env.pos_indices]
            + jnp.logical_not(binary_mask)[:, None]
            * transition.observation[new_goals_idx][:, env.goal_indices]
        )

        # Transform observation
        new_obs = transition.observation.at[..., env.goal_indices].set(new_goals)

        # Recalculate reward
        dist = jnp.linalg.norm(new_obs[:, env.goal_indices] - new_obs[:, env.pos_indices], axis=1)
        new_reward = jnp.array(dist < env.goal_reach_thresh, dtype=float)

        # Transform next observation
        new_next_obs = transition.next_observation.at[..., env.goal_indices].set(new_goals)

        return transition._replace(
            observation=jnp.squeeze(new_obs),
            next_observation=jnp.squeeze(new_next_obs),
            reward=jnp.squeeze(new_reward),
        )

    return transition


@dataclass
class TrainingState:
    """Contains training state for the learner."""

    policy_optimizer_state: optax.OptState
    policy_params: Params
    q_optimizer_state: optax.OptState
    q_params: Params
    target_q_params: Params
    cost_q_optimizer_state: optax.OptState
    cost_q_params: Params
    target_cost_q_params: Params
    gradient_steps: jnp.ndarray
    env_steps: jnp.ndarray
    lambda_optimizer_state: optax.OptState
    lambda_params: Params
    normalizer_params: running_statistics.RunningStatisticsState


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _init_training_state(
    key: PRNGKey,
    obs_size: int,
    local_devices_to_use: int,
    acddpg_network: networks.ACDDPGNetworks,
    lambda_optimizer: optax.GradientTransformation,
    policy_optimizer: optax.GradientTransformation,
    q_optimizer: optax.GradientTransformation,
    cost_q_optimizer: optax.GradientTransformation,
) -> TrainingState:
    """Inits the training state and replicates it over devices."""
    key_policy, key_q, key_cost_q = jax.random.split(key, 3)

    # Inital value for lambda, the Lagrangian multiplier. Also its optimizer
    lambda_initial = 0.001
    lambda_optimizer_state = lambda_optimizer.init(lambda_initial)

    policy_params = acddpg_network.policy_network.init(key_policy)
    policy_optimizer_state = policy_optimizer.init(policy_params)
    q_params = acddpg_network.q_network.init(key_q)
    q_optimizer_state = q_optimizer.init(q_params)
    cost_q_params = acddpg_network.cost_q_network.init(key_cost_q)
    cost_q_optimizer_state = cost_q_optimizer.init(cost_q_params)

    normalizer_params = running_statistics.init_state(specs.Array((obs_size,), jnp.dtype("float32")))

    training_state = TrainingState(
        policy_optimizer_state=policy_optimizer_state,
        policy_params=policy_params,
        q_optimizer_state=q_optimizer_state,
        q_params=q_params,
        target_q_params=q_params,
        cost_q_optimizer_state=cost_q_optimizer_state,
        cost_q_params=cost_q_params,
        target_cost_q_params=cost_q_params,
        gradient_steps=jnp.zeros((), dtype=jnp.int32),
        env_steps=jnp.zeros((), dtype=jnp.int32),
        lambda_optimizer_state=lambda_optimizer_state,
        lambda_params=lambda_initial,
        normalizer_params=normalizer_params,
    )
    return jax.device_put_replicated(training_state, jax.local_devices()[:local_devices_to_use])


def make_gamma_scheduler(
        init_value,
        update_period,
        decay,
        end_value,
        goal_value,
):

    def scheduler(x):
        # num_decay = jnp.floor_divide(x, update_period)
        num_decay = jnp.divide(x, update_period)
        tmp_value = goal_value - (goal_value - init_value) * jnp.pow(decay, num_decay)
        return jnp.where(tmp_value >= end_value, end_value, tmp_value)

    return scheduler


@dataclass
class ACDDPG:
    """ACDDPG agent."""

    learning_rate: float = 1e-4
    discounting: float = 0.9
    batch_size: int = 256
    normalize_observations: bool = False
    reward_scaling: float = 1.0
    cost_scaling: float = 1.0
    # target update rate
    tau: float = 0.005
    min_replay_size: int = 0
    max_replay_size: Optional[int] = 10000
    deterministic_eval: bool = True
    train_step_multiplier: int = 1
    unroll_length: int = 50
    h_dim: int = 256
    n_hidden: int = 2
    # layer norm
    use_ln: bool = False
    # hindsight experience replay
    use_her: bool = False
    # other
    lambda_update_interval: int = 12
    gamma_init_value = 0.80
    gamma_update_period = None
    gamma_decay = 0.15
    gamma_end_value = 0.98
    gamma_goal_value = 1.0
    safety_threshold: float = 0.0
    network_type: str = "old_default"
    use_sum_cost_critic: bool = False

    def _define_training_functions(
            self,
            lambda_update,
            critic_update,
            cost_critic_update,
            actor_update,
            safety_gamma_scheduler,
            num_prefill_actor_steps,
            make_policy,
            env,
            qr_input_size,
            replay_buffer,
            env_steps_per_actor_step,
            num_training_steps_per_epoch,
    ):

        def lambda_update_helper(
                lambda_params,
                cost_q_params,
                normalizer_params,
                transitions,
                key_lambda,
                optimizer_state,
        ):
            return lambda_update(
                lambda_params,
                cost_q_params,
                normalizer_params,
                transitions,
                key_lambda,
                optimizer_state=optimizer_state,
            )

        def no_lambda_update_helper(
                lambda_params,
                cost_q_params,
                normalizer_params,
                transitions,
                key_lambda,
                optimizer_state,
        ):
            return (0.0, {
                # "cost_q": 0.0,
                "mean_violation": 0.0,
            }), lambda_params, optimizer_state



        def update_step(
            carry: Tuple[TrainingState, PRNGKey], transitions: Transition
        ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
            training_state, key = carry

            key, key_lambda, key_critic, key_cost_critic, key_actor = jax.random.split(key, 5)

            (lambda_loss, lambda_info), lambda_params, lambda_optimizer_state = jax.lax.cond(
                training_state.gradient_steps % self.lambda_update_interval == 0,
                lambda_update_helper,
                no_lambda_update_helper,
                training_state.lambda_params,
                training_state.cost_q_params,
                training_state.normalizer_params,
                transitions,
                key_lambda,
                training_state.lambda_optimizer_state,
            )
            lambda_multiplier = training_state.lambda_params

            (critic_loss, critic_info), q_params, q_optimizer_state = critic_update(
                training_state.q_params,
                training_state.policy_params,
                training_state.normalizer_params,
                training_state.target_q_params,
                training_state.target_cost_q_params,
                transitions,
                key_critic,
                optimizer_state=training_state.q_optimizer_state,
            )

            cost_gamma = safety_gamma_scheduler(training_state.gradient_steps)

            (cost_critic_loss, cost_critic_info), cost_q_params, cost_q_optimizer_state = cost_critic_update(
                training_state.cost_q_params,
                training_state.policy_params,
                training_state.normalizer_params,
                training_state.target_cost_q_params,
                transitions,
                cost_gamma,
                key_cost_critic,
                optimizer_state=training_state.cost_q_optimizer_state,
            )

            (actor_loss, actor_info), policy_params, policy_optimizer_state = actor_update(
                training_state.policy_params,
                training_state.normalizer_params,
                training_state.q_params,
                training_state.cost_q_params,
                lambda_multiplier,
                transitions,
                key_actor,
                optimizer_state=training_state.policy_optimizer_state
            )

            new_target_q_params = jax.tree_util.tree_map(
                lambda x, y: x * (1 - self.tau) + y * self.tau,
                training_state.target_q_params,
                q_params,
            )
            new_target_cost_q_params = jax.tree_util.tree_map(
                lambda x, y: x * (1 - self.tau) + y * self.tau,
                training_state.target_cost_q_params,
                cost_q_params,
            )

            metrics = {
                "critic_loss": critic_loss,
                "actor_loss": actor_loss,
                'cost_critic_loss': cost_critic_loss,
                'actor_loss': actor_loss,
                'lambda': lambda_multiplier,
                # **actor_info,
                **critic_info,
                **cost_critic_info,
                # **lambda_info,
            }

            new_training_state = TrainingState(
                policy_optimizer_state=policy_optimizer_state,
                policy_params=policy_params,
                q_optimizer_state=q_optimizer_state,
                q_params=q_params,
                target_q_params=new_target_q_params,
                cost_q_optimizer_state=cost_q_optimizer_state,
                cost_q_params=cost_q_params,
                target_cost_q_params=new_target_cost_q_params,
                gradient_steps=training_state.gradient_steps + 1,
                env_steps=training_state.env_steps,
                lambda_optimizer_state=lambda_optimizer_state,
                lambda_params=lambda_params,
                normalizer_params=training_state.normalizer_params,
            )
            return (new_training_state, key), metrics

        def get_experience(
            normalizer_params: running_statistics.RunningStatisticsState,
            policy_params: Params,
            env_state: Union[envs.State, envs_v1.State],
            buffer_state: ReplayBufferState,
            key: PRNGKey,
        ) -> Tuple[
            running_statistics.RunningStatisticsState,
            Union[envs.State, envs_v1.State],
            ReplayBufferState,
        ]:
            policy = make_policy((normalizer_params, policy_params))

            @jax.jit
            def f(carry, unused_t):
                env_state, current_key = carry
                current_key, next_key = jax.random.split(current_key)
                env_state, transition = actor_step(
                    env,
                    env_state,
                    policy,
                    current_key,
                    extra_fields=(
                        "truncation",
                        "traj_id",
                        'automata_state',
                        'made_transition',
                        'cost',
                    ),
                )
                return (env_state, next_key), transition

            (env_state, _), data = jax.lax.scan(f, (env_state, key), (), length=self.unroll_length)

            normalizer_params = running_statistics.update(
                normalizer_params,
                # TODO: figure out best way to get observation with dim input_obs_size
                jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
                ).observation[..., :qr_input_size],  # so that batch size*unroll_length is the first dimension
                pmap_axis_name=_PMAP_AXIS_NAME,
            )

            normalizer_params = jax.tree.map(lambda x: x.at[env.goal_indices].set(x[env.pos_indices]) if x.ndim >= 1 else x, normalizer_params)

            buffer_state = replay_buffer.insert(buffer_state, data)
            return normalizer_params, env_state, buffer_state

        def training_step(
            training_state: TrainingState,
            env_state: envs.State,
            buffer_state: ReplayBufferState,
            key: PRNGKey,
        ) -> Tuple[TrainingState, Union[envs.State, envs_v1.State], ReplayBufferState, Metrics]:
            experience_key, training_key = jax.random.split(key)
            normalizer_params, env_state, buffer_state = get_experience(
                training_state.normalizer_params,
                training_state.policy_params,
                env_state,
                buffer_state,
                experience_key,
            )
            training_state = training_state.replace(
                normalizer_params=normalizer_params,
                env_steps=training_state.env_steps + env_steps_per_actor_step,
            )

            training_state, buffer_state, metrics = train_steps(training_state, buffer_state, training_key)
            return training_state, env_state, buffer_state, metrics

        def prefill_replay_buffer(
            training_state: TrainingState,
            env_state: envs.State,
            buffer_state: ReplayBufferState,
            key: PRNGKey,
        ) -> Tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey]:
            def f(carry, unused):
                del unused
                training_state, env_state, buffer_state, key = carry
                key, new_key = jax.random.split(key)
                new_normalizer_params, env_state, buffer_state = get_experience(
                    training_state.normalizer_params,
                    training_state.policy_params,
                    env_state,
                    buffer_state,
                    key,
                )
                new_training_state = training_state.replace(
                    normalizer_params=new_normalizer_params,
                    env_steps=training_state.env_steps + env_steps_per_actor_step,
                )
                return (new_training_state, env_state, buffer_state, new_key), ()

            return jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, key),
                (),
                length=num_prefill_actor_steps,
            )[0]

        prefill_replay_buffer = jax.pmap(prefill_replay_buffer, axis_name=_PMAP_AXIS_NAME)

        def train_steps(
            training_state: TrainingState,
            buffer_state: ReplayBufferState,
            key: PRNGKey,
        ) -> Tuple[TrainingState, ReplayBufferState, Metrics]:
            experience_key, training_key, sampling_key = jax.random.split(key, 3)
            buffer_state, transitions = replay_buffer.sample(buffer_state)

            batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])

            transitions = jax.vmap(flatten_batch, in_axes=(None, None, 0, 0))(
                self, env, transitions, batch_keys
            )

            # Shuffle transitions and reshape them into (num_update_steps, batch_size, ...)
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"),
                transitions,
            )
            permutation = jax.random.permutation(experience_key, len(transitions.observation))
            transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1, self.batch_size) + x.shape[1:]),
                transitions,
            )

            (training_state, _), metrics = jax.lax.scan(
                update_step, (training_state, training_key), transitions
            )
            return training_state, buffer_state, metrics

        def scan_train_steps(n, ts, bs, update_key):
            def body(carry, unsued_t):
                ts, bs, update_key = carry
                new_key, update_key = jax.random.split(update_key)
                ts, bs, metrics = train_steps(ts, bs, update_key)
                return (ts, bs, new_key), metrics

            return jax.lax.scan(body, (ts, bs, update_key), (), length=n)

        def training_epoch(
            training_state: TrainingState,
            env_state: envs.State,
            buffer_state: ReplayBufferState,
            key: PRNGKey,
        ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
            def f(carry, unused_t):
                ts, es, bs, k = carry
                k, new_key, update_key = jax.random.split(k, 3)
                ts, es, bs, metrics = training_step(ts, es, bs, k)
                (ts, bs, update_key), _ = scan_train_steps(self.train_step_multiplier - 1, ts, bs, update_key)
                return (ts, es, bs, new_key), metrics

            (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, key),
                (),
                length=num_training_steps_per_epoch,
            )
            metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            return training_state, env_state, buffer_state, metrics

        training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

        # Note that this is NOT a pure jittable method.
        def training_epoch_with_timing(
            training_state: TrainingState,
            env_state: envs.State,
            buffer_state: ReplayBufferState,
            training_walltime: float,
            key: PRNGKey,
        ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
            t = time.time()
            (training_state, env_state, buffer_state, metrics) = training_epoch(
                training_state, env_state, buffer_state, key
            )
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

            epoch_training_time = time.time() - t
            training_walltime += epoch_training_time
            sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_training_time
            metrics = {
                "training/sps": sps,
                "training/walltime": training_walltime,
                **{f"training/{name}": value for name, value in metrics.items()},
            }
            return (
                training_state,
                env_state,
                buffer_state,
                training_walltime,
                metrics,
            )  # pytype: disable=bad-return-type  # py311-upgrade

        return prefill_replay_buffer, training_epoch_with_timing

    def train_fn(
        self,
        config,
        train_env: Union[envs_v1.Env, envs.Env],
        specification: Callable[..., jnp.ndarray],
        state_var,
        eval_env: Optional[Union[envs_v1.Env, envs.Env]] = None,
        randomization_fn: Optional[
            Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
        ] = None,
        progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    ):
        process_id = jax.process_index()
        local_devices_to_use = jax.local_device_count()
        if config.max_devices_per_host is not None:
            local_devices_to_use = min(local_devices_to_use, config.max_devices_per_host)
        device_count = local_devices_to_use * jax.process_count()
        logging.info(
            "local_device_count: %s; total_device_count: %s",
            local_devices_to_use,
            device_count,
        )
        network_factory: types.NetworkFactory[networks.DDPGNetworks] = networks.make_acddpg_networks

        if self.min_replay_size >= config.total_env_steps:
            raise ValueError("No training will happen because min_replay_size >= total_env_steps")

        if self.max_replay_size is None:
            max_replay_size = config.total_env_steps
        else:
            max_replay_size = self.max_replay_size

        # The number of environment steps executed for every `actor_step()` call.
        env_steps_per_actor_step = config.action_repeat * config.num_envs * self.unroll_length
        num_prefill_actor_steps = self.min_replay_size // self.unroll_length + 1
        logging.info("Num_prefill_actor_steps: %s", num_prefill_actor_steps)
        num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
        assert config.total_env_steps - self.min_replay_size >= 0
        num_evals_after_init = max(config.num_evals - 1, 1)
        # The number of epoch calls per training
        # equals to
        # ceil(config.total_env_steps - num_prefill_env_steps /
        #      (num_evals_after_init * env_steps_per_actor_step))
        num_training_steps_per_epoch = -(
            -(config.total_env_steps - num_prefill_env_steps)
            // (num_evals_after_init * env_steps_per_actor_step)
        )

        assert config.num_envs % device_count == 0
        env = train_env
        if isinstance(env, envs.Env):
            wrap_for_training = envs.training.wrap
        else:
            wrap_for_training = envs_v1.wrappers.wrap_for_training

        rng = jax.random.PRNGKey(config.seed)
        rng, key = jax.random.split(rng)
        v_randomization_fn = None
        if randomization_fn is not None:
            v_randomization_fn = functools.partial(
                randomization_fn,
                rng=jax.random.split(key, config.num_envs // jax.process_count() // local_devices_to_use),
            )
        env = TrajectoryIdWrapper(env)
        env = wrap_for_training(
            env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
            randomization_fn=v_randomization_fn,
        )
        unwrapped_env = train_env

        obs_size = env.observation_size
        qr_input_size = env.qr_nn_input_size
        qc_input_size = env.qc_nn_input_size
        action_size = env.action_size

        def normalize_fn(x, y):
            return x
        def cost_normalize_fn(x, y):
            return x
        if self.normalize_observations:
            normalize_fn = running_statistics.normalize
            cost_normalize_fn = running_statistics.normalize
        acddpg_network = network_factory(
            observation_size=qr_input_size,
            cost_observation_size=qc_input_size,
            num_aut_states=env.automaton.n_states,
            num_unique_safety_conditions=env.n_unique_safety_conds,
            action_size=action_size,
            env=env,
            network_type=self.network_type,
            preprocess_observations_fn=normalize_fn,
            preprocess_cost_observations_fn=cost_normalize_fn,
            layer_norm=self.use_ln,
            hidden_layer_sizes=[self.h_dim] * self.n_hidden,
            use_sum_cost_critic=self.use_sum_cost_critic,
        )

        make_policy = networks.make_inference_fn(acddpg_network)

        lambda_optimizer = optax.adam(learning_rate=1e-4)
        policy_optimizer = optax.adam(learning_rate=self.learning_rate)
        q_optimizer = optax.adam(learning_rate=self.learning_rate)
        cost_q_optimizer = optax.adam(learning_rate=self.learning_rate)

        dummy_obs = jnp.zeros((obs_size,))
        dummy_action = jnp.zeros((action_size,))
        dummy_transition = Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
            observation=dummy_obs,
            next_observation=dummy_obs,
            action=dummy_action,
            reward=0.0,
            discount=0.0,
            extras={
                "state_extras": {
                    "truncation": 0.0,
                    "traj_id": 0.0,
                    'automata_state': 0,
                    'made_transition': 0,
                    'cost': 0.0,
                },
                "policy_extras": {},
            },
        )
        replay_buffer = jit_wrap(
            TrajectoryUniformSamplingQueue(
                max_replay_size=max_replay_size // device_count,
                dummy_data_sample=dummy_transition,
                sample_batch_size=self.batch_size // device_count,
                num_envs=config.num_envs,
                episode_length=config.episode_length,
            )
        )

        safety_gamma_scheduler = make_gamma_scheduler(
            self.gamma_init_value,
            self.gamma_update_period if self.gamma_update_period else config.total_env_steps / 40,
            self.gamma_decay,
            self.gamma_end_value,
            self.gamma_goal_value,
        )

        lambda_loss, critic_loss, cost_critic_loss, actor_loss = acddpg_losses.make_losses(
            acddpg_network=acddpg_network,
            env=env,
            reward_scaling=self.reward_scaling,
            cost_scaling=self.cost_scaling,
            safety_threshold=self.safety_threshold,
            discounting=self.discounting,
            use_sum_cost_critic=self.use_sum_cost_critic,
        )
        lambda_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
            lambda_loss, lambda_optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)
        critic_update = gradients.gradient_update_fn(
            critic_loss, q_optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)
        cost_critic_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
            cost_critic_loss, cost_q_optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)
        actor_update = gradients.gradient_update_fn(
            actor_loss, policy_optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)

        prefill_replay_buffer, training_epoch_with_timing = self._define_training_functions(
            lambda_update,
            critic_update,
            cost_critic_update,
            actor_update,
            safety_gamma_scheduler,
            num_prefill_actor_steps,
            make_policy,
            env,
            qr_input_size,
            replay_buffer,
            env_steps_per_actor_step,
            num_training_steps_per_epoch,
        )

        global_key, local_key = jax.random.split(rng)
        local_key = jax.random.fold_in(local_key, process_id)

        # Training state init
        training_state = _init_training_state(
            key=global_key,
            obs_size=qr_input_size,
            local_devices_to_use=local_devices_to_use,
            acddpg_network=acddpg_network,
            lambda_optimizer=lambda_optimizer,
            policy_optimizer=policy_optimizer,
            q_optimizer=q_optimizer,
            cost_q_optimizer=cost_q_optimizer,
        )
        del global_key

        local_key, rb_key, env_key, eval_key = jax.random.split(local_key, 4)

        # Env init
        env_keys = jax.random.split(env_key, config.num_envs // jax.process_count())
        env_keys = jnp.reshape(env_keys, (local_devices_to_use, -1) + env_keys.shape[1:])
        env_state = jax.pmap(env.reset)(env_keys)

        # Replay buffer init
        buffer_state = jax.pmap(replay_buffer.init)(jax.random.split(rb_key, local_devices_to_use))

        if not eval_env:
            unwrapped_eval_env = unwrapped_env
            eval_env = env
        else:
            unwrapped_eval_env = eval_env
            eval_env = TrajectoryIdWrapper(eval_env)
            eval_env = wrap_for_training(
                eval_env,
                episode_length=config.episode_length,
                action_repeat=config.action_repeat,
                randomization_fn=v_randomization_fn,
            )

        if randomization_fn is not None:
            v_randomization_fn = functools.partial(
                randomization_fn, rng=jax.random.split(eval_key, config.num_eval_envs)
            )

        evaluator = EvaluatorWithSpecification(
            eval_env,
            functools.partial(make_policy, deterministic=self.deterministic_eval),
            specification=specification,
            state_var=state_var,
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
            key=eval_key
        )

        # Run initial eval
        metrics = {}
        if process_id == 0 and config.num_evals > 1:
            metrics = evaluator.run_evaluation(
                _unpmap((training_state.normalizer_params, training_state.policy_params)),
                training_metrics={},
            )
            progress_fn(
                0,
                metrics,
                env=unwrapped_eval_env,
                make_policy=make_policy,
                network=acddpg_network,
                params=_unpmap((training_state.normalizer_params, training_state.q_params, training_state.cost_q_params, training_state.policy_params))
            )

        # Create and initialize the replay buffer.
        t = time.time()
        prefill_key, local_key = jax.random.split(local_key)
        prefill_keys = jax.random.split(prefill_key, local_devices_to_use)
        training_state, env_state, buffer_state, _ = prefill_replay_buffer(
            training_state, env_state, buffer_state, prefill_keys
        )

        replay_size = jnp.sum(jax.vmap(replay_buffer.size)(buffer_state)) * jax.process_count()
        logging.info("replay size after prefill %s", replay_size)
        assert replay_size >= self.min_replay_size
        training_walltime = time.time() - t

        current_step = 0
        for eval_epoch_num in range(num_evals_after_init):
            logging.info("step %s", current_step)

            # Optimization
            epoch_key, local_key = jax.random.split(local_key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
            (training_state, env_state, buffer_state, training_walltime, training_metrics) = training_epoch_with_timing(
                training_state, env_state, buffer_state, training_walltime, epoch_keys
            )
            current_step = int(_unpmap(training_state.env_steps))

            # Eval and logging
            if process_id == 0:
                if config.checkpoint_logdir:
                    # Save current policy.
                    params = _unpmap((training_state.normalizer_params, training_state.policy_params))
                    path = f"{config.checkpoint_logdir}_acddpg_{current_step}.pkl"
                    model.save_params(path, params)

                # Run evals.
                metrics = evaluator.run_evaluation(
                    _unpmap((training_state.normalizer_params, training_state.policy_params)),
                    training_metrics,
                )
                progress_fn(
                    current_step,
                    metrics,
                    env=unwrapped_eval_env,
                    make_policy=make_policy,
                    network=acddpg_network,
                    params=_unpmap((training_state.normalizer_params, training_state.q_params, training_state.cost_q_params, training_state.policy_params)),
                )

        total_steps = current_step
        assert total_steps >= config.total_env_steps

        params = _unpmap((training_state.normalizer_params, training_state.policy_params))

        # If there was no mistakes the training_state should still be identical on all
        # devices.
        pmap.assert_is_replicated(training_state)
        logging.info("total steps: %s", total_steps)
        pmap.synchronize_hosts()
        return make_policy, params, metrics
