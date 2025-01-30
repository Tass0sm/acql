import functools
import time
from typing import Any, Callable, Optional, Tuple, Union, Sequence

from absl import logging
from brax import base
from brax import envs
from brax.envs import wrappers
from brax.io import model
from brax.training import acting
from brax.training import gradients
from brax.training import pmap
from brax.training.replay_buffers_test import jit_wrap
from brax.training import types
from brax.training.acme import specs
from brax.training.types import Params
from brax.training.types import PRNGKey
import flax
import jax
import jax.numpy as jnp
import optax
import numpy as np

from achql.brax.training.acme import running_statistics
from achql.brax.her import replay_buffers

from achql.hierarchy.training.evaluator import HierarchicalEvaluatorWithSpecification
from achql.brax.agents.hdcqn_automaton_her import losses as hdcqn_losses
from achql.brax.agents.hdcqn_automaton_her import networks as hdcq_networks
from achql.hierarchy.training import acting as hierarchical_acting
from achql.hierarchy.training import types as h_types

from achql.hierarchy.envs.options_wrapper import OptionsWrapper


Metrics = types.Metrics
Transition = types.Transition
InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]

# ReplayBufferState = Any

# _PMAP_AXIS_NAME = 'i'


# @flax.struct.dataclass
# class TrainingState:
#   """Contains training state for the learner."""
#   option_q_optimizer_state: optax.OptState
#   option_q_params: Params
#   target_option_q_params: Params
#   cost_q_optimizer_state: optax.OptState
#   cost_q_params: Params
#   target_cost_q_params: Params
#   gradient_steps: jnp.ndarray
#   env_steps: jnp.ndarray
#   normalizer_params: running_statistics.RunningStatisticsState


# def _unpmap(v):
#   return jax.tree_map(lambda x: x[0], v)


# def _init_training_state(
#     key: PRNGKey,
#     obs_size: int,
#     local_devices_to_use: int,
#     hdcq_network: hdcq_networks.HDCQNetworks,
#     # policy_optimizer: optax.GradientTransformation,
#     option_q_optimizer: optax.GradientTransformation,
#     cost_q_optimizer: optax.GradientTransformation,
# ) -> TrainingState:
#   """Inits the training state and replicates it over devices."""
#   key_policy, key_q, key_cost_q = jax.random.split(key, 3)
#   option_q_params = hdcq_network.option_q_network.init(key_q)
#   option_q_optimizer_state = option_q_optimizer.init(option_q_params)

#   cost_q_params = hdcq_network.cost_q_network.init(key_cost_q)
#   cost_q_optimizer_state = cost_q_optimizer.init(cost_q_params)

#   normalizer_params = running_statistics.init_state(
#       specs.Array((obs_size,), jnp.float32))

#   training_state = TrainingState(
#       option_q_optimizer_state=option_q_optimizer_state,
#       option_q_params=option_q_params,
#       target_option_q_params=option_q_params,
#       cost_q_optimizer_state=cost_q_optimizer_state,
#       cost_q_params=cost_q_params,
#       target_cost_q_params=cost_q_params,
#       gradient_steps=jnp.zeros((), dtype=jnp.int32),
#       env_steps=jnp.zeros(()),
#       normalizer_params=normalizer_params)
#   return jax.device_put_replicated(training_state, jax.local_devices()[:local_devices_to_use])


# def make_gamma_scheduler(
#     init_value,
#     update_period,
#     decay,
#     end_value,
#     goal_value,
# ):

#   def scheduler(x):
#     # num_decay = jnp.floor_divide(x, update_period)
#     num_decay = jnp.divide(x, update_period)
#     tmp_value = goal_value - (goal_value - init_value) * jnp.pow(decay, num_decay)
#     return jnp.where(tmp_value >= end_value, end_value, tmp_value)

#   return scheduler

def train(
    environment: envs.Env,
    specification: Callable[..., jnp.ndarray],
    state_var,
    num_timesteps,
    episode_length: int = 1000,
    wrap_env: bool = True,
    action_repeat: int = 1,
    unroll_length: int = 50,
    num_envs: int = 128,
    # num_sampling_per_update: int = 50,
    num_eval_envs: int = 16,
    learning_rate: float = 1e-4,
    discounting: float = 0.9,
    seed: int = 0,
    batch_size: int = 256,
    num_evals: int = 1,
    normalize_observations: bool = True,
    max_devices_per_host: Optional[int] = None,
    reward_scaling: float = 1.,
    cost_scaling: float = 1.0,
    safety_minimum: float = 0.0,
    tau: float = 0.005,
    min_replay_size: int = 0,
    max_replay_size: Optional[int] = 10_0000,
    use_her: bool = False,
    grad_updates_per_step: int = 1,
    multiplier_num_sgd_steps: int = 1,
    deterministic_eval: bool = True,
    tensorboard_flag = True,
    logdir = './logs',
    network_factory: types.NetworkFactory[hdcq_networks.HDCQNetworks] = hdcq_networks.make_hdcq_networks,
    options=[],
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    checkpoint_logdir: Optional[str] = None,
    randomization_fn: Optional[
      Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    replay_buffer_class_name = "TrajectoryUniformSamplingQueue",
):

  breakpoint()


  pass


  # process_id = jax.process_index()
  # local_devices_to_use = jax.local_device_count()
  # if max_devices_per_host is not None:
  #   local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
  # device_count = local_devices_to_use * jax.process_count()
  # logging.info('local_device_count: %s; total_device_count: %s', local_devices_to_use, device_count)

  # if min_replay_size >= num_timesteps:
  #   raise ValueError(
  #       'No training will happen because min_replay_size >= num_timesteps')

  # if max_replay_size is None:
  #   max_replay_size = num_timesteps

  # # The number of environment steps executed for every `actor_step()` call.
  # env_steps_per_actor_step = action_repeat * num_envs * unroll_length
  # num_prefill_actor_steps = min_replay_size // unroll_length + 1
  # print("Num_prefill_actor_steps: ", num_prefill_actor_steps)
  # num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
  # assert num_timesteps - min_replay_size >= 0
  # num_evals_after_init = max(num_evals - 1, 1)
  # # The number of epoch calls per training
  # # equals to
  # # ceil(num_timesteps - num_prefill_env_steps /
  # #      (num_evals_after_init * env_steps_per_actor_step))
  # num_training_steps_per_epoch = -(
  #     -(num_timesteps - num_prefill_env_steps) // (num_evals_after_init * env_steps_per_actor_step)
  # )

  # assert num_envs % device_count == 0
  # env = OptionsWrapper(
  #   environment,
  #   options,
  #   discounting=discounting
  # )
  # if wrap_env:
  #   if isinstance(env, envs.Env):
  #     wrap_for_training = envs.training.wrap
  #   else:
  #     wrap_for_training = envs_v1.wrappers.wrap_for_training

  #   rng = jax.random.PRNGKey(seed)
  #   rng, key = jax.random.split(rng)
  #   v_randomization_fn = None
  #   if randomization_fn is not None:
  #     v_randomization_fn = functools.partial(
  #         randomization_fn,
  #         rng=jax.random.split(
  #             key, num_envs // jax.process_count() // local_devices_to_use),
  #     )
  #   env = wrap_for_training(
  #       env,
  #       episode_length=episode_length,
  #       action_repeat=action_repeat,
  #       randomization_fn=v_randomization_fn,
  #   )

  # full_obs_size = env.full_observation_size
  # input_obs_size = env.observation_size
  # cost_input_obs_size = env.cost_observation_size
  # action_size = env.action_size

  # normalize_fn = lambda x, y: x
  # cost_normalize_fn = lambda x, y: x
  # if normalize_observations:
  #   normalize_fn = functools.partial(running_statistics.normalize)
  #   normalization_mask = jnp.concatenate((jnp.ones((env.goalless_observation_size,), dtype=jnp.int32),
  #                                         # jnp.zeros((env.automaton.n_states,), dtype=jnp.int32)
  #                                         ))
  #   cost_normalize_fn = functools.partial(running_statistics.normalize,
  #                                         mask=normalization_mask)

  # hdcq_network = network_factory(
  #     observation_size=input_obs_size,
  #     cost_observation_size=cost_input_obs_size,
  #     num_aut_states=env.automaton.n_states,
  #     action_size=action_size,
  #     options=options,
  #     preprocess_observations_fn=normalize_fn,
  #     preprocess_cost_observations_fn=cost_normalize_fn,
  # )
  # make_policy = hdcq_networks.make_option_inference_fn(hdcq_network, env, safety_minimum)
  # make_flat_policy = hdcq_networks.make_inference_fn(hdcq_network, env, safety_minimum)

  # # policy_optimizer = optax.adam(learning_rate=learning_rate)
  # option_q_optimizer = optax.adam(learning_rate=learning_rate)
  # cost_q_optimizer = optax.adam(learning_rate=learning_rate)

  # dummy_obs = jnp.zeros((full_obs_size,))
  # dummy_transition = Transition(
  #     observation=dummy_obs,
  #     action=0,
  #     reward=0.,
  #     discount=0.,
  #     next_observation=dummy_obs,
  #     extras={
  #         'state_extras': {
  #           'truncation': 0.0,
  #           'seed': 0.0,
  #           'automata_state': 0,
  #           'made_transition': 0,
  #           'cost': 0.0,
  #         },
  #         'policy_extras': {},
  #     }
  # )

  # ReplayBufferClass = getattr(replay_buffers, replay_buffer_class_name)

  # replay_buffer = jit_wrap(
  #   ReplayBufferClass(
  #     max_replay_size=max_replay_size // device_count,
  #     dummy_data_sample=dummy_transition,
  #     sample_batch_size=batch_size // device_count,
  #     num_envs=num_envs,
  #     episode_length=episode_length,
  #     automaton=getattr(env, "automaton", None)
  #   )
  # )

  # # SAFETY GAMMA SCHEDULER

  # if gamma_update_period is None:
  #   gamma_update_period = num_timesteps / 40

  # safety_gamma_scheduler = make_gamma_scheduler(
  #   gamma_init_value,
  #   gamma_update_period,
  #   gamma_decay,
  #   gamma_end_value,
  #   gamma_goal_value,
  # )

  # critic_loss, cost_critic_loss = hdcqn_losses.make_losses(
  #     hdcq_network=hdcq_network,
  #     env=env,
  #     reward_scaling=reward_scaling,
  #     cost_scaling=cost_scaling,
  #     safety_minimum=safety_minimum,
  #     discounting=discounting,
  # )
  # critic_update = gradients.gradient_update_fn(
  #     critic_loss, option_q_optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)
  # cost_critic_update = gradients.gradient_update_fn(
  #     cost_critic_loss, cost_q_optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)

  # def sgd_step(
  #     carry: Tuple[TrainingState, PRNGKey],
  #     transitions: Transition) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
  #   training_state, key = carry

  #   key, key_critic, key_cost_critic = jax.random.split(key, 3)

  #   (critic_loss, critic_info), option_q_params, option_q_optimizer_state = critic_update(
  #       training_state.option_q_params,
  #       training_state.cost_q_params,
  #       training_state.normalizer_params,
  #       training_state.target_option_q_params,
  #       training_state.target_cost_q_params,
  #       transitions,
  #       key_critic,
  #       optimizer_state=training_state.option_q_optimizer_state)

  #   cost_gamma = safety_gamma_scheduler(training_state.gradient_steps)

  #   (cost_critic_loss, cost_critic_info), cost_q_params, cost_q_optimizer_state = cost_critic_update(
  #     training_state.cost_q_params,
  #     training_state.option_q_params,
  #     training_state.normalizer_params,
  #     training_state.target_cost_q_params,
  #     training_state.target_option_q_params,
  #     transitions,
  #     cost_gamma,
  #     key_cost_critic,
  #     optimizer_state=training_state.cost_q_optimizer_state)

  #   new_target_option_q_params = jax.tree_map(lambda x, y: x * (1 - tau) + y * tau,
  #                                      training_state.target_option_q_params, option_q_params)
  #   new_target_cost_q_params = jax.tree_map(lambda x, y: x * (1 - tau) + y * tau,
  #                                      training_state.target_cost_q_params, cost_q_params)

  #   metrics = {
  #       'grad_steps': training_state.gradient_steps,
  #       'cost_gamma': cost_gamma,
  #       'critic_loss': critic_loss,
  #       'cost_critic_loss': cost_critic_loss,
  #       # 'actor_loss': actor_loss,
  #       **critic_info,
  #       **cost_critic_info,
  #   }

  #   new_training_state = TrainingState(
  #       option_q_optimizer_state=option_q_optimizer_state,
  #       option_q_params=option_q_params,
  #       target_option_q_params=new_target_option_q_params,
  #       cost_q_optimizer_state=cost_q_optimizer_state,
  #       cost_q_params=cost_q_params,
  #       target_cost_q_params=new_target_cost_q_params,
  #       gradient_steps=training_state.gradient_steps + 1,
  #       env_steps=training_state.env_steps,
  #       normalizer_params=training_state.normalizer_params)
  #   return (new_training_state, key), metrics


  # def get_experience(
  #     normalizer_params: running_statistics.RunningStatisticsState,
  #     option_q_params: Params,
  #     cost_q_params: Params,
  #     env_state: envs.State,
  #     buffer_state: ReplayBufferState,
  #     key: PRNGKey,
  # ) -> Tuple[
  #   running_statistics.RunningStatisticsState,
  #   envs.State,
  #   ReplayBufferState,
  # ]:
  #   policy = make_policy((normalizer_params, option_q_params, cost_q_params))

  #   @jax.jit
  #   def f(carry, unused_t):
  #     env_state, current_key = carry
  #     current_key, next_key = jax.random.split(current_key)
  #     env_state, transition = hierarchical_acting.semimdp_actor_step(
  #       env,
  #       env_state,
  #       policy,
  #       current_key,
  #       extra_fields=(
  #         'truncation',
  #         'seed',
  #         'automata_state',
  #         'made_transition',
  #         'cost',
  #       ),
  #     )
  #     return (env_state, next_key), transition

  #   (env_state, _), data = jax.lax.scan(f, (env_state, key), (), length=unroll_length)

  #   normalizer_params = running_statistics.update(
  #       normalizer_params,
  #     # TODO: figure out best way to get observation with dim input_obs_size
  #       jax.tree_util.tree_map(
  #           lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
  #       ).observation[..., :input_obs_size],  # so that batch size*unroll_length is the first dimension
  #       pmap_axis_name=_PMAP_AXIS_NAME,
  #   )
  #   buffer_state = replay_buffer.insert(buffer_state, data)
  #   return normalizer_params, env_state, buffer_state

  # def training_step(
  #     training_state: TrainingState,
  #     env_state: envs.State,
  #     buffer_state: ReplayBufferState,
  #     key: PRNGKey
  # ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
  #   experience_key, training_key = jax.random.split(key)
  #   normalizer_params, env_state, buffer_state = get_experience(
  #       training_state.normalizer_params,
  #       training_state.option_q_params,
  #       training_state.cost_q_params,
  #       env_state,
  #       buffer_state,
  #       experience_key
  #   )

  #   training_state = training_state.replace(
  #     normalizer_params=normalizer_params,
  #     env_steps=training_state.env_steps + env_steps_per_actor_step
  #   )

  #   training_state, buffer_state, metrics = additional_sgds(training_state, buffer_state, training_key)
  #   return training_state, env_state, buffer_state, metrics

  # def prefill_replay_buffer(
  #     training_state: TrainingState,
  #     env_state: envs.State,
  #     buffer_state: ReplayBufferState,
  #     key: PRNGKey
  # ) -> Tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey]:
  #   def f(carry, unused):
  #     del unused
  #     training_state, env_state, buffer_state, key = carry
  #     key, new_key = jax.random.split(key)
  #     new_normalizer_params, env_state, buffer_state = get_experience(
  #         training_state.normalizer_params,
  #         training_state.option_q_params,
  #         training_state.cost_q_params,
  #         env_state,
  #         buffer_state,
  #         key,
  #     )
  #     new_training_state = training_state.replace(
  #         normalizer_params=new_normalizer_params,
  #         env_steps=training_state.env_steps + env_steps_per_actor_step,
  #     )
  #     return (new_training_state, env_state, buffer_state, new_key), ()

  #   return jax.lax.scan(
  #     f,
  #     (training_state, env_state, buffer_state, key),
  #     (),
  #     length=num_prefill_actor_steps
  #   )[0]

  # prefill_replay_buffer = jax.pmap(prefill_replay_buffer, axis_name=_PMAP_AXIS_NAME)

  # def additional_sgds(
  #     training_state: TrainingState,
  #     buffer_state: ReplayBufferState,
  #     key: PRNGKey,
  # ) -> Tuple[TrainingState, ReplayBufferState, Metrics]:
  #   experience_key, training_key, sampling_key = jax.random.split(key, 3)
  #   buffer_state, transitions = replay_buffer.sample(buffer_state)

  #   batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
  #   transitions = jax.vmap(ReplayBufferClass.flatten_crl_fn, in_axes=(None, None, 0, 0))(
  #       use_her, env, transitions, batch_keys
  #   )

  #   # Shuffle transitions and reshape them into (number_of_sgd_steps, batch_size, ...)
  #   transitions = jax.tree_util.tree_map(
  #       lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"),
  #       transitions,
  #   )
  #   permutation = jax.random.permutation(experience_key, len(transitions.observation))
  #   transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
  #   transitions = jax.tree_util.tree_map(
  #       lambda x: jnp.reshape(x, (-1, batch_size) + x.shape[1:]),
  #       transitions,
  #   )

  #   (training_state, _), metrics = jax.lax.scan(sgd_step, (training_state, training_key), transitions)

  #   return training_state, buffer_state, metrics

  # def scan_additional_sgds(n, ts, bs, a_sgd_key):
  #   def body(carry, unsued_t):
  #     ts, bs, a_sgd_key = carry
  #     new_key, a_sgd_key = jax.random.split(a_sgd_key)
  #     ts, bs, metrics = additional_sgds(ts, bs, a_sgd_key)
  #     return (ts, bs, new_key), metrics

  #   return jax.lax.scan(body, (ts, bs, a_sgd_key), (), length=n)

  # def training_epoch(
  #     training_state: TrainingState,
  #     env_state: envs.State,
  #     buffer_state: ReplayBufferState,
  #     key: PRNGKey
  # ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:

  #   def f(carry, unused_t):
  #     ts, es, bs, k = carry
  #     k, new_key, a_sgd_key = jax.random.split(k, 3)
  #     ts, es, bs, metrics = training_step(ts, es, bs, k)
  #     (ts, bs, a_sgd_key), _ = scan_additional_sgds(multiplier_num_sgd_steps - 1, ts, bs, a_sgd_key)
  #     return (ts, es, bs, new_key), metrics

  #   (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
  #       f,
  #       (training_state, env_state, buffer_state, key),
  #       (),
  #       length=num_training_steps_per_epoch
  #   )
  #   metrics = jax.tree_map(jnp.mean, metrics)
  #   return training_state, env_state, buffer_state, metrics

  # training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

  # # Note that this is NOT a pure jittable method.
  # def training_epoch_with_timing(
  #     training_state: TrainingState,
  #     env_state: envs.State,
  #     buffer_state: ReplayBufferState,
  #     key: PRNGKey
  # ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
  #   nonlocal training_walltime
  #   t = time.time()

  #   (training_state, env_state, buffer_state, metrics) = training_epoch(
  #     training_state, env_state, buffer_state, key
  #   )
  #   metrics = jax.tree_map(jnp.mean, metrics)
  #   jax.tree_map(lambda x: x.block_until_ready(), metrics)

  #   epoch_training_time = time.time() - t
  #   training_walltime += epoch_training_time
  #   sps = (env_steps_per_actor_step *
  #          num_training_steps_per_epoch) / epoch_training_time
  #   metrics = {
  #       'training/sps': sps,
  #       'training/walltime': training_walltime,
  #       **{f'training/{name}': value for name, value in metrics.items()}
  #   }
  #   return training_state, env_state, buffer_state, metrics

  # global_key, local_key = jax.random.split(jax.random.PRNGKey(seed))
  # local_key = jax.random.fold_in(local_key, process_id)

  # # Training state init
  # training_state = _init_training_state(
  #     key=global_key,
  #     obs_size=input_obs_size,
  #     local_devices_to_use=local_devices_to_use,
  #     hdcq_network=hdcq_network,
  #     # policy_optimizer=policy_optimizer,
  #     option_q_optimizer=option_q_optimizer,
  #     cost_q_optimizer=cost_q_optimizer,
  # )
  # del global_key

  # local_key, rb_key, env_key, eval_key = jax.random.split(local_key, 4)

  # # Env init
  # env_keys = jax.random.split(env_key, num_envs // jax.process_count())
  # env_keys = jnp.reshape(env_keys, (local_devices_to_use, -1) + env_keys.shape[1:])
  # env_state = jax.pmap(env.reset)(env_keys)

  # # Replay buffer init
  # buffer_state = jax.pmap(replay_buffer.init)(jax.random.split(rb_key, local_devices_to_use))

  # evaluator = HierarchicalEvaluatorWithSpecification(
  #     env,
  #     functools.partial(make_policy, deterministic=deterministic_eval),
  #     options,
  #     specification=specification,
  #     state_var=state_var,
  #     num_eval_envs=num_eval_envs,
  #     episode_length=episode_length,
  #     action_repeat=action_repeat,
  #     key=eval_key
  # )

  # # Run initial eval
  # if process_id == 0 and num_evals > 1:
  #   metrics = evaluator.run_evaluation(
  #       _unpmap((training_state.normalizer_params, training_state.option_q_params, training_state.cost_q_params)),
  #       training_metrics={})
  #   logging.info(metrics)
  #   progress_fn(
  #     0,
  #     metrics,
  #     env=environment,
  #     make_policy=make_policy,
  #     network=hdcq_network,
  #     params=_unpmap((training_state.normalizer_params,
  #                     training_state.option_q_params,
  #                     training_state.cost_q_params))
  #   )

  # # Create and initialize the replay buffer.
  # t = time.time()
  # prefill_key, local_key = jax.random.split(local_key)
  # prefill_keys = jax.random.split(prefill_key, local_devices_to_use)
  # training_state, env_state, buffer_state, _ = prefill_replay_buffer(
  #     training_state, env_state, buffer_state, prefill_keys
  # )

  # replay_size = jnp.sum(jax.vmap(replay_buffer.size)(buffer_state)) * jax.process_count()
  # logging.info('replay size after prefill %s', replay_size)
  # assert replay_size >= min_replay_size
  # training_walltime = time.time() - t

  # current_step = 0
  # for _ in range(num_evals_after_init):
  #   logging.info('step %s', current_step)

  #   # Optimization
  #   epoch_key, local_key = jax.random.split(local_key)
  #   epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
  #   (training_state, env_state, buffer_state, training_metrics) = training_epoch_with_timing(
  #     training_state, env_state, buffer_state, epoch_keys
  #   )
  #   current_step = int(_unpmap(training_state.env_steps))

  #   # Eval and logging
  #   if process_id == 0:
  #     if checkpoint_logdir:
  #       # Save current policy.
  #       params = _unpmap((training_state.normalizer_params, training_state.option_q_params, training_state.cost_q_params))
  #       path = f'{checkpoint_logdir}_dq_{current_step}.pkl'
  #       model.save_params(path, params)

  #     # Run evals.
  #     metrics = evaluator.run_evaluation(_unpmap((training_state.normalizer_params, training_state.option_q_params, training_state.cost_q_params)), training_metrics)
  #     logging.info(metrics)
  #     progress_fn(
  #       current_step,
  #       metrics,
  #       env=environment,
  #       make_policy=make_policy,
  #       network=hdcq_network,
  #       params=_unpmap((training_state.normalizer_params,
  #                       training_state.option_q_params,
  #                       training_state.cost_q_params)),
  #     )

  # total_steps = current_step
  # assert total_steps >= num_timesteps

  # params = _unpmap((training_state.normalizer_params, training_state.option_q_params, training_state.cost_q_params))

  # # If there was no mistakes the training_state should still be identical on all
  # # devices.
  # pmap.assert_is_replicated(training_state)
  # logging.info('total steps: %s', total_steps)
  # pmap.synchronize_hosts()
  # return (make_flat_policy, params, metrics)
