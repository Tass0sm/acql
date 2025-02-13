import functools
import time
from typing import Any, Callable, Optional, Tuple, Union, Sequence

from absl import logging
from brax import base
from brax.envs import wrappers
from brax.io import model
from brax.training import acting
from brax.training import gradients
from brax.training import pmap
from brax.training.replay_buffers_test import jit_wrap
from brax.training import replay_buffers
from brax.training import types
from brax.training.acme import specs
from brax.training.types import Params
from brax.training.types import PRNGKey
import flax
import jax
import jax.numpy as jnp
import optax
import numpy as np

from navix import Timestep


from achql.brax.training.acme import running_statistics
# from achql.brax.her import replay_buffers
from achql.navix.envs.wrappers.training import wrap_for_training

from achql.navix.training.evaluator import NavixEvaluator
from achql.navix.agents.tabular_hql import tables
from achql.navix.training import hierarchical_acting
from achql.hierarchy.training import types as h_types

from achql.navix.envs.wrappers.options_wrapper import OptionsWrapper


Metrics = types.Metrics
Transition = types.Transition
InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]

ReplayBufferState = Any

_PMAP_AXIS_NAME = 'i'


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""
    option_q_params: Params
    update_steps: jnp.ndarray
    env_steps: jnp.ndarray


def _unpmap(v):
    return jax.tree_map(lambda x: x[0], v)


def _init_training_state(
    key: PRNGKey,
    obs_size: int,
    local_devices_to_use: int,
    hql_tables: tables.HQLTables,
) -> TrainingState:
    """Inits the training state and replicates it over devices."""
    key_policy, key_q = jax.random.split(key, 2)
    option_q_params = hql_tables.option_q_table.init(key_q)
  
    training_state = TrainingState(
        option_q_params=option_q_params,
        update_steps=jnp.zeros((), dtype=jnp.int32),
        env_steps=jnp.zeros(()))
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
  

def train(
        environment,
        specification: Callable[..., jnp.ndarray],
        state_var,
        num_timesteps,
        episode_length: int = 1000,
        wrap_env: bool = True,
        # action_repeat: int = 1,
        num_envs: int = 128,
        # num_sampling_per_update: int = 50,
        num_eval_envs: int = 16,
        learning_rate: float = 1e-1,
        discounting: float = 0.9,
        seed: int = 0,
        batch_size: int = 256,
        num_evals: int = 1,
        normalize_observations: bool = True,
        max_devices_per_host: Optional[int] = None,
        reward_scaling: float = 1.,
        tau: float = 0.005,
        min_replay_size: int = 0,
        max_replay_size: Optional[int] = 10_0000,
        updates_per_step: int = 1,
        deterministic_eval: bool = True,
        tensorboard_flag = True,
        logdir = './logs',
        table_factory = tables.make_hql_tables,
        options=[],
        progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
        checkpoint_logdir: Optional[str] = None,
        eval_env = None,
):
    process_id = jax.process_index()
    local_devices_to_use = jax.local_device_count()
    if max_devices_per_host is not None:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    device_count = local_devices_to_use * jax.process_count()
    logging.info('local_device_count: %s; total_device_count: %s', local_devices_to_use, device_count)
  
    if min_replay_size >= num_timesteps:
        raise ValueError(
          'No training will happen because min_replay_size >= num_timesteps')
  
    if max_replay_size is None:
        max_replay_size = num_timesteps
  
    # The number of environment steps executed for every `actor_step()` call.
    env_steps_per_actor_step = num_envs
    # equals to ceil(min_replay_size / env_steps_per_actor_step)
    num_prefill_actor_steps = -(-min_replay_size // num_envs)
    num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
    assert num_timesteps - num_prefill_env_steps >= 0
    num_evals_after_init = max(num_evals - 1, 1)
    # The number of epoch calls per training
    # equals to
    # ceil(num_timesteps - num_prefill_env_steps /
    #      (num_evals_after_init * env_steps_per_actor_step))
    num_training_steps_per_epoch = -(
        -(num_timesteps - num_prefill_env_steps) // (num_evals_after_init * env_steps_per_actor_step)
    )
  
    assert num_envs % device_count == 0
    env = OptionsWrapper(
        environment,
        options,
        discounting=discounting,
    )
    if wrap_env:
        rng = jax.random.PRNGKey(seed)
        rng, key = jax.random.split(rng)
        # v_randomization_fn = None
        # if randomization_fn is not None:
        #   v_randomization_fn = functools.partial(
        #       randomization_fn,
        #       rng=jax.random.split(
        #           key, num_envs // jax.process_count() // local_devices_to_use),
        #   )
        env = wrap_for_training(
            env,
            episode_length=episode_length,
            # action_repeat=action_repeat,
            # randomization_fn=v_randomization_fn,
        )
  
    obs_size = env.observation_space.shape[0]
    
    hql_tables = table_factory(
        observation_space=env.observation_space,
        action_space=env.action_space,
        options=options,
    )
    make_policy = tables.make_option_inference_fn(hql_tables)
    make_flat_policy = tables.make_inference_fn(hql_tables)
  
    dummy_transition = Transition(
        observation=0, # observation is a single integer with an environment compatible with tabular algorithms.
        action=0,
        reward=0.,
        discount=0.,
        next_observation=0,
        extras={
            'state_extras': {
                'truncation': 0.0,
            },
            'policy_extras': {},
        }
    )
  
    replay_buffer = replay_buffers.UniformSamplingQueue(
        max_replay_size=max_replay_size // device_count,
        dummy_data_sample=dummy_transition,
        sample_batch_size=batch_size * updates_per_step // device_count)
  
    option_q_table = hql_tables.option_q_table

    def option_q_table_update(
            option_q_params,
            transition,
    ):
        o_idx = transition.observation
        a_idx = transition.action

        qs_old = option_q_table.apply(option_q_params, o_idx)
        q_old = qs_old[a_idx]
  
        next_qs = option_q_table.apply(option_q_params, transition.next_observation)
        next_v = next_qs.max(axis=-1)
  
        target_q = transition.reward * reward_scaling + transition.discount * discounting * next_v

        # jax.lax.cond(next_v > 0, lambda x: jax.debug.breakpoint(), lambda x: None, (
        #     o_idx,
        #     a_idx,
        #     option_q_params,
        #     transition.next_observation,
        #     transition.reward,
        #     reward_scaling,
        #     transition.discount,
        #     discounting,
        #     next_v,
        # ))

        new_option_q_params = option_q_params.at[o_idx, a_idx].set(q_old + learning_rate * (target_q - q_old))
        return new_option_q_params, None
  

    def update_step(
            carry: Tuple[TrainingState, PRNGKey],
            transitions: Transition) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
        training_state, key = carry
  
        new_option_q_params, _ = jax.lax.scan(
            option_q_table_update,
            training_state.option_q_params,
            transitions)

        metrics = {
            'update_steps': training_state.update_steps,
        }
        
        new_training_state = TrainingState(
            option_q_params=new_option_q_params,
            update_steps=training_state.update_steps + 1,
            env_steps=training_state.env_steps,
        )
        return (new_training_state, key), metrics
  
  
    def get_experience(
            option_q_params: Params,
            env_state: Timestep,
            buffer_state: ReplayBufferState,
            key: PRNGKey,
    ) -> Tuple[
        Timestep,
        ReplayBufferState,
    ]:
        policy = make_policy(option_q_params)
        env_state, transitions = hierarchical_acting.semimdp_actor_step(
            env,
            env_state,
            policy,
            key,
            extra_fields=()
        )
        
        buffer_state = replay_buffer.insert(buffer_state, transitions)
        return env_state, buffer_state
  
    def training_step(
            training_state: TrainingState,
            env_state: Timestep,
            buffer_state: ReplayBufferState,
            key: PRNGKey
    ) -> Tuple[TrainingState, Timestep, ReplayBufferState, Metrics]:
        experience_key, training_key = jax.random.split(key)
        env_state, buffer_state = get_experience(
            training_state.option_q_params,
            env_state,
            buffer_state,
            experience_key
        )
        
        training_state = training_state.replace(
            env_steps=training_state.env_steps + env_steps_per_actor_step
        )
        
        buffer_state, transitions = replay_buffer.sample(buffer_state)

        # Change the front dimension of transitions so 'update_step' is called
        # grad_updates_per_step times by the scan.
        transitions = jax.tree_map(lambda x: jnp.reshape(x, (updates_per_step, -1) + x.shape[1:]), transitions)
        (training_state, _), metrics = jax.lax.scan(update_step, (training_state, training_key), transitions)

        metrics['buffer_current_size'] = replay_buffer.size(buffer_state)
        return training_state, env_state, buffer_state, metrics
  
    def prefill_replay_buffer(
        training_state: TrainingState,
        env_state: Timestep,
        buffer_state: ReplayBufferState,
        key: PRNGKey
    ) -> Tuple[TrainingState, Timestep, ReplayBufferState, PRNGKey]:
        def f(carry, unused):
            del unused
            training_state, env_state, buffer_state, key = carry
            key, new_key = jax.random.split(key)
            env_state, buffer_state = get_experience(
                training_state.option_q_params,
                env_state,
                buffer_state,
                key,
            )
            new_training_state = training_state.replace(
                env_steps=training_state.env_steps + env_steps_per_actor_step,
            )
            return (new_training_state, env_state, buffer_state, new_key), ()
        
        return jax.lax.scan(
            f,
            (training_state, env_state, buffer_state, key),
            (),
            length=num_prefill_actor_steps
        )[0]
  
    prefill_replay_buffer = jax.pmap(prefill_replay_buffer, axis_name=_PMAP_AXIS_NAME)
  
    def additional_updates(
            training_state: TrainingState,
            buffer_state: ReplayBufferState,
            key: PRNGKey,
    ) -> Tuple[TrainingState, ReplayBufferState, Metrics]:
        experience_key, training_key, sampling_key = jax.random.split(key, 3)
        buffer_state, transitions = replay_buffer.sample(buffer_state)
        
        batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
        transitions = jax.vmap(ReplayBufferClass.flatten_crl_fn, in_axes=(None, None, 0, 0))(
            use_her, env, transitions, batch_keys
        )
        
        # Shuffle transitions and reshape them into (number_of_sgd_steps, batch_size, ...)
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"),
            transitions,
        )
        permutation = jax.random.permutation(experience_key, len(transitions.observation))
        transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1, batch_size) + x.shape[1:]),
            transitions,
        )
        
        (training_state, _), metrics = jax.lax.scan(update_step, (training_state, training_key), transitions)
        
        return training_state, buffer_state, metrics
  
    def scan_additional_updates(n, ts, bs, a_update_key):
        def body(carry, unsued_t):
            ts, bs, a_update_key = carry
            new_key, a_update_key = jax.random.split(a_update_key)
            ts, bs, metrics = additional_updates(ts, bs, a_update_key)
            return (ts, bs, new_key), metrics
        
        return jax.lax.scan(body, (ts, bs, a_update_key), (), length=n)
  
    def training_epoch(
            training_state: TrainingState,
            env_state: Timestep,
            buffer_state: ReplayBufferState,
            key: PRNGKey
    ) -> Tuple[TrainingState, Timestep, ReplayBufferState, Metrics]:

        def f(carry, unused_t):
            ts, es, bs, k = carry
            k, new_key = jax.random.split(k)
            ts, es, bs, metrics = training_step(ts, es, bs, k)
            return (ts, es, bs, new_key), metrics

        (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
            f,
            (training_state, env_state, buffer_state, key),
            (),
            length=num_training_steps_per_epoch
        )
        metrics = jax.tree_map(jnp.mean, metrics)
        return training_state, env_state, buffer_state, metrics

    training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)
  
    # Note that this is NOT a pure jittable method.
    def training_epoch_with_timing(
            training_state: TrainingState,
            env_state: Timestep,
            buffer_state: ReplayBufferState,
            key: PRNGKey
    ) -> Tuple[TrainingState, Timestep, ReplayBufferState, Metrics]:
        nonlocal training_walltime
        t = time.time()
        
        (training_state, env_state, buffer_state, metrics) = training_epoch(
            training_state, env_state, buffer_state, key
        )
        metrics = jax.tree_map(jnp.mean, metrics)
        jax.tree_map(lambda x: x.block_until_ready(), metrics)
        
        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        sps = (env_steps_per_actor_step *
               num_training_steps_per_epoch) / epoch_training_time
        metrics = {
            'training/sps': sps,
            'training/walltime': training_walltime,
            **{f'training/{name}': value for name, value in metrics.items()}
        }
        return training_state, env_state, buffer_state, metrics
  
    global_key, local_key = jax.random.split(jax.random.PRNGKey(seed))
    local_key = jax.random.fold_in(local_key, process_id)
  
    # Training state init
    training_state = _init_training_state(
        key=global_key,
        obs_size=obs_size,
        local_devices_to_use=local_devices_to_use,
        hql_tables=hql_tables,
    )
    del global_key
  
    local_key, rb_key, env_key, eval_key = jax.random.split(local_key, 4)
  
    # Env init
    env_keys = jax.random.split(env_key, num_envs // jax.process_count())
    env_keys = jnp.reshape(env_keys, (local_devices_to_use, -1) + env_keys.shape[1:])
    env_state = jax.pmap(env.reset)(env_keys)
  
    # Replay buffer init
    buffer_state = jax.pmap(replay_buffer.init)(jax.random.split(rb_key, local_devices_to_use))
  
    if not eval_env:
        eval_env = env
    else:
        if wrap_env:
            eval_env = OptionsWrapper(
                eval_env,
                options,
                discounting=discounting,
            )
  
        eval_env = wrap_for_training(
            eval_env,
            episode_length=episode_length,
            # action_repeat=action_repeat,
            # randomization_fn=v_randomization_fn,
        )
  
    evaluator = NavixEvaluator(
        eval_env,
        functools.partial(make_policy, deterministic=deterministic_eval),
        options,
        # specification=specification,
        # state_var=state_var,
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        # action_repeat=action_repeat,
        key=eval_key
    )
  
    # Run initial eval
    if process_id == 0 and num_evals > 1:
        metrics = evaluator.run_evaluation(
            _unpmap((training_state.option_q_params)),
            training_metrics={})
        logging.info(metrics)
        progress_fn(
            0,
            metrics,
            env=environment,
            make_policy=make_policy,
            network=hql_tables,
            params=_unpmap((training_state.option_q_params))
      )
  
    # Create and initialize the replay buffer.
    t = time.time()
    prefill_key, local_key = jax.random.split(local_key)
    prefill_keys = jax.random.split(prefill_key, local_devices_to_use)
    training_state, env_state, buffer_state, _ = prefill_replay_buffer(
        training_state, env_state, buffer_state, prefill_keys
    )
  
    replay_size = jnp.sum(jax.vmap(replay_buffer.size)(buffer_state)) * jax.process_count()
    logging.info('replay size after prefill %s', replay_size)
    assert replay_size >= min_replay_size
    training_walltime = time.time() - t
  
    current_step = 0
    for _ in range(num_evals_after_init):
        logging.info('step %s', current_step)
  
        # Optimization
        epoch_key, local_key = jax.random.split(local_key)
        epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
        (training_state, env_state, buffer_state, training_metrics) = training_epoch_with_timing(
            training_state, env_state, buffer_state, epoch_keys
        )
        current_step = int(_unpmap(training_state.env_steps))
  
        # Eval and logging
        if process_id == 0:
            if checkpoint_logdir:
                # Save current policy.
                params = _unpmap((training_state.option_q_params))
                path = f'{checkpoint_logdir}_dq_{current_step}.pkl'
                model.save_params(path, params)
  
            # Run evals.
            metrics = evaluator.run_evaluation(_unpmap((training_state.option_q_params)), training_metrics)
            logging.info(metrics)
            progress_fn(
                current_step,
                metrics,
                env=environment,
                make_policy=make_policy,
                tables=hql_tables,
                params=_unpmap((training_state.option_q_params)),
            )
  
    total_steps = current_step
    assert total_steps >= num_timesteps
  
    params = _unpmap((training_state.option_q_params))
  
    # If there was no mistakes the training_state should still be identical on all
    # devices.
    pmap.assert_is_replicated(training_state)
    logging.info('total steps: %s', total_steps)
    pmap.synchronize_hosts()
    return (make_flat_policy, params, metrics)
