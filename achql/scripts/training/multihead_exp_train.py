import functools
from typing import Sequence

import jax.numpy as jnp
from brax.training import networks

from brax.training import types
from flax import linen

from achql.hierarchy.training import networks as h_networks
from achql.hierarchy.option import Option

from achql.baselines.reward_machines.qrm import train as qrm
from achql.baselines.reward_machines.crm import train as crm

from achql.brax.agents.hdqn.networks import HDQNetworks

from achql.scripts.train import train_for_all, training_run
from achql.brax.utils import make_reward_machine_mdp


def make_network_factory(env):

    def network_factory(
            observation_size: int,
            action_size: int,
            preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
            hidden_layer_sizes: Sequence[int] = (256, 256),
            activation: networks.ActivationFn = linen.relu,
            options: Sequence[Option] = [],
    ) -> HDQNetworks:

        assert len(options) > 0, "Must pass at least one option"

        def preprocess_cond_fn(aut_state_obs: jnp.ndarray) -> int:
            return env.automaton.one_hot_decode(aut_state_obs)

        option_q_network = h_networks.make_multi_headed_option_q_network(
                observation_size - env.automaton.n_states,    # network input size
                env.automaton.n_states,                       # number of heads
                len(options),
                env,
                preprocess_observations_fn=preprocess_observations_fn,
                preprocess_cond_fn=preprocess_cond_fn,
                shared_hidden_layer_sizes=hidden_layer_sizes[:1],
                head_hidden_layer_sizes=hidden_layer_sizes[1:],
                activation=activation
        )

        return HDQNetworks(
            option_q_network=option_q_network,
            options=options,
        )

    return network_factory

def crm_with_multihead_network_train(run, task, seed, spec, reward_shaping=False):
    options = task.get_options()

    env = make_reward_machine_mdp(task, reward_shaping=reward_shaping)
    network_factory = make_network_factory(env)
    train_fn = functools.partial(crm.train, network_factory=network_factory)

    make_inference_fn, params = training_run(
        run.info.run_id,
        env,
        seed,
        train_fn=train_fn,
        hyperparameters=task.crm_hps,
        extras={
            "options": options,
            "specification": spec,
            "state_var": task.obs_var,
        }
    )


def qrm_with_multihead_network_train(run, task, seed, spec, reward_shaping=False):
    options = task.get_options()

    env = make_reward_machine_mdp(task, reward_shaping=reward_shaping)
    network_factory = make_network_factory(env)
    train_fn = functools.partial(qrm.train, network_factory=network_factory)

    make_inference_fn, params = training_run(
        run.info.run_id,
        env,
        seed,
        train_fn=train_fn,
        hyperparameters=task.crm_hps,
        extras={
            "options": options,
            "specification": spec,
            "state_var": task.obs_var,
        }
    )


def main():
    # train_for_all(["SimpleMaze"], ["TwoSubgoals"], crm_with_multihead_network_train, "CRM_WITH_MULTIHEAD", seed_range=(0, 1))
    # train_for_all(["SimpleMaze"], ["SingleSubgoal"], crm_with_multihead_network_train, "CRM_WITH_MULTIHEAD", seed_range=(0, 1))
    # train_for_all(["SimpleMaze"], ["TwoSubgoals"], qrm_with_multihead_network_train, "QRM_WITH_MULTIHEAD", seed_range=(0, 1))
    train_for_all(["SimpleMaze"], ["NotUntilAlwaysSubgoal"], qrm_with_multihead_network_train, "QRM_WITH_MULTIHEAD", seed_range=(0, 1))

            
if __name__ == "__main__":
    main()
