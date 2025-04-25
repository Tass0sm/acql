from typing import Callable

import mlflow

import jax
import jax.numpy as jnp
# from brax import envs
# from brax.training.types import Params
# from brax.training.types import PRNGKey
from achql.brax.agents.achql import networks as achql_networks

from achql.visualization.critic import plot_function_grid, plot_simple_maze_option_arrows


def make_plots_for_hdqn(
        env,
        network,
        params,
        label: str = "",
        grid_size: int = 100,
        tmp_state_fn = lambda x: x,
        x_min: float = 0.0,
        x_max: float = 16.0,
        y_min: float = 0.0,
        y_max: float = 16.0,
        # save_and_close = True,
        with_dressing = True,
        seed: int = 0,
):
    reset_key = jax.random.PRNGKey(seed)
    tmp_state = env.reset(reset_key)
    tmp_state = tmp_state_fn(tmp_state)

    breakpoint()
    
    normalizer_params, option_q_params = params

    x_values = jnp.linspace(x_min, x_max, grid_size)
    y_values = jnp.linspace(y_min, y_max, grid_size)
    X, Y = jnp.meshgrid(x_values, y_values)
    positions = jnp.stack([X.ravel(), Y.ravel()], axis=1)

    # Prepare batched input for value_net
    obs_batch = jnp.repeat(jnp.expand_dims(tmp_state.obs, axis=0), positions.shape[0], axis=0)
    obs_batch = obs_batch.at[:, 0:2].set(positions)

    # goal_repeated = np.repeat(goal[2:].reshape(1, -1), positions.shape[0], axis=0)
    # batch_input = np.hstack([positions, goal_repeated])
    # batch_goal = np.repeat(goal.reshape(1, -1), positions.shape[0], axis=0)

    # Compute the value function for the grid
    value_q_function_output = network.option_q_network.apply(normalizer_params, option_q_params, obs_batch).min(axis=-1)
    value_function_output = value_q_function_output.max(axis=-1)
    value_function_grid = value_function_output.reshape(grid_size, grid_size)

    options = value_q_function_output.argmax(axis=-1)

    # # Option arrows
    # arrows = jnp.array([[0.0, 1.0],
    #                     [1.0, 0.0],
    #                     [-1.0, 0.0],
    #                     [0.0, -1.0]])
    # option_vectors = jax.vmap(lambda o: arrows.at[o].get())(options)
    # option_vector_grid = option_vectors.reshape(grid_size, grid_size, 2)
    # option_vector_grid = option_vector_grid[::2, ::2]

    start_position = tmp_state.obs[env.pos_indices][:2]
    goal_position = tmp_state.obs[env.goal_indices][:2]

    fig, ax = plot_function_grid(
        X, Y,
        x_min, x_max,
        y_min, y_max,
        value_function_grid,
        start_position, goal_position,
        label,
        with_dressing=with_dressing
    )

    plot_simple_maze_option_arrows([ax], X, Y, options, grid_size)

    return [(fig, ax)]


def make_plots_for_achql(
        env,
        make_policy: Callable,
        network,
        params,
        label: str = "",
        grid_size: int = 100,
        tmp_state_fn = lambda x: x,
        # save_and_close: bool = True,
        x_min: float = 0.0,
        x_max: float = 16.0,
        y_min: float = 0.0,
        y_max: float = 16.0,
        with_dressing = True,
        seed: int = 0,
):
    key = jax.random.PRNGKey(seed)
    reset_key, policy_key = jax.random.split(key)
    tmp_state = env.reset(reset_key)
    tmp_state = tmp_state_fn(tmp_state)

    normalizer_params, option_q_params, cost_q_params = params

    x_values = jnp.linspace(x_min, x_max, grid_size)
    y_values = jnp.linspace(y_min, y_max, grid_size)
    X, Y = jnp.meshgrid(x_values, y_values)
    positions = jnp.stack([X.ravel(), Y.ravel()], axis=1)

    # Prepare batched input for value_net
    obs_batch = jnp.repeat(jnp.expand_dims(tmp_state.obs, axis=0), positions.shape[0], axis=0)
    # obs_batch = jnp.concatenate((env.goalless_obs(obs_batch),
    #                              env.ith_goal(obs_batch, 0)), axis=-1)
    obs_batch = obs_batch.at[:, 0:2].set(positions)

    # qr
    make_qr = achql_networks.make_option_qr_fn(network, env, 0.0)
    qr_fn = make_qr(params)

    # qc
    make_qc = achql_networks.make_option_qc_fn(network, env, 0.0)
    qc_fn = make_qc(params)

    # policy
    policy = make_policy(params, deterministic=True)
    options, _ = policy(obs_batch, policy_key)

    # Compute the value function for the grid
    # value_q_function_output = network.option_q_network.apply(normalizer_params, option_q_params, obs_batch).min(axis=-1)
    value_q_function_output = qr_fn(obs_batch)
    value_function_output = jax.vmap(lambda x, i: x.at[i].get())(value_q_function_output, options)
    value_function_grid = value_function_output.reshape(grid_size, grid_size)

    # Compute the cost value function for the grid
    cost_q_function_output = qc_fn(obs_batch)
    cost_value_function_output = jax.vmap(lambda x, i: x.at[i].get())(cost_q_function_output, options)
    cost_value_function_grid = cost_value_function_output.reshape(grid_size, grid_size)

    start_position = tmp_state.obs[:2]
    goal_position = env.ith_goal(tmp_state.obs, 0)
    fig1, ax1 = plot_function_grid(
        X, Y,
        x_min, x_max,
        y_min, y_max,
        value_function_grid,
        start_position, goal_position,
        f"Reward - {label}",
        with_dressing=with_dressing
    )

    plot_simple_maze_option_arrows([ax1], X, Y, options, grid_size)

    # if save_and_close:
    #     fig1.savefig(f"figures/value_function_{label}.png", format="png", bbox_inches='tight', pad_inches=0)
    #     # fig1.savefig(f"figures/value_function_{label}.pdf", format="pdf", bbox_inches='tight', pad_inches=0)
    #     fig1.savefig(f"figures/value_function_{label}.svg", format="svg", bbox_inches='tight', pad_inches=0)
    #     plt.close(fig1)

    fig2, ax2 = plot_function_grid(
        X, Y,
        x_min, x_max,
        y_min, y_max,
        cost_value_function_grid,
        start_position, goal_position,
        f"Cost - {label}",
        with_dressing=with_dressing,
        vmin=-1.0,
        vmax=1.0,
    )

    plot_simple_maze_option_arrows([ax2], X, Y, options, grid_size)

    # if save_and_close:
    #     fig2.savefig(f"figures/cost_value_function_{label}.png", format="png", bbox_inches='tight', pad_inches=0)
    #     # fig2.savefig(f"figures/cost_value_function_{label}.pdf", format="pdf", bbox_inches='tight', pad_inches=0)
    #     fig2.savefig(f"figures/cost_value_function_{label}.svg", format="svg", bbox_inches='tight', pad_inches=0)
    #     plt.close(fig2)

    return [(fig1, ax1), (fig2, ax2)]


def make_plots_for_ddpg(
        env,
        make_policy: Callable,
        network,
        params,
        label: str = "",
        grid_size: int = 100,
        tmp_state_fn = lambda x: x,
        # save_and_close: bool = True,
        x_min: float = 0.0,
        x_max: float = 16.0,
        y_min: float = 0.0,
        y_max: float = 16.0,
        with_dressing = True,
        seed: int = 0,
):
    key = jax.random.PRNGKey(seed)
    reset_key, policy_key = jax.random.split(key)
    tmp_state = env.reset(reset_key)
    tmp_state = tmp_state_fn(tmp_state)

    x_values = jnp.linspace(x_min, x_max, grid_size)
    y_values = jnp.linspace(y_min, y_max, grid_size)
    X, Y = jnp.meshgrid(x_values, y_values)
    positions = jnp.stack([X.ravel(), Y.ravel()], axis=1)

    # Prepare batched input for value_net
    obs_batch = jnp.repeat(jnp.expand_dims(tmp_state.obs, axis=0), positions.shape[0], axis=0)
    # obs_batch = jnp.concatenate((env.goalless_obs(obs_batch),
    #                              env.ith_goal(obs_batch, 0)), axis=-1)
    obs_batch = obs_batch.at[:, 0:2].set(positions)

    # policy
    policy = make_policy(params, deterministic=True)
    actions, _ = policy(obs_batch, policy_key)

    # Compute the value function for the grid
    normalizer_params, q_params, _ = params
    value_function_output = network.q_network.apply(normalizer_params, q_params, obs_batch, actions).min(axis=-1)
    value_function_grid = value_function_output.reshape(grid_size, grid_size)

    start_position = tmp_state.obs[:2]
    goal_position = tmp_state.obs[-2:]
    fig, ax = plot_function_grid(
        X, Y,
        x_min, x_max,
        y_min, y_max,
        value_function_grid,
        start_position, goal_position,
        f"Reward - {label}",
        with_dressing=with_dressing
    )

    return [(fig, ax)]


