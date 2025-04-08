from typing import Callable
import matplotlib.pyplot as plt

from numpy import ma

import jax
import jax.numpy as jnp
from brax import envs
from brax.training.types import Params
from brax.training.types import PRNGKey

from achql.brax.agents.hdqn import networks as hdq_networks
from achql.brax.agents.hdcqn import networks as hdcq_networks
from achql.brax.agents.achql import networks as achql_networks


def plot_simple_maze_option_arrows(axs, X, Y, options, grid_size):
    # Option arrows
    arrows = jnp.array([[0.0, 1.0],
                        [1.0, 0.0],
                        [-1.0, 0.0],
                        [0.0, -1.0]])
    option_vectors = jax.vmap(lambda o: arrows.at[o].get())(options)
    option_vector_grid = option_vectors.reshape(grid_size, grid_size, 2)
    option_vector_grid = option_vector_grid[::2, ::2]

    for ax in axs:
        ax.quiver(X[::2, ::2], Y[::2, ::2], option_vector_grid[..., 0], option_vector_grid[..., 1])


def plot_3d_env_option_arrows(axs, X, Y, options, grid_size):
    # Option arrows

    arrows = jnp.array([[0.0, 0.0],   # , 1.0],
                        [0.0, 1.0],   # , 0.0],
                        [1.0, 0.0],   # , 0.0],
                        [-1.0, 0.0],   # , 0.0],
                        [0.0, -1.0],   # , 0.0],
                        [0.0, 0.0]])   # , -1.0]])

    option_vectors = jax.vmap(lambda o: arrows.at[o].get())(options)
    option_vector_grid = option_vectors.reshape(grid_size, grid_size, 2)
    option_vector_grid = option_vector_grid[::2, ::2]

    for ax in axs:
        ax.quiver(X[::2, ::2], Y[::2, ::2], option_vector_grid[..., 0], option_vector_grid[..., 1])


def plot_function_grid(
        X, Y,
        x_min, x_max,
        y_min, y_max,
        function_values_grid: jax.Array,
        start_position, goal_position,
        label,
        with_dressing=True,
):

    fig, ax = plt.subplots(figsize=(8, 8), frameon=with_dressing)

    if not with_dressing:
        ax.set_axis_off()

    im = ax.imshow(function_values_grid, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='plasma', zorder=1)

    if with_dressing:
        plt.colorbar(im, ax=ax, label='Value (Goal-Conditioned)')

    # Add contour lines
    contour_levels = jnp.linspace(function_values_grid.min(), function_values_grid.max(), 50)  # Adjust levels if needed
    ax.contour(X, Y, function_values_grid, levels=contour_levels, colors='black', linewidths=0.5, zorder=2)

    # # Overlay the maze structure (with higher zorder)
    # draw(env, ax)
    # for patch in ax.patches:
    #     patch.set_zorder(3)  # Ensure maze patches have the highest z-order

    # Mark start position
    # ax.plot(start_position[0], start_position[1], 'ro', markersize=10, label='Start', zorder=4)

    # # Mark goal position
    # subgoal_position = jnp.array([12.0, 4.0])
    # ax.plot(subgoal_position[0], subgoal_position[1], 'bo', markersize=10, label='Subgoal', zorder=4)

    # Mark goal position
    # ax.plot(goal_position[0], goal_position[1], 'go', markersize=10, label='Goal', zorder=4)

    # Plot Details
    if with_dressing:
        ax.set_title(f"Goal-Conditioned Value Function - {label}")
        # ax.legend(loc='lower left')

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    return fig, ax
    

def make_plots_for_hdqn(
        env: envs.Env,
        network: hdq_networks.HDQNetworks,
        params: Params,
        label: str,
        grid_size=100,
        tmp_state_fn = lambda x: x,
        save_and_close = True,
        seed=0,
        with_dressing=True,
):
    reset_key = jax.random.PRNGKey(seed)
    tmp_state = env.reset(reset_key)
    tmp_state = tmp_state_fn(tmp_state)

    normalizer_params, option_q_params = params

    x_min, x_max = 0, 16
    y_min, y_max = 0, 16
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

    if save_and_close:
        fig.savefig(f"hdqn_figures/value_function_{label}.png", format="png", bbox_inches='tight', pad_inches=0)
        plt.close(fig)



def make_test_plot_for_hdqn(
        env: envs.Env,
        network: hdq_networks.HDQNetworks,
        params: Params,
        label: str,
        grid_size=100,
        tmp_state_fn = lambda x: x,
        save_and_close = True,
        seed=0
):
    reset_key = jax.random.PRNGKey(seed)
    tmp_state1 = env.reset(reset_key)
    tmp_state1 = tmp_state1.replace(obs=tmp_state1.obs.at[:6].get())

    suffix = jnp.array([4.0, 12.0])
    tmp_state2 = tmp_state1.replace(obs=tmp_state1.obs.at[:6].get())
    tmp_state2 = tmp_state2.replace(obs=tmp_state2.obs.at[4:].set(suffix))

    normalizer_params, option_q_params = params

    x_min, x_max = 0, 16
    y_min, y_max = 0, 16
    x_values = jnp.linspace(x_min, x_max, grid_size)
    y_values = jnp.linspace(y_min, y_max, grid_size)
    X, Y = jnp.meshgrid(x_values, y_values)
    positions = jnp.stack([X.ravel(), Y.ravel()], axis=1)

    # Prepare batched input for value_net
    obs_batch1 = jnp.repeat(jnp.expand_dims(tmp_state1.obs, axis=0), positions.shape[0], axis=0)
    obs_batch1 = obs_batch1.at[:, 0:2].set(positions)

    obs_batch2 = jnp.repeat(jnp.expand_dims(tmp_state2.obs, axis=0), positions.shape[0], axis=0)
    obs_batch2 = obs_batch2.at[:, 0:2].set(positions)

    # goal_repeated = np.repeat(goal[2:].reshape(1, -1), positions.shape[0], axis=0)
    # batch_input = np.hstack([positions, goal_repeated])
    # batch_goal = np.repeat(goal.reshape(1, -1), positions.shape[0], axis=0)
    
    # Compute the value function for the grid
    value_q_function_output1 = network.option_q_network.apply(normalizer_params, option_q_params, obs_batch1)
    value_q_function_output2 = network.option_q_network.apply(normalizer_params, option_q_params, obs_batch2)

    value_q_function_output = jnp.max(jnp.stack((value_q_function_output1, value_q_function_output2), axis=-1), axis=-1).min(axis=-1)
    value_function_output = value_q_function_output.max(axis=-1)
    value_function_grid = value_function_output.reshape(grid_size, grid_size)

    options = value_q_function_output.argmax(axis=-1)

    start_position = tmp_state1.obs[:2]
    goal_position = tmp_state1.obs[4:6]
    # goal_position = jnp.array([12.0, 4.0])
    fig, ax = plot_function_grid(
        X, Y,
        x_min, x_max,
        y_min, y_max,
        value_function_grid,
        start_position, goal_position,
        label
    )

    # ax.quiver(X[::2, ::2], Y[::2, ::2], option_vector_grid[..., 0], option_vector_grid[..., 1])

    if save_and_close:
        fig.savefig(f"hdqn_figures/value_function_{label}.png", format="png", bbox_inches='tight', pad_inches=0)
        plt.close(fig)


def make_plots_for_hdcqn(
        env: envs.Env,
        make_policy: Callable,
        network: hdq_networks.HDQNetworks,
        params: Params,
        label: str,
        grid_size=100,
        tmp_state_fn = lambda x: x,
        save_and_close = True,
        seed=0
):
    key = jax.random.PRNGKey(seed)
    reset_key, policy_key = jax.random.split(key)
    tmp_state = env.reset(reset_key)
    tmp_state = tmp_state_fn(tmp_state)

    normalizer_params, option_q_params, cost_q_params = params

    x_min, x_max = 0, 16
    y_min, y_max = 0, 16
    x_values = jnp.linspace(x_min, x_max, grid_size)
    y_values = jnp.linspace(y_min, y_max, grid_size)
    X, Y = jnp.meshgrid(x_values, y_values)
    positions = jnp.stack([X.ravel(), Y.ravel()], axis=1)

    # Prepare batched input for value_net
    obs_batch = jnp.repeat(jnp.expand_dims(tmp_state.obs, axis=0), positions.shape[0], axis=0)
    # obs_batch = jnp.concatenate((env.goalless_obs(obs_batch),
    #                              env.ith_goal(obs_batch, 0)), axis=-1)
    obs_batch = obs_batch.at[:, 0:2].set(positions)

    make_q = hdcq_networks.make_option_q_fn(network, env, 0.0)
    q_fn = make_q(params)

    # policy
    policy = make_policy(params, deterministic=True)
    options, _ = policy(obs_batch, policy_key)

    # Compute the value function for the grid
    # value_q_function_output = network.option_q_network.apply(normalizer_params, option_q_params, obs_batch).min(axis=-1)
    value_q_function_output = q_fn(obs_batch)
    value_function_output = jax.vmap(lambda x, i: x.at[i].get())(value_q_function_output, options)
    value_function_grid = value_function_output.reshape(grid_size, grid_size)

    # Compute the cost value function for the grid
    cost_q_function_output = network.cost_q_network.apply(normalizer_params, cost_q_params, obs_batch).min(axis=-1)
    cost_value_function_output = jax.vmap(lambda x, i: x.at[i].get())(cost_q_function_output, options)
    cost_value_function_grid = cost_value_function_output.reshape(grid_size, grid_size)

    start_position = tmp_state.obs[:2]
    goal_position = jnp.zeros_like(start_position)
    fig1, ax1 = plot_function_grid(
        X, Y,
        x_min, x_max,
        y_min, y_max,
        value_function_grid,
        start_position, goal_position,
        f"Reward - {label}"
    )

    plot_simple_maze_option_arrows([ax1], X, Y, options, grid_size)

    if save_and_close:
        fig1.savefig(f"figures/value_function_{label}.png", format="png", bbox_inches='tight', pad_inches=0)
        # fig1.savefig(f"figures/value_function_{label}.pdf", format="pdf", bbox_inches='tight', pad_inches=0)
        plt.close(fig1)

    fig2, ax2 = plot_function_grid(
        X, Y,
        x_min, x_max,
        y_min, y_max,
        cost_value_function_grid,
        start_position, goal_position,
        f"Cost - {label}"
    )

    plot_simple_maze_option_arrows([ax2], X, Y, options, grid_size)

    if save_and_close:
        fig2.savefig(f"figures/cost_value_function_{label}.png", format="png", bbox_inches='tight', pad_inches=0)
        # fig2.savefig(f"figures/cost_value_function_{label}.pdf", format="pdf", bbox_inches='tight', pad_inches=0)
        plt.close(fig2)

    # return (fig1, ax1), (fig2, ax2)
    return (fig2, ax2)


def make_plots_for_3d_hdcqn_aut(
        env: envs.Env,
        make_policy: Callable,
        network: hdq_networks.HDQNetworks,
        params: Params,
        label: str,
        grid_size=100,
        tmp_state_fn = lambda x: x,
        save_and_close: bool = True,
        seed: int = 0,
        x_min: float = 0.0,
        x_max: float = 16.0,
        y_min: float = 0.0,
        y_max: float = 16.0,
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

    make_q = hdcq_aut_networks.make_option_q_fn(network, env, 0.0)
    q_fn = make_q(params)

    # policy
    policy = make_policy(params, deterministic=True)
    options, _ = policy(obs_batch, policy_key)

    # Compute the value function for the grid
    # value_q_function_output = network.option_q_network.apply(normalizer_params, option_q_params, obs_batch).min(axis=-1)
    value_q_function_output = q_fn(obs_batch)
    value_function_output = jax.vmap(lambda x, i: x.at[i].get())(value_q_function_output, options)
    value_function_grid = value_function_output.reshape(grid_size, grid_size)

    # Compute the cost value function for the grid
    trimmed_normalizer_params = jax.tree.map(lambda x: x[..., :env.cost_observation_size] if x.ndim >= 1 else x, normalizer_params)
    cost_obs_batch = env.cost_obs(obs_batch)
    cost_q_function_output = network.cost_q_network.apply(trimmed_normalizer_params, cost_q_params, cost_obs_batch).min(axis=-1)
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
        f"Reward - {label}"
    )

    plot_3d_env_option_arrows([ax1], X, Y, options, grid_size)

    if save_and_close:
        fig1.savefig(f"figures/value_function_{label}.png", format="png", bbox_inches='tight', pad_inches=0)
        # fig1.savefig(f"figures/value_function_{label}.pdf", format="pdf", bbox_inches='tight', pad_inches=0)
        plt.close(fig1)

    fig2, ax2 = plot_function_grid(
        X, Y,
        x_min, x_max,
        y_min, y_max,
        cost_value_function_grid,
        start_position, goal_position,
        f"Cost - {label}"
    )

    plot_3d_env_option_arrows([ax2], X, Y, options, grid_size)

    if save_and_close:
        fig2.savefig(f"figures/cost_value_function_{label}.png", format="png", bbox_inches='tight', pad_inches=0)
        # fig2.savefig(f"figures/cost_value_function_{label}.pdf", format="pdf", bbox_inches='tight', pad_inches=0)
        plt.close(fig2)

    # return (fig1, ax1), (fig2, ax2)
    return (fig2, ax2)


def make_plots_for_achql(
        env: envs.Env,
        make_policy: Callable,
        network: hdq_networks.HDQNetworks,
        params: Params,
        label: str,
        grid_size=100,
        tmp_state_fn = lambda x: x,
        save_and_close: bool = True,
        seed: int = 0,
        x_min: float = 0.0,
        x_max: float = 16.0,
        y_min: float = 0.0,
        y_max: float = 16.0,
        with_dressing=True,
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

    if save_and_close:
        fig1.savefig(f"figures/value_function_{label}.png", format="png", bbox_inches='tight', pad_inches=0)
        # fig1.savefig(f"figures/value_function_{label}.pdf", format="pdf", bbox_inches='tight', pad_inches=0)
        fig1.savefig(f"figures/value_function_{label}.svg", format="svg", bbox_inches='tight', pad_inches=0)
        plt.close(fig1)

    fig2, ax2 = plot_function_grid(
        X, Y,
        x_min, x_max,
        y_min, y_max,
        cost_value_function_grid,
        start_position, goal_position,
        f"Cost - {label}",
        with_dressing=with_dressing
    )

    plot_simple_maze_option_arrows([ax2], X, Y, options, grid_size)

    if save_and_close:
        fig2.savefig(f"figures/cost_value_function_{label}.png", format="png", bbox_inches='tight', pad_inches=0)
        # fig2.savefig(f"figures/cost_value_function_{label}.pdf", format="pdf", bbox_inches='tight', pad_inches=0)
        fig2.savefig(f"figures/cost_value_function_{label}.svg", format="svg", bbox_inches='tight', pad_inches=0)
        plt.close(fig2)

    return (fig1, ax1), (fig2, ax2)
    # return (fig2, ax2)


def make_plots_for_forbidden_actions(
        env: envs.Env,
        safety_minimum: float,
        make_policy: Callable,
        network: hdcq_networks.HDCQNetworks,
        params: Params,
        label: str,
        grid_size=100,
        tmp_state_fn = lambda x: x,
        save_and_close = True,
        seed=0
):
    reset_key = jax.random.PRNGKey(seed)
    tmp_state = env.reset(reset_key)
    tmp_state = tmp_state_fn(tmp_state)

    normalizer_params, option_q_params, cost_q_params = params

    x_min, x_max = 0, 16
    y_min, y_max = 0, 16
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
    options, _ = policy(obs_batch, None)

    # Compute the cost value function for the grid
    trimmed_normalizer_params = jax.tree.map(lambda x: x[..., :env.cost_observation_size] if x.ndim >= 1 else x, normalizer_params)
    cost_obs_batch = env.cost_obs(obs_batch)
    cost_q_function_output = network.cost_q_network.apply(trimmed_normalizer_params, cost_q_params, cost_obs_batch).min(axis=-1)
    cost_q_function_output = jax.vmap(lambda x, i: x.at[i].get())(cost_q_function_output, options)
    cost_q_function_output_grid = cost_q_function_output.reshape(grid_size, grid_size)
    is_forbidden_grid = cost_q_function_output_grid <= safety_minimum

    # Option arrows
    arrows = jnp.array([[0.0, 1.0],
                        [1.0, 0.0],
                        [-1.0, 0.0],
                        [0.0, -1.0]])
    option_vectors = jax.vmap(lambda o: arrows.at[o].get())(options)
    option_vector_grid = option_vectors.reshape(grid_size, grid_size, 2)

    # make plot

    start_position = tmp_state.obs[:2]
    goal_position = env.ith_goal(tmp_state.obs, 0)
    fig1, ax1 = plot_function_grid(
        X, Y,
        x_min, x_max,
        y_min, y_max,
        cost_q_function_output_grid,
        start_position, goal_position,
        f"Reward - {label}"
    )

    masked_option_vector_grid = ma.masked_array(option_vector_grid, mask=jnp.stack((is_forbidden_grid,
                                                                                    is_forbidden_grid), axis=-1))
    masked_option_vector_grid = masked_option_vector_grid[::2, ::2]
    ax1.quiver(X[::2, ::2], Y[::2, ::2], masked_option_vector_grid[..., 0], masked_option_vector_grid[..., 1])

    if save_and_close:
        fig1.savefig(f"figures/masked_options.png", format="png", bbox_inches='tight', pad_inches=0)
        plt.close(fig1)

    return (fig1, ax1)
