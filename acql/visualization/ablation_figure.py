from typing import Callable
import matplotlib.pyplot as plt

from numpy import ma

import jax
import jax.numpy as jnp
from brax import envs
from brax.training.types import Params
from brax.training.types import PRNGKey

from acql.brax.agents.hdqn import networks as hdq_networks
from acql.brax.agents.hdcqn import networks as hdcq_networks
from acql.brax.agents.acql import networks as acql_networks


def fill_ax_with_function_grid(
        ax,
        X, Y,
        x_min, x_max,
        y_min, y_max,
        function_values_grid: jax.Array,
        start_position, goal_position,
        label,
        with_dressing=True,
):

    if not with_dressing:
        ax.set_axis_off()

    ax.set_xticks([])
    ax.set_yticks([])

    im = ax.imshow(function_values_grid, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='plasma', zorder=1)

    # if with_dressing:
    #     plt.colorbar(im, ax=ax, label='Value (Goal-Conditioned)')

    sbs = ax.get_subplotspec().get_topmost_subplotspec()
    # from IPython.core.debugger import Pdb; Pdb().set_trace()
    if (sbs.is_first_row() and not sbs.is_first_col()):
        plt.colorbar(im, ax=ax, ticks=[function_values_grid.min(), 60, function_values_grid.max()])
    else:
        plt.colorbar(im, ax=ax, ticks=[function_values_grid.min(), function_values_grid.max()])

    # Add contour lines
    contour_levels = jnp.linspace(function_values_grid.min(), function_values_grid.max(), 20)  # Adjust levels if needed
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
    # if with_dressing:
    #     ax.set_title(f"Goal-Conditioned Value Function - {label}")
    #     ax.legend(loc='lower left')

    # ax.set_xlabel("X Position")
    # ax.set_ylabel("Y Position")


def fill_axes_for_hdcqn_aut(
        ax1,
        ax2,
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
    reset_key = jax.random.PRNGKey(seed)
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

    make_q = acql_networks.make_option_q_fn(network, env, 0.0)
    q_fn = make_q(params)

    # policy
    policy = make_policy(params, deterministic=True)
    options, _ = policy(obs_batch, None)

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

    fill_ax_with_function_grid(
        ax1,
        X, Y,
        x_min, x_max,
        y_min, y_max,
        value_function_grid,
        start_position, goal_position,
        f"Reward - {label}",
        with_dressing=with_dressing
    )

    # plot_simple_maze_option_arrows([ax1], X, Y, options, grid_size)

    # if save_and_close:
    #     fig1.savefig(f"figures/value_function_{label}.png", format="png", bbox_inches='tight', pad_inches=0)
    #     # fig1.savefig(f"figures/value_function_{label}.pdf", format="pdf", bbox_inches='tight', pad_inches=0)
    #     fig1.savefig(f"figures/value_function_{label}.svg", format="svg", bbox_inches='tight', pad_inches=0)
    #     plt.close(fig1)

    fill_ax_with_function_grid(
        ax2,
        X, Y,
        x_min, x_max,
        y_min, y_max,
        cost_value_function_grid,
        start_position, goal_position,
        f"Cost - {label}",
        with_dressing=with_dressing
    )

    # plot_simple_maze_option_arrows([ax2], X, Y, options, grid_size)

    # if save_and_close:
    #     fig2.savefig(f"figures/cost_value_function_{label}.png", format="png", bbox_inches='tight', pad_inches=0)
    #     # fig2.savefig(f"figures/cost_value_function_{label}.pdf", format="pdf", bbox_inches='tight', pad_inches=0)
    #     fig2.savefig(f"figures/cost_value_function_{label}.svg", format="svg", bbox_inches='tight', pad_inches=0)
    #     plt.close(fig2)

    # return (fig1, ax1), (fig2, ax2)
    # return (fig2, ax2)


def add_headers(
    fig,
    *,
    row_headers=None,
    col_headers=None,
    row_pad=1,
    col_pad=5,
    rotate_row_headers=True,
    **text_kwargs
):
    # Based on https://stackoverflow.com/a/25814386

    axes = fig.get_axes()

    for ax in axes:
        sbs = ax.get_subplotspec()

        # Putting headers on cols
        if (col_headers is not None) and sbs.is_first_row():
            print("Annotating column")
            ax.annotate(
                col_headers[sbs.colspan.start],
                xy=(0.5, 1.025),
                xytext=(0, col_pad),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",
                **text_kwargs,
            )

        # Putting headers on rows
        if (row_headers is not None) and sbs.is_first_col():
            print("Annotating row")
            ax.annotate(
                row_headers[sbs.rowspan.start],
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - row_pad, 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
                rotation=rotate_row_headers * 90,
                **text_kwargs,
            )


def plot_ablation_figure_grid(env, make_policies, networks, paramss, labels, grid_size=20, col_headers=["Min. CMDP (Ours)", "Sum CMDP"], row_headers=["$Q^c$", "$Q^r$"], text_kwargs={}):

    n_columns = len(make_policies)

    # Create a 2xn grid of subplots
    fig, axs = plt.subplots(2, n_columns, squeeze=False)
    
    add_headers(fig, col_headers=col_headers, row_headers=row_headers, **text_kwargs)

    for i, (make_policy, network, params, label) in enumerate(zip(make_policies, networks, paramss, labels)):
        fill_axes_for_hdcqn_aut(
            axs[1, i],
            axs[0, i],
            env,
            make_policy,
            network,
            params,
            label,
            grid_size=grid_size
            # save_and_close: bool = True,
            # seed: int = 0,
            # x_min: float = 0.0,
            # x_max: float = 16.0,
            # y_min: float = 0.0,
            # y_max: float = 16.0,
            # with_dressing=True,
        )

    # Adjust layout and show the plot
    fig.tight_layout()
    fig.show()

    return fig
    # fig.savefig("./figures/ablation_comparison.pdf")
