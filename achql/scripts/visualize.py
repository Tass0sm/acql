import mlflow

import jax.numpy as jnp

from achql.visualization.plots import make_plots


mlflow.set_tracking_uri("file:///home/tassos/.local/share/mlflow")
mlflow.set_experiment("proj2-notebook")

def main():
    training_run_id = "428237866d314cd090eea114554ac4c6"

    plots = make_plots(
        training_run_id,
        # label="SumCost",
        # tmp_state_fn=lambda x: x.replace(obs=x.obs.at[4:].set(jnp.array([12., 4.]))),
        # tmp_state_fn=lambda x: x.replace(obs=x.obs.at[4:].set(jnp.array([0., 0., 1.]))),
        # tmp_state_fn=lambda x: x.replace(obs=x.obs.at[4:].set(jnp.array([1., 0., 0.]))),
        # tmp_state_fn=lambda x: x.replace(obs=x.obs.at[4:].set(jnp.array([12., 12., 0., 1.]))),
        # save_and_close=False,
        seed=0,
        label="Initial",
        grid_size=50,
        # x_min = 0.0,
        # x_max = 16.0,
        # y_min = 0.0,
        # y_max = 16.0,
    )

    # plots += make_plots(
    #     training_run_id,
    #     tmp_state_fn=lambda x: x.replace(obs=x.obs.at[4:].set(jnp.array([1., 0., 0.]))),
    #     label="State 0",
    #     grid_size=50,
    # )

    # plots += make_plots(
    #     training_run_id,
    #     tmp_state_fn=lambda x: x.replace(obs=x.obs.at[4:].set(jnp.array([1., 0., 0.]))),
    #     grid_size=50,
    # )

    
    # plot1, plot2 = make_plots_for_achql(
    #         mdp,
    #         make_option_policy,
    #         network,
    #         params,
    #         "SumCost",
    #         # tmp_state_fn=lambda x: x.replace(obs=x.obs.at[4:].set(jnp.array([12., 12., 0., 1., 0., 0.]))),
    #         # tmp_state_fn=lambda x: x.replace(obs=x.obs.at[4:].set(jnp.array([4., 4., 0., 0., 1., 0.]))),
    #         save_and_close=False,
    #         seed=0,
    #         grid_size=50,
    #         x_min = 0.0,
    #         x_max = 16.0,
    #         y_min = 0.0,
    #         y_max = 16.0,
    # )

    # return plot1, plot2
    return plots


if __name__ == "__main__":
    plots = main()

    for plot in plots:
        plot[0].show()
