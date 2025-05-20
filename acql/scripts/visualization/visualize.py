import mlflow

import jax.numpy as jnp

from acql.visualization.interface import make_plots


mlflow.set_tracking_uri("REDACTED")
mlflow.set_experiment("proj2-notebook")

def main():
    training_run_id = "f1beb21813b64fc9b29329cadae54d9b"

    # goal_params = jnp.array([[8., 8., 4., 12.],
    #                          [8., 8., 8., 12.],
    #                          [8., 8., 12., 12.],
    #                          [8., 8., 12., 8.],
    #                          [8., 8., 12., 4.]])

    plots = make_plots(
        training_run_id,
        label="First Stage",
        # tmp_state_fn=lambda x: x.replace(obs=x.obs.at[4:].set(jnp.array([12., 4.]))),
        # tmp_state_fn=lambda x: x.replace(obs=x.obs.at[4:].set(jnp.array([0., 0., 1.]))),
        # tmp_state_fn=lambda x: x.replace(obs=x.obs.at[-4:].set(jnp.array([0., 0., 1., 0.]))),
        # tmp_state_fn=lambda x: x.replace(obs=x.obs.at[4:8].set(goal_param)),
        # save_and_close=False,
        seed=0,
        # label="Initial",
        grid_size=50,
        # x_min = 0.0,
        # x_max = 16.0,
        # y_min = 0.0,
        # y_max = 16.0,
    )

    # plots += make_plots(
    #     training_run_id,
    #     label="Second Stage",
    #     tmp_state_fn=lambda x: x.replace(obs=x.obs.at[-8:].set(jnp.array([4., 12., 0., 0., 0., 1., 0., 0.]))),
    #     seed=0,
    #     # label="Initial",
    #     grid_size=50,
    # )

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

    for i, plot in enumerate(plots):
        plot[0].savefig(f"./tmp_{i}.png")
