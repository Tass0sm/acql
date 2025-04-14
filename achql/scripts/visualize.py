import mlflow


from achql.visualization.plots import make_plots


mlflow.set_tracking_uri("file:///home/tassos/.local/share/mlflow")
mlflow.set_experiment("proj2-notebook")

def main():
    training_run_id = "d3c23de2ef7b4f41885e691caa49e824"

    plots = make_plots(
        training_run_id,
        # label="SumCost",
        # tmp_state_fn=lambda x: x.replace(obs=x.obs.at[4:].set(jnp.array([12., 12., 0., 1., 0., 0.]))),
        # tmp_state_fn=lambda x: x.replace(obs=x.obs.at[4:].set(jnp.array([4., 4., 0., 0., 1., 0.]))),
        # save_and_close=False,
        # seed=0,
        # grid_size=50,
        # x_min = 0.0,
        # x_max = 16.0,
        # y_min = 0.0,
        # y_max = 16.0,
    )

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
