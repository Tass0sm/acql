import mlflow

from brax.io import model

from achql.visualization.utils import get_mdp_network_policy_and_params
from achql.visualization.critic import make_plots_for_achql


mlflow.set_tracking_uri("file:///home/tassos/.local/share/mlflow")
mlflow.set_experiment("proj2-notebook")

def main():
    training_run_id = "eb6d4844d40c44318a74712145b543f5"
    logged_model_path = f'runs:/{training_run_id}/policy_params'
    real_path = mlflow.artifacts.download_artifacts(logged_model_path)
    params = model.load_params(real_path)

    mdp, options, network, make_option_policy, make_policy, params = get_mdp_network_policy_and_params(training_run_id)

    plot1, plot2 = make_plots_for_achql(
            mdp,
            make_option_policy,
            network,
            params,
            "SumCost",
            # tmp_state_fn=lambda x: x.replace(obs=x.obs.at[4:].set(jnp.array([12., 12., 0., 1., 0., 0.]))),
            # tmp_state_fn=lambda x: x.replace(obs=x.obs.at[4:].set(jnp.array([4., 4., 0., 0., 1., 0.]))),
            save_and_close=False,
            seed=0,
            grid_size=50,
            x_min = 0.0,
            x_max = 16.0,
            y_min = 0.0,
            y_max = 16.0,
    )

    return plot1, plot2


if __name__ == "__main__":
    plot1, plot2 = main()
