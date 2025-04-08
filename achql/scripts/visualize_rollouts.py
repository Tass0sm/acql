import jax
import mlflow

import mediapy

from brax.io import model

from achql.visualization.utils import get_mdp_network_policy_and_params
from achql.visualization.hierarchy import get_rollout

# tasks

mlflow.set_tracking_uri("file:///home/tassos/.local/share/mlflow")
mlflow.set_experiment("proj2-notebook")

def main():
    training_run_id = "f363ef1146d14f6c92f486f6f0861f96"
    logged_model_path = f'runs:/{training_run_id}/policy_params'
    real_path = mlflow.artifacts.download_artifacts(logged_model_path)
    params = model.load_params(real_path)

    mdp, options, network, make_option_policy, make_policy, params = get_mdp_network_policy_and_params(training_run_id)

    inference_fn = make_policy(params, deterministic=True)
    jit_inference_fn = jax.jit(inference_fn)

    rollout, opt_traj, action = get_rollout(mdp, jit_inference_fn, options, n_steps=300, seed=2, render_every=1)

    mediapy.write_video('./rollout.mp4',
        mdp.render(
            [s.pipeline_state for s in rollout],
            camera='overview'
        ), fps=1.0 / mdp.dt
    )

if __name__ == "__main__":
    main()
