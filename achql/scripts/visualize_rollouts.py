import jax
import mlflow

import mediapy

from brax.io import model

from achql.visualization.utils import get_mdp_network_policy_and_params
from achql.visualization import hierarchy, flat

# tasks

mlflow.set_tracking_uri("file:///home/tassos/.local/share/mlflow")
mlflow.set_experiment("proj2-notebook")

def main():
    training_run_id = "428237866d314cd090eea114554ac4c6"
    logged_model_path = f'runs:/{training_run_id}/policy_params'
    real_path = mlflow.artifacts.download_artifacts(logged_model_path)
    params = model.load_params(real_path)

    mdp, options, network, make_option_policy, make_policy, params = get_mdp_network_policy_and_params(training_run_id)

    inference_fn = make_policy(params, deterministic=True)
    jit_inference_fn = jax.jit(inference_fn)

    rollout, opt_traj, action = hierarchy.get_rollout(mdp, jit_inference_fn, options, n_steps=300, seed=2, render_every=1)
    # rollout, action = flat.get_rollout(mdp, jit_inference_fn, n_steps=300, seed=0, render_every=1)

    mediapy.write_video('./rollout.mp4',
        mdp.render(
            [s.pipeline_state for s in rollout],
            camera='overview'
        ), fps=1.0 / mdp.dt
    )

    return rollout, opt_traj, action

if __name__ == "__main__":
    rollout, opt_traj, action = main()
