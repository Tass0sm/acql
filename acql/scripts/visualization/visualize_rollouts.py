import jax
import mlflow

import mediapy

from brax.io import model

from acql.baselines.logical_options_framework import lof_rollout

from acql.visualization.utils import get_mdp_network_policy_and_params
from acql.visualization import hierarchy, flat

# tasks

def main():
    mlflow.set_tracking_uri("REDACTED")
    # mlflow.set_experiment("proj2-notebook")

    training_run_id = "cc278d824b614e95a5ef34a88a9f2842"
    logged_model_path = f'runs:/{training_run_id}/policy_params'
    real_path = mlflow.artifacts.download_artifacts(logged_model_path)
    params = model.load_params(real_path)

    run = mlflow.get_run(run_id=training_run_id)

    if (alg := run.data.tags.get("alg", None)) == "LOF":
        mdp, logical_options, _, network, _, make_policy, params = get_mdp_network_policy_and_params(training_run_id)

        inference_fn = make_policy(params, deterministic=True)
        jit_inference_fn = jax.jit(inference_fn)

        rollout, opt_traj, action = lof_rollout.get_rollout(mdp, jit_inference_fn, logical_options, n_steps=500, seed=0, render_every=1)
    else:
        mdp, options, _, _, make_policy, _ = get_mdp_network_policy_and_params(training_run_id)

        # params = (params[0], params[3])
        inference_fn = make_policy(params, deterministic=True)
        jit_inference_fn = jax.jit(inference_fn)

        rollout, opt_traj, action = hierarchy.get_rollout(mdp, jit_inference_fn, options, n_steps=500, seed=2, render_every=1)
        # rollout, action = flat.get_rollout(mdp, jit_inference_fn, n_steps=150, seed=2, render_every=1)

    mediapy.write_video('./rollout.mp4',
        mdp.render(
            [s.pipeline_state for s in rollout],
            camera='overview'
        ), fps=1.0 / mdp.dt
    )

    # return rollout, opt_traj, action, mdp
    return rollout, None, action, mdp


if __name__ == "__main__":
    rollout, opt_traj, action, mdp = main()
    print("final reward", rollout[-1].reward)
