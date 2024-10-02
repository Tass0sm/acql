import os
import torch
import pickle
import mlflow

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg

from corallab_lib import Gym
from corallab_policies import Policy, Dataset

PROJECT_ROOT = "/home/tassos/phd/research/second-project/task-aware-skill-composition"
TRACKING_DIR = os.path.join(PROJECT_ROOT, "tracking")

if __name__ == '__main__':

    mlflow.set_tracking_uri(f"file://{TRACKING_DIR}")

    gym = Gym("AntMaze_UMaze-v4", render_mode="human", backend="gymnasium_robotics")
    policy = Policy(
        "DecisionTransformer",
        gym.state_dim,
        gym.action_dim,
        20,
        backend="decision_transformer"
    )

    training_run_id = "b1cf7258fa4a4c5fa872b3dda375446d"
    logged_model_path = f'runs:/{training_run_id}/model'
    model = mlflow.pytorch.load_model(logged_model_path)
    policy.policy_impl.model = torch.compile(model)

    scale = 1000
    target_rew = 5000
    mode = "normal"

    with open("state_mean.pkl", "rb") as f:
        policy.state_mean = pickle.load(f)

    with open("state_std.pkl", "rb") as f:
        policy.state_std = pickle.load(f)

    ret, length = evaluate_episode_rtg(
        gym.gym_impl.gym_impl,
        gym.state_dim,
        gym.action_dim,
        model,
        max_ep_len=policy.max_ep_len,
        scale=scale,
        target_return=target_rew/scale,
        mode=mode,
        state_mean=policy.state_mean,
        state_std=policy.state_std,
        device="cuda",
    )


    breakpoint()
