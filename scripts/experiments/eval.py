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

    # gym = Gym("AntMaze_UMaze-v4", render_mode="human", backend="gymnasium_robotics")
    # gym = Gym("Hopper-v5", render_mode="human", backend="gymnasium")
    # gym = Gym("HalfCheetah-v5", render_mode="human", backend="gymnasium")
    gym = Gym("Walker2d-v5", render_mode="human", backend="gymnasium")

    policy = Policy(
        "DecisionTransformer",
        gym.state_dim,
        gym.action_dim,
        20,
        backend="decision_transformer"
    )

    # hopper 1 iteration: b5bbd91002754b959b379c7a832f21fa
    # half cheetah 1 iteration: d53ee7e5f7024956b46c7c77fcd95cc5
    # walker 2d 1 iteration: 1fb4392f93fc47fa897241428c665205
    # walker 2d 2 iteration: 8cc139d943ac4f959eb897416f225dfe
    # walker 2d 5 iteration: b9f0db232d594ee0b92892e043ff4378

    training_run_id = "b9f0db232d594ee0b92892e043ff4378"
    logged_model_path = f'runs:/{training_run_id}/model'
    model = mlflow.pytorch.load_model(logged_model_path)
    # policy.policy_impl.model = torch.compile(model)
    policy.entity_impl.model = model

    scale = 1000
    # target_rew = 3600
    # target_rew = 12000
    target_rew = 5000
    mode = "normal"

    with open("state_mean.pkl", "rb") as f:
        policy.state_mean = pickle.load(f)

    with open("state_std.pkl", "rb") as f:
        policy.state_std = pickle.load(f)

    breakpoint()

    ret, length = evaluate_episode_rtg(
        gym.gym_impl,
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
