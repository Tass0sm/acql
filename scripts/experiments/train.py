import os
import torch
import mlflow
import pickle

from corallab_lib import Gym
from corallab_policies import Policy, Dataset

PROJECT_ROOT = "/home/tassos/phd/research/second-project/task-aware-skill-composition"
TRACKING_DIR = os.path.join(PROJECT_ROOT, "tracking")

if __name__ == '__main__':

    mlflow.set_tracking_uri(f"file://{TRACKING_DIR}")
    mlflow.set_experiment("decision-transformer-training")

    # gym = Gym("AntMaze_UMaze-v4", render_mode=None, backend="gymnasium_robotics")
    # dataset = Dataset("D4RL/antmaze/umaze-v1", backend="minari")

    # gym = Gym("Hopper-v5", backend="gymnasium")
    # dataset = Dataset("hopper-expert-v2", gym=gym, backend="d4rl")

    # gym = Gym("HalfCheetah-v5", backend="gymnasium")
    # dataset = Dataset("halfcheetah-expert-v2", gym=gym, backend="d4rl")

    gym = Gym("Walker2d-v5", backend="gymnasium")
    dataset = Dataset("walker2d-expert-v2", gym=gym, backend="d4rl")

    policy = Policy(
        "DecisionTransformer",
        gym.state_dim,
        gym.action_dim,
        20,
        max_ep_len=1000,
        backend="decision_transformer"
    )

    run = mlflow.start_run()

    mlflow.log_param("gym", gym.name)

    policy.train(gym=gym, dataset=dataset, max_iters=0)

    # with open("state_mean.pkl", "wb") as f:
    #     pickle.dump(policy.state_mean, f)

    # with open("state_std.pkl", "wb") as f:
    #     pickle.dump(policy.state_std, f)

    # # Tracking the trained model
    # mlflow.pytorch.log_model(policy.entity_impl.model, "model")





    with open("policy.pkl", "wb") as f:
        pickle.dump(policy, f)


    mlflow.end_run()
