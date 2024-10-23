import argparse
import os

import torch
import yaml

import corallab_stl as stl

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from mobrob.rl_control.ppo import PPOCtrl

from task_aware_skill_composition.sb.dscrl_ctrl import DSCRLCtrl
from task_aware_skill_composition.sb.dscrl import DSCRL
from task_aware_skill_composition.utils import DATA_DIR

# for fast inference
torch.set_num_threads(1)

if __name__ == "__main__":
    """
    Train a PPO agent with the given environment name.
    The training logs and intermediate models are saved in DATA_DIR/policies/tmp/{env_name}-ppo.

    @param env_name: The name of the environment to train on.
    @param finetune: Whether to finetune a pretrained policy.
    @param save_freq: The frequency (each save_freq timesteps) which to save the policy.
    """

    policy_name = "dscrl"
    env_name = "turtlebot3"
    save_freq = 1_000_000
    finetune = False

    config = yaml.load(
        open(f"{DATA_DIR}/configs/{env_name}-{policy_name}.yaml", "r"), Loader=yaml.FullLoader
    )

    action = stl.Var("action", dim=2)
    box1 = (torch.tensor([0.8, 0.0]), torch.tensor([1.1, 0.1])) # turning right
    box2 = (torch.tensor([0.0, 0.8]), torch.tensor([0.1, 1.1])) # turning left
    in_box1 = stl.InBox(action, *box1)
    in_box2 = stl.InBox(action, *box2)
    # specification = stl.Always(stl.Or(in_box1, in_box2))
    specification = stl.Or(in_box1, in_box2)
    # specification = specification.to("cuda")

    # controller = PPOCtrl.from_config(config=config)
    controller = DSCRLCtrl.from_config(config=config, specification=specification)

    if finetune:
        controller.dscrl.policy.load_state_dict(
            DSCRL.load(f"{DATA_DIR}/policies/{env_name}-{policy_name}.zip").policy.state_dict(),
        )

        # controller.ppo.policy.load_state_dict(
        #     PPO.load(f"{DATA_DIR}/policies/{env_name}-ppo.zip").policy.state_dict(),
        # )

    temp_dir = f"{DATA_DIR}/policies/tmp/{env_name}-{policy_name}"
    save_callback = CheckpointCallback(
        save_freq=save_freq // config["n_envs"],
        save_path=f"{temp_dir}/models",
        name_prefix=f"timestep",
        verbose=1,
    )


    new_logger = configure(os.path.join(temp_dir, "logs"), ["stdout", "csv", "tensorboard"])
    controller.dscrl.set_logger(new_logger)

    controller.learn(
        total_timesteps=config["total_timesteps"],
        callback=save_callback,
        progress_bar=True,
    )

    os.makedirs(f"{DATA_DIR}/policies", exist_ok=True)
    controller.save_model(f"{DATA_DIR}/policies/{env_name}-{policy_name}.zip")
