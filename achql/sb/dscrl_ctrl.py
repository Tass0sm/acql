from .dscrl import DSCRL
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from mobrob.envs.wrapper import get_env

from achql.utils import DATA_DIR

try:
    import tensorboard
except ImportError:
    tensorboard = None


class DSCRLCtrl:
    def __init__(
        self,
        dscrl_kwargs: dict,
        env_name: str,
        time_limit: int,
        n_env: int,
        vec_env_type: str = "dummy",
        enable_gui: bool = False,
        seed: int = 0,
    ) -> None:
        self.dscrl_kwargs = dscrl_kwargs
        self.env_name = env_name
        self.time_limit = time_limit
        self.n_env = n_env

        if vec_env_type == "subproc":
            vec_env_cls = SubprocVecEnv
        elif vec_env_type == "dummy":
            vec_env_cls = DummyVecEnv
        else:
            raise ValueError(f"Unknown vec_env_type: {vec_env_type}")

        vec_env = make_vec_env(
            get_env,
            n_envs=n_env,
            env_kwargs={
                "env_name": env_name,
                "enable_gui": enable_gui,
                "terminate_on_goal": True,
                "time_limit": time_limit,
            },
            vec_env_cls=vec_env_cls,
            seed=seed,
        )

        self.dscrl = DSCRL(
            env=vec_env,
            seed=seed,
            tensorboard_log=(
                f"{DATA_DIR}/policies/tmp/{env_name}-dscrl/tensorboard"
                if tensorboard is not None
                else None
            ),
            **dscrl_kwargs,
        )

    @classmethod
    def from_config(cls, config: dict, specification) -> "DSCRLCtrl":
        dscrl_kwargs = {
            **config["dscrl_kwargs"],
            "specification": specification
        }

        return cls(
            dscrl_kwargs=dscrl_kwargs,
            env_name=config["env_name"],
            time_limit=config["time_limit"],
            n_env=config["n_envs"],
            vec_env_type=config["vec_env_type"],
            enable_gui=config["enable_gui"],
            seed=config["seed"],
        )

    def learn(self, *args, **kwargs) -> None:
        self.dscrl.learn(*args, **kwargs)

    def save_model(self, save_path: str) -> None:
        self.dscrl.save(save_path)
