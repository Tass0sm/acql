from gymnasium.envs.registration import register

from brax import envs
from brax.envs.wrappers.gym import GymWrapper, VectorGymWrapper


def brax_env_creator(env_id, is_vec_env=False):

    if is_vec_env:
        Wrapper = GymWrapper
    else:
        Wrapper = VectorGymWrapper

    def f(**kwargs):
        base_env = envs.create(env_id)
        return Wrapper(base_env, **kwargs)
    return f


register(
    id='BraxAnt-v0',
    entry_point=brax_env_creator("ant"),
    vector_entry_point=brax_env_creator("ant", is_vec_env=True),
)
