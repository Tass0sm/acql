import gymnasium as gym

import task_aware_skill_composition.envs


env = gym.make("BraxAnt-v0")

# base_env = envs.create('pusher')
# env = gym.GymWrapper(base_env)

# breakpoint()

# obs, _ = env.reset()
# for _ in range(1000):
#     action, _ = policy.predict(obs, deterministic=True)
#     obs, r, terminated, _, _ = env.step(action)

#     if terminated:
#         obs, _ = env.reset()

#     cum_reward += r

#     # if not no_gui:
#     #     env.render()

#     # if env_name in ("drone", "turtlebot3"):
#     #     time.sleep(0.005)
