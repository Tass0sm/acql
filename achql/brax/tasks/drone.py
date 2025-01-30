import random

import numpy as np
import jax.numpy as jnp

from jaxgcrl.envs.simple_maze import SimpleMaze
from achql.brax.envs.base import GoalConditionedEnv
from achql.brax.tasks.base import TaskBase
from achql.brax.tasks.templates import sequence, inside_circle, outside_circle, inside_box

from achql.stl import Expression, Var
import achql.stl.expression_jax2 as stl

import random

import numpy as np

# from schrl.envs.wrapper import DronePIDEnv, GoalEnv
# from schrl.tasks.base import TaskBase
# from schrl.tltl.spec import DiffTLTLSpec
# from schrl.tltl.template import sequence, coverage, signal, dim_lb, end_state, loop


class DroneTaskBase(TaskBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create_vars(self):
        self.wp_var = Var("wp", idx=0, dim=3)
        self.obs_var = Var("obs", idx=0, dim=self.env.observation_size, position=(0, 3))

    @property
    def hdqn_hps(self):
        return {
            "num_timesteps": 10_000_000,
            "num_evals": 20,
            "reward_scaling": 30,
            "episode_length": 1000,
            "normalize_observations": True,
            "action_repeat": 1,
            "discounting": 0.997,
            "learning_rate": 6e-4,
            "num_envs": 128,
            "batch_size": 512,
            "grad_updates_per_step": 64,
            "max_devices_per_host": 1,
            "max_replay_size": 1048576,
            "min_replay_size": 8192,
        }

    @property
    def hdqn_her_hps(self):
        return {
            "num_timesteps": 10_000_000,
            "reward_scaling": 1,
            "num_evals": 50,
            "episode_length": 1000,
            "normalize_observations": False,
            "action_repeat": 1,
            "discounting": 0.99,
            # "learning_rate": 3e-4,
            "num_envs": 256,
            "batch_size": 256,
            "unroll_length": 62,
            "multiplier_num_sgd_steps": 1,
            "max_devices_per_host": 1,
            "max_replay_size": 10000,
            # 8192, the default, causes the error "TypeError: broadcast_in_dim shape must have every element be nonnegative, got (-2, 50)."
            "min_replay_size": 1000,
            "use_her": True,
        }

    @property
    def hdcqn_her_hps(self):
        return {
            "num_timesteps": 10_000_000,
            "reward_scaling": 1,
            "cost_scaling": 1,
            "cost_budget": 3.0,
            "num_evals": 50,
            "episode_length": 1000,
            "normalize_observations": True,
            "action_repeat": 1,
            "discounting": 0.99,
            # "learning_rate": 3e-4,
            "num_envs": 256,
            "batch_size": 256,
            "unroll_length": 62,
            "multiplier_num_sgd_steps": 1,
            "max_devices_per_host": 1,
            "max_replay_size": 10000,
            # 8192, the default, causes the error "TypeError: broadcast_in_dim shape must have every element be nonnegative, got (-2, 50)."
            "min_replay_size": 1000,
            "use_her": True,
        }


class DroneNav(SimpleMazeTaskBase):
    def __init__(self, backend="mjx"):
        super().__init__(None, 1000, backend=backend)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = SimpleMaze(
            terminate_when_unhealthy=False,
            backend=backend
        )
        return env

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        true = stl.STLPredicate(obs_var, lambda s: 999, lower_bound=0.0)
        return true


# class DroneSequence(TaskBase):
#     def __init__(self, enable_gui: bool = True):
#         self.goals = np.array([
#             [2, 2, 1],
#             [1, -2, 2],
#             [-2, -1, 2],
#             [0, 2, 1]
#         ])
#         self.seq_time = [
#             [1, 2],
#             [3, 4],
#             [5, 6],
#             [7, 8]
#         ]

#         super(DroneSequence, self).__init__(8, 2000, enable_gui)

#     def _build_env(self, enable_gui) -> GoalEnv:
#         env = DronePIDEnv(gui=enable_gui)

#         self.duck_size = 20
#         env.env_editor.add_duck(pos=np.array([0, 0, 0.099 * self.duck_size * 0.1]), size=self.duck_size)

#         for goal in self.goals:
#             env.env_editor.add_ball(goal, color=(0, 0, 1, 0.5))
#         return env

#     def _build_spec(self) -> DiffTLTLSpec:
#         goal_spec = sequence(self.goals, close_radius=0.5, seq_time=self.seq_time)
#         ground_spec = dim_lb(0.1, dim=2)

#         spec = goal_spec & ground_spec
#         return spec

#     def demo(self, n_trajs: int) -> np.ndarray:
#         dataset = []
#         for _ in range(n_trajs):
#             traj = [[self.env.sample_init_state()]]
#             for i in range(len(self.goals) - 1):
#                 part_traj = np.linspace(self.goals[i], self.goals[i + 1], endpoint=False, num=2)
#                 traj.append(part_traj)
#             traj.append([self.goals[-1]])
#             dataset.append(np.concatenate(traj))

#         return np.array(dataset)


# class DroneBranch(TaskBase):
#     def __init__(self, enable_gui: bool = True):
#         self.branch_1 = np.array([
#             [1, 0, 1],
#             [-1, 0, 1],
#         ])
#         self.branch_2 = np.array([
#             [1, -1, 1],
#             [-1, 1, 1],
#         ])
#         self.branch_3 = np.array([
#             [0, 1, 2],
#             [0, -1, 2],
#         ])

#         self.seq_time = [
#             [1, 2],
#             [3, 4]
#         ]

#         super(DroneBranch, self).__init__(4, 1000, enable_gui)

#     def _build_env(self, enable_gui: bool):
#         env = DronePIDEnv(enable_gui)

#         for goal in self.branch_1:
#             env.env_editor.add_ball(goal, color=(1, 0, 0, 0.5))

#         for goal in self.branch_2:
#             env.env_editor.add_ball(goal, color=(0, 1, 1, 0.5))

#         for goal in self.branch_3:
#             env.env_editor.add_ball(goal, color=(0, 0, 1, 0.5))

#         return env

#     def _build_spec(self) -> DiffTLTLSpec:
#         b1 = sequence(self.branch_1, seq_time=self.seq_time, name_prefix="b1")
#         b2 = sequence(self.branch_2, seq_time=self.seq_time, name_prefix="b2")
#         b3 = sequence(self.branch_3, seq_time=self.seq_time, name_prefix="b3")

#         final_1 = end_state(self.branch_1[-1])
#         final_2 = end_state(self.branch_2[-1])
#         final_3 = end_state(self.branch_3[-1])

#         imply_final_1 = b1.implies(final_1)
#         imply_final_2 = b2.implies(final_2)
#         imply_final_3 = b3.implies(final_3)

#         g = dim_lb(0.1, dim=2)

#         return g & imply_final_1 & imply_final_2 & imply_final_3 & (b1 | b2 | b3)

#     def demo(self, n_trajs: int) -> np.ndarray:
#         dataset = []
#         for _ in range(n_trajs):
#             traj = [[self.env.sample_init_state()]]
#             branch = random.choice([self.branch_1, self.branch_2, self.branch_3])
#             for i in range(len(branch) - 1):
#                 part_traj = np.linspace(branch[i], branch[i + 1], endpoint=False, num=2)
#                 traj.append(part_traj)
#             traj.append([branch[-1]])
#             dataset.append(np.concatenate(traj))

#         return np.array(dataset)


# class DroneCover(TaskBase):

#     def __init__(self, enable_gui: bool = True):
#         self.goals = np.array([
#             [2, 2, 1],
#             [1, -2, 2],
#             [-2, -1, 2],
#             [0, 2, 1]
#         ])[::-1]

#         super(DroneCover, self).__init__(8, 2000, enable_gui)

#     def _build_env(self, enable_gui: bool):
#         env = DronePIDEnv(enable_gui)

#         self.duck_size = 15
#         env.env_editor.add_duck(pos=np.array([0, 0, 0.099 * self.duck_size * 0.1]), size=self.duck_size)

#         for goal in self.goals:
#             env.env_editor.add_ball(goal, color=(0, 0, 1, 0.5))
#         return env

#     def _build_spec(self) -> DiffTLTLSpec:
#         coverage_spec = coverage(self.goals, close_radius=0.5)
#         ground_spec = dim_lb(0.1, dim=2)

#         spec = coverage_spec & ground_spec

#         return spec

#     def demo(self, n_trajs: int) -> np.ndarray:
#         dataset = []
#         for _ in range(n_trajs):
#             traj = [[self.env.sample_init_state()]]
#             for i in range(len(self.goals) - 1):
#                 part_traj = np.linspace(self.goals[i], self.goals[i + 1], endpoint=False, num=2)
#                 traj.append(part_traj)
#             traj.append([self.goals[-1]])
#             dataset.append(np.concatenate(traj))

#         return np.array(dataset)


# class DroneLoop(TaskBase):

#     def __init__(self, enable_gui: bool = False):
#         self.waypoints = ([-1, -1, 1],
#                           [1, 1, 2],
#                           [2, 1, 1])
#         super(DroneLoop, self).__init__(21, 4000, enable_gui)

#     def _build_env(self, enable_gui: bool):
#         env = DronePIDEnv(enable_gui)
#         for goal in self.waypoints:
#             env.env_editor.add_ball(goal, color=(0, 0, 1, 0.5))
#         return env

#     def _build_spec(self) -> DiffTLTLSpec:
#         spec = loop(self.waypoints, close_radius=0.5)
#         g = dim_lb(0.1, dim=2)

#         return spec & g

#     def demo(self, n_trajs: int) -> np.ndarray:
#         dataset = []
#         for _ in range(n_trajs):
#             traj = [self.env.sample_init_state()]
#             traj_len_so_far = 0
#             not_yet = True

#             while not_yet:
#                 for wp in self.waypoints:
#                     traj.append(wp)
#                     traj_len_so_far += 1
#                     if traj_len_so_far >= self.path_max_len:
#                         not_yet = False
#                         break

#             dataset.append(traj)

#         return np.array(dataset)


# class DroneSignal(TaskBase):
#     def __init__(self, enable_gui: bool):
#         self.loop_waypoints = ([-1, -1, 1.5], [1, 1, 2], [1, 1, 1])
#         self.final_goal = [-2, -2, 2]
#         self.until_time = 10

#         super(DroneSignal, self).__init__(self.until_time + 2, 4000, enable_gui)

#     def _build_env(self, enable_gui: bool):
#         env = DronePIDEnv(enable_gui)

#         for i, goal in enumerate(self.loop_waypoints):
#             env.env_editor.add_ball(goal, color=(1, 0, i / len(goal), 0.5))

#         env.env_editor.add_ball(self.final_goal, color=(1, 1, 0, 0.5))

#         return env

#     def _build_spec(self) -> DiffTLTLSpec:
#         spec = signal(self.loop_waypoints, self.final_goal, self.until_time, close_radius=0.5)
#         g = dim_lb(0.1, dim=2)

#         return spec & g

#     def demo(self, n_trajs: int) -> np.ndarray:
#         dateset = []

#         for _ in range(n_trajs):
#             traj = [self.env.sample_init_state()]
#             i = 0
#             while i < self.until_time:
#                 traj.append(self.loop_waypoints[i % len(self.loop_waypoints)])
#                 i += 1
#             traj.append(self.final_goal)
#             dateset.append(traj)

#         return np.array(dateset)
