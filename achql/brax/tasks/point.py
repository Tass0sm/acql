import random

import numpy as np
import jax.numpy as jnp

from gymnasium.spaces.utils import flatdim

from achql.brax.envs.point import Point
from achql.brax.envs.base import GoalConditionedEnv
from achql.brax.tasks.base import TaskBase
# from schrl.tltl.spec import DiffTLTLSpec
# from schrl.tltl.template import sequence, ith_state, loop, signal, coverage
from achql.brax.tasks.templates import sequence, inside_circle, outside_circle

from achql.stl import Expression, Var
import achql.stl.expression_jax2 as stl


class PointTaskBase(TaskBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create_vars(self):
        self.wp_var = Var("wp", idx=0, dim=2)
        self.obs_var = Var("obs", idx=0, dim=flatdim(self.env.observation_space),
                           position=(0, 2),
                           y_velocity=14)


class PointStraightSequence(PointTaskBase):
    """
    F(region1 ^ X(F(region2)))
    """

    def __init__(self, backend="mjx"):
        self.goal1_location = jnp.array([2.0, 0.0])
        self.goal2_location = jnp.array([6.0, 0.0])
        self.goal_radius = 1.0

        super().__init__(None, 1000, backend=backend)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = Point(backend=backend)
        return env

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        at_goal1 = inside_circle(obs_var.position, self.goal1_location, self.goal_radius)
        at_goal2 = inside_circle(obs_var.position, self.goal2_location, self.goal_radius)

        phi = stl.STLUntimedEventually(
            stl.STLAnd(at_goal1, stl.STLNext(stl.STLUntimedEventually(at_goal2)))
        )

        return phi


class PointTurnSequence(PointTaskBase):
    """
    F(region1 ^ X(F(region2)))
    """

    def __init__(self, backend="mjx"):
        self.goal1_location = jnp.array([2.0, 0.0])
        self.goal2_location = jnp.array([2.0, 4.0])
        self.goal_radius = 1.0

        super().__init__(None, 1000, backend=backend)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = Point(backend=backend)
        return env

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        at_goal1 = inside_circle(obs_var.position, self.goal1_location, self.goal_radius)
        at_goal2 = inside_circle(obs_var.position, self.goal2_location, self.goal_radius)

        phi = stl.STLUntimedEventually(
            stl.STLAnd(at_goal1, stl.STLNext(stl.STLUntimedEventually(at_goal2)))
        )

        return phi


class PointOneGoal(PointTaskBase):
    """Based on the office grid world example reward machines paper

    F(coffee)
    """

    def __init__(self, backend="mjx"):

        self.goal1_location = jnp.array([4.0, -4.0])

        super().__init__(None, 1000, backend=backend)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        config = {
            'walls_locations': [
                [ 1.0, -1.0], [ 1.0, -0.5], [1.0, 0.5], [1.0, 1.0],
                [ 0.5, -1.0], [ 0.5, 1.0],
                [ 0.0, -1.0], [ 0.0, 1.0],
                [-0.5, -1.0], [-0.5, 1.0],
                [-1.0, -1.0], [-1.0, -0.5], [-1.0, 0.0], [-1.0, 0.5], [-1.0, 1.0],
            ],
            'walls_size': 0.25,
        }
        env = Point(wall_config=config, backend=backend)
        # for pos in self.goals:
        #     env.add_goal_mark(pos, color=(0, 1, 1, 0.5))
        return env

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        at_goal1 = inside_circle(obs_var.position, self.goal1_location, 1.0)
        phi = stl.STLUntimedEventually(at_goal1)
        return phi


class PointTwoGoals(TaskBase):
    """Based on the office grid world example reward machines paper

    F(coffee ^ X(F(office))) ^ G(not obs)
    """

    def __init__(self, backend="mjx"):

        self.goal1_location = jnp.array([4.0, -4.0])
        self.goal2_location = jnp.array([0.0, 0.0])

        self.obstacle_location = jnp.array([4.0, 4.0])
        self.obstacle_radius = 1.0

        super().__init__(None, 1000, backend=backend)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        config = {
            'walls_locations': [
                [ 1.0, -1.0], [ 1.0, -0.5], [1.0, 0.5], [1.0, 1.0],
                [ 0.5, -1.0], [ 0.5, 1.0],
                [ 0.0, -1.0], [ 0.0, 1.0],
                [-0.5, -1.0], [-0.5, 1.0],
                [-1.0, -1.0], [-1.0, -0.5], [-1.0, 0.0], [-1.0, 0.5], [-1.0, 1.0],
            ],
            'walls_size': 0.25,
        }
        env = Point(wall_config=config, backend=backend)
        # for pos in self.goals:
        #     env.add_goal_mark(pos, color=(0, 1, 1, 0.5))
        return env

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        at_goal1 = inside_circle(obs_var.position, self.goal1_location, 1.0)
        at_goal2 = inside_circle(obs_var.position, self.goal2_location, 1.0)
        not_in_obstacle = outside_circle(obs_var.position, self.obstacle_location, self.obstacle_radius)

        phi = stl.STLAnd(
            stl.STLUntimedEventually(
                stl.STLAnd(at_goal1, stl.STLNext(stl.STLUntimedEventually(at_goal2)))
            ),
            stl.STLUntimedAlways(not_in_obstacle)
        )

        return phi


class PointSequence(TaskBase):
    def __init__(self, backend="mjx"):
        self.goals = jnp.array([[2.0, 2.0],
                                [-1.0, 2.0],
                                [-2.0, -1.0],
                                [1.0, -2.0]])
        self.seq_time = [[1, 2],
                         [3, 4],
                         [5, 6],
                         [7, 8]]

        super().__init__(8, 1000, backend=backend)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        config = {
            'walls_num': 17,
            'walls_locations': [
                [0, 0],
                [0, 0.5], [0, 1], [0, 3], [0, 3.5],
                [0.5, 0], [1, 0], [3, 0], [3.5, 0],
                [0, -0.5], [0, -1], [0, -3], [0, -3.5],
                [-0.5, 0], [-1, 0], [-3, 0], [-3.5, 0],
            ],
            'walls_size': 0.25,
        }
        env = Point(wall_config=config, backend=backend)
        # for pos in self.goals:
        #     env.add_goal_mark(pos, color=(0, 1, 1, 0.5))
        return env

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        goal_spec = sequence(wp_var, self.goals, close_radius=0.5, seq_times=self.seq_time)
        return goal_spec

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        pass


# class PointCover(TaskBase):
#     def __init__(self, enable_gui: bool = True):
#         self.goals = np.array([
#             [2, 2],
#             [1, -2],
#             [-2, -1],
#             [-1, 2]
#         ])
#         self.seq_time = [
#             [1, 2],
#             [3, 4],
#             [5, 6],
#             [7, 8]
#         ]

#         super(PointCover, self).__init__(8, 1000, enable_gui)

#     def _build_env(self, enable_gui=True) -> GoalEnv:
#         config = {
#             'walls_num': 17,
#             'walls_locations': [
#                 [0, 0],
#                 [0, 0.5], [0, 1], [0, 3], [0, 3.5],
#                 [0.5, 0], [1, 0], [3, 0], [3.5, 0],
#                 [0, -0.5], [0, -1], [0, -3], [0, -3.5],
#                 [-0.5, 0], [-1, 0], [-3, 0], [-3.5, 0],
#             ],
#             'walls_size': 0.25,
#         }
#         env = Point(config)
#         for pos in self.goals:
#             env.add_goal_mark(pos, color=(0, 1, 1, 0.5))
#         return env

#     def _build_spec(self) -> DiffTLTLSpec:
#         goal_spec = coverage(self.goals, close_radius=0.5)
#         return goal_spec

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


# class PointBranch(TaskBase):
#     def __init__(self, enable_gui: bool = True):
#         self.branches = np.array([
#             [[0, 2], [0, -2]],
#             [[2, -2], [-2, 2]],
#             [[2, 0], [-2, 0]]
#         ])

#         self.seq_time = [
#             [1, 2],
#             [3, 4]
#         ]

#         super(PointBranch, self).__init__(4, 800, enable_gui)

#     def _build_env(self, enable_gui: bool):
#         config = {
#             'walls_num': 17,
#             'walls_locations': [
#                 [0, 0],
#                 [0, 0.5], [0, 1], [0, 3], [0, 3.5],
#                 [0.5, 0], [1, 0], [3, 0], [3.5, 0],
#                 [0, -0.5], [0, -1], [0, -3], [0, -3.5],
#                 [-0.5, 0], [-1, 0], [-3, 0], [-3.5, 0],
#             ],
#             'walls_size': 0.25,
#         }
#         env = Point(config)

#         init_low = np.array([-2, -2], dtype=np.float32)
#         init_high = np.array([-1, -1], dtype=np.float32)
#         env.update_initial_space(init_low, init_high)

#         for goal in self.branches[0]:
#             env.add_goal_mark(goal, color=(1, 0, 0, 0.5))

#         for goal in self.branches[1]:
#             env.add_goal_mark(goal, color=(0, 1, 0, 0.5))

#         for goal in self.branches[2]:
#             env.add_goal_mark(goal, color=(0, 0, 1, 0.5))

#         return env

#     def _build_spec(self) -> DiffTLTLSpec:
#         start_1 = ith_state((self.branches[0][0]), i=1, close_radius=0.6)
#         start_2 = ith_state((self.branches[1][0]), i=1, close_radius=0.6)
#         start_3 = ith_state((self.branches[2][0]), i=1, close_radius=0.6)

#         b1 = sequence(self.branches[0], close_radius=0.6, seq_time=self.seq_time, name_prefix="b1")
#         b2 = sequence(self.branches[1], close_radius=0.6, seq_time=self.seq_time, name_prefix="b2")
#         b3 = sequence(self.branches[2], close_radius=0.6, seq_time=self.seq_time, name_prefix="b3")

#         imply_final_1 = start_1.implies(b1)
#         imply_final_2 = start_2.implies(b2)
#         imply_final_3 = start_3.implies(b3)

#         return imply_final_1 & imply_final_2 & imply_final_3 & (start_1 | start_2 | start_3)

#     def demo(self, n_trajs: int) -> np.ndarray:
#         dataset = []
#         mid_points = np.array([[0, 2.5], [2.5, 2.5], [0, 2.5]])
#         for _ in range(n_trajs):
#             traj = [[self.env.sample_init_state()]]
#             indx = random.choice([1])
#             for j in range(len(self.branches[indx]) - 1):
#                 part_traj = np.array([self.branches[indx][j], mid_points[indx]])
#                 traj.append(part_traj)
#             traj.append([self.branches[indx][-1]])
#             dataset.append(np.concatenate(traj))

#         return np.array(dataset)


# class PointLoop(TaskBase):
#     def __init__(self, enable_gui: bool = False):
#         self.waypoints = ([-2, -2],
#                           [-2, 2],
#                           [2, 2],
#                           [2, -2])
#         super(PointLoop, self).__init__(17, 2000, enable_gui)

#     def _build_env(self, enable_gui: bool):
#         config = {
#             'walls_num': 17,
#             'walls_locations': [
#                 [0, 0],
#                 [0, 0.5], [0, 1], [0, 3], [0, 3.5],
#                 [0.5, 0], [1, 0], [3, 0], [3.5, 0],
#                 [0, -0.5], [0, -1], [0, -3], [0, -3.5],
#                 [-0.5, 0], [-1, 0], [-3, 0], [-3.5, 0],
#             ],
#             'walls_size': 0.25,
#         }
#         env = Point(config)
#         for goal in self.waypoints:
#             env.add_goal_mark(goal, color=(1, 0, 0, 0.5))
#         return env

#     def _build_spec(self) -> DiffTLTLSpec:
#         spec = loop(self.waypoints, close_radius=0.6)

#         return spec

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


# class PointSignal(TaskBase):
#     def __init__(self, enable_gui: bool):
#         self.loop_waypoints = ([-2, -2],
#                                [-2, 2],
#                                [2, 2],
#                                [2, -2])
#         self.final_goal = [0, 0]
#         self.until_time = 10

#         super(PointSignal, self).__init__(self.until_time + 2, 2000, enable_gui)

#     def _build_env(self, enable_gui: bool):
#         env = Point()

#         for i, goal in enumerate(self.loop_waypoints):
#             env.add_goal_mark(goal, color=(0, 1, i / len(goal), 0.5))

#         env.add_goal_mark(self.final_goal, color=(1, 1, 0, 0.5))

#         return env

#     def _build_spec(self) -> DiffTLTLSpec:
#         spec = signal(self.loop_waypoints, self.final_goal, self.until_time, close_radius=0.8)

#         return spec

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
