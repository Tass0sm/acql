from abc import ABC, abstractmethod

from gymnasium.spaces import Space
# from brax import base
# from brax import math
from brax.envs.base import PipelineEnv, State
# from brax.io import mjcf
# from etils import epath
# import jax
# from jax import numpy as jp
# import mujoco

import xml.etree.ElementTree as ET

from corallab_stl import Var


def load_and_configure_xml(path, wall_config=None):
    tree = ET.parse(path)

    # Get worldbody element
    worldbody = tree.find(".//worldbody")

    if wall_config is not None:
        walls_locations, walls_size = (
            wall_config["walls_locations"],
            wall_config["walls_size"]
        )

        WALL_HEIGHT = 0.5

        for i, (x, y) in enumerate(walls_locations):
            ET.SubElement(
                worldbody,
                "geom", name=f"wall_{i}",
                pos=f"{x} {y} {WALL_HEIGHT / 2}",
                size=" ".join([str(walls_size)] * 3),
                type="box",
                rgba="0.7 0.5 0.3 1.0",
            )

    # # Add geom blocks for every wall in the maze
    # for i in range(len(maze_layout)):
    #     for j in range(len(maze_layout[0])):
    #         struct = maze_layout[i][j]
    #         if struct == 1:
    #             ET.SubElement(
    #                 worldbody, "geom",
    #                 name="block_%d_%d" % (i, j),
    #                 pos="%f %f %f" % (i * maze_size_scaling,
    #                                 j * maze_size_scaling,
    #                                 MAZE_HEIGHT / 2 * maze_size_scaling),
    #                 size="%f %f %f" % (0.5 * maze_size_scaling,
    #                                     0.5 * maze_size_scaling,
    #                                     MAZE_HEIGHT / 2 * maze_size_scaling),
    #                 type="box",
    #                 material="",
    #                 contype="1",
    #                 conaffinity="1",
    #                 rgba="0.7 0.5 0.3 1.0",
    #             )

    xml_string = ET.tostring(tree.getroot())
    return xml_string


class GoalConditionedEnv(PipelineEnv, ABC):
    def __init__(
            self,
            # observation_space: Box,
            # action_space: Box,
            # init_space: Box,
            # goal_space: Box,
            # goal: np.ndarray,
            # reach_reward: float
            **kwargs
    ):
        super().__init__(**kwargs)

        # self.observation_space = observation_space
        # self.action_space = action_space
        # self.init_space = init_space
        # self.goal_space = goal_space
        # self._goal = goal
        # self.reach_reward = reach_reward

        # self.prev_dist_to_goal = None
        # self.init_state = None
        # self._goal_space_size = None
        # self._stop = False
        pass

    # @abstractmethod
    # def set_state(self, state: np.ndarray):
    #     pass

    # @abstractmethod
    # def get_obs(self) -> np.ndarray:
    #     pass

    # @abstractmethod
    # def reach(self) -> bool:
    #     pass

    # @abstractmethod
    # def update_state(self, action: np.ndarray) -> Dict:
    #     pass

    # @abstractmethod
    # def get_state(self) -> np.ndarray:
    #     pass

    # @abstractmethod
    # def collision(self) -> bool:
    #     pass

    @property
    @abstractmethod
    def observation_space(self) -> Var:
        pass

    @property
    @abstractmethod
    def action_space(self) -> Var:
        pass

    # def sample_init_state(self) -> np.ndarray:
    #     return self.init_space.sample()

    # def reset(self, state: np.ndarray = None) -> np.ndarray:
    #     self.prev_dist_to_goal = None
    #     self._stop = False
    #     if state is None:
    #         state = self.sample_init_state()
    #     self.set_state(state)
    #     self.init_state = self.get_state()

    #     return self.get_obs()

    # def step(self, action: np.ndarray):
    #     info = self.update_state(action)
    #     reach = self.reach()
    #     if reach:
    #         reward = self.reach_reward
    #     else:
    #         reward = self._ctrl_reward()

    #     info["reach"] = reach
    #     info["collision"] = self.collision()
    #     done = info["collision"] or self._stop

    #     return self.get_obs(), reward, done, info

    # def set_goal(self, goal: np.ndarray):
    #     self._goal = goal
    #     self.prev_dist_to_goal = None

    # def get_goal(self) -> np.ndarray:
    #     return self._goal

    # def _ctrl_reward(self) -> float:
    #     """
    #     a ``stateful'' reward function
    #     """
    #     dist_to_goal = np.linalg.norm(self.get_goal() - self.get_state())

    #     if self.prev_dist_to_goal is None or self.reach():
    #         self.prev_dist_to_goal = dist_to_goal
    #         return 0.0

    #     disp = self.prev_dist_to_goal - dist_to_goal
    #     reward = disp  # positive if closer to goal

    #     self.prev_dist_to_goal = dist_to_goal
    #     return reward

    # @property
    # def goal_space_size(self) -> int:
    #     if self._goal_space_size is None:
    #         self._goal_space_size = len(self.sample_init_state())
    #     return self._goal_space_size

    # def stop_in_next_step(self):
    #     self._stop = True
