from .base import TaskBase
# from .car import CarSequence, CarCover, CarBranch, CarLoop, CarSignal
# from .doggo import DoggoSequence, DoggoCover, DoggoBranch, DoggoLoop, DoggoSignal
# from .drone import DroneSequence, DroneCover, DroneBranch, DroneLoop, DroneSignal
from .point import (
    PointStraightSequence,
    PointTurnSequence,
    PointOneGoal,
    PointTwoGoals,
    PointSequence
) # , PointCover, PointBranch, PointLoop, PointSignal
from .xy_point import (
    # XYPointMazeNav,
    XYPointStraight,
    XYPointDiagonal1,
    XYPointDiagonal2,
    XYPointStraightSequence,
    XYPointTurnSequence,
    XYPointObstacle,
    XYPointObstacle2,
)
from .xy_point_maze import (
    XYPointMazeNav,
)
from .hopper import (
    HopperStraight,
)
from .ant_maze import (
    AntMazeNav,
    AntMazeUMazeConstraint,
    AntMazeSingleSubgoal,
    AntMazeTwoSubgoals,
    AntMazeBranching1,
    AntMazeBranching2,
    AntMazeObligationConstraint1,
)
from .ant_push import (
    AntPushDefault,
)
from .simple_maze import (
    SimpleMazeNav,
    SimpleMazeCenterConstraint,
    SimpleMazeUMazeConstraint,
    SimpleMazeSingleSubgoal,
    SimpleMazeTwoSubgoals,
    SimpleMazeBranching1,
    SimpleMazeBranching2,
    SimpleMazeObligationConstraint1,
    SimpleMazeObligationConstraint2,
    SimpleMazeUntil1,
)
from .simple_maze_3d import (
    SimpleMaze3DNav,
    SimpleMaze3DUMazeConstraint,
    SimpleMaze3DSingleSubgoal,
    SimpleMaze3DTwoSubgoals,
    SimpleMaze3DBranching1,
)
from .drone_maze import (
    DroneMazeNav,
)
from .skate import (
    SkateStraightSequence,
    SkateTurnSequence,
)
from .ant import (
    AntRun,
    AntPositiveY,
    AntStraightSequence,
    AntTurnSequence
)
from .panda import (
    PandaReach,
)
from .ur5e import (
    UR5eReachTask,
)


def get_task(robot_name: str, task_name: str) -> TaskBase:
    _table = {
        # "drone": {
        #     "seq": DroneSequence,
        #     "cover": DroneCover,
        #     "branch": DroneBranch,
        #     "loop": DroneLoop,
        #     "signal": DroneSignal
        # },
        "ant": {
            "run": AntRun,
            "positive_y": AntPositiveY,
            "straight_seq": AntStraightSequence,
            "turn_seq": AntTurnSequence,
        },
        "skate": {
            "straight_seq": SkateStraightSequence,
            "turn_seq": SkateTurnSequence,
        },
        "xy_point": {
            "diagonal1": XYPointDiagonal1,
            "diagonal2": XYPointDiagonal2,
            "straight": XYPointStraight,
            "straight_seq": XYPointStraightSequence,
            "turn_seq": XYPointTurnSequence,
            "obstacle": XYPointObstacle,
            "obstacle2": XYPointObstacle2,
        },
        "xy_point_maze": {
            "true": XYPointMazeNav,
        },
        "hopper": {
            "straight": HopperStraight,
        },
        "ant_maze": {
            "true": AntMazeNav,
            "umaze_constraint": AntMazeUMazeConstraint,
            "subgoal": AntMazeSingleSubgoal,
            "two_subgoals": AntMazeTwoSubgoals,
            "branching1": AntMazeBranching1,
            "branching2": AntMazeBranching2,
            "obligation1": AntMazeObligationConstraint1,
        },
        "ant_push": {
            "true": AntPushDefault,
        },
        "simple_maze": {
            "true": SimpleMazeNav,
            "center_constraint": SimpleMazeCenterConstraint,
            "umaze_constraint": SimpleMazeUMazeConstraint,
            "subgoal": SimpleMazeSingleSubgoal,
            "two_subgoals": SimpleMazeTwoSubgoals,
            "branching1": SimpleMazeBranching1,
            "branching2": SimpleMazeBranching2,
            "obligation1": SimpleMazeObligationConstraint1,
            "obligation2": SimpleMazeObligationConstraint2,
            "until1": SimpleMazeUntil1,
        },
        "simple_maze_3d": {
            "true": SimpleMaze3DNav,
            "umaze_constraint": SimpleMaze3DUMazeConstraint,
            "subgoal": SimpleMaze3DSingleSubgoal,
            "two_subgoals": SimpleMaze3DTwoSubgoals,
            "branching1": SimpleMaze3DBranching1,
        },
        "drone_maze": {
            "true": DroneMazeNav,
        },
        "point": {
            "straight_seq": PointStraightSequence,
            "turn_seq": PointTurnSequence,
            "one_goal": PointOneGoal,
            "two_goals": PointTwoGoals,
            "seq": PointSequence,
            # "cover": PointCover,
            # "branch": PointBranch,
            # "loop": PointLoop,
            # "signal": PointSignal
        },
        "panda": {
            "reach": PandaReach,
        },
        "ur5e": {
            "reach": UR5eReachTask,
        },
        # "car": {
        #     "seq": CarSequence,
        #     "cover": CarCover,
        #     "branch": CarBranch,
        #     "loop": CarLoop,
        #     "signal": CarSignal
        # },
        # "doggo": {
        #     "seq": DoggoSequence,
        #     "cover": DoggoCover,
        #     "branch": DoggoBranch,
        #     "loop": DoggoLoop,
        #     "signal": DoggoSignal
        # },
    }

    return _table[robot_name][task_name]()
