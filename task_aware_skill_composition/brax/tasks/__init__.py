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
)
from .ant_push import (
    AntPushDefault,
)
from .simple_maze import (
    SimpleMazeNav,
    SimpleMazeCenterConstraint,
    SimpleMazeUMazeConstraint,
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
        },
        "ant_push": {
            "true": AntPushDefault,
        },
        "simple_maze": {
            "true": SimpleMazeNav,
            "center_constraint": SimpleMazeCenterConstraint,
            "umaze_constraint": SimpleMazeUMazeConstraint
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
