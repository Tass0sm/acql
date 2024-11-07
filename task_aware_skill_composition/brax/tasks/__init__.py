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
    XYPointStraightSequence,
    XYPointTurnSequence,
)
from .skate import (
    SkateStraightSequence,
    SkateTurnSequence,
)
from .ant import (
    AntPositiveY,
    AntStraightSequence
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
            "positive_y": AntPositiveY,
            "straight_seq": AntStraightSequence,
        },
        "skate": {
            "straight_seq": SkateStraightSequence,
            "turn_seq": SkateTurnSequence,
        },
        "xy_point": {
            "straight_seq": XYPointStraightSequence,
            "turn_seq": XYPointTurnSequence,
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
