from .base import BraxTaskBase
from .ant_maze import (
    AntMazeNav,
    AntMazeUMazeConstraint,
    AntMazeSingleSubgoal,
    AntMazeTwoSubgoals,
    AntMazeBranching1,
    AntMazeBranching2,
    AntMazeObligationConstraint1,
    AntMazeObligationConstraint2,
    AntMazeObligationConstraint3,
    AntMazeDiagonal,
    AntMazeObligationConstraint6,
)
from .simple_maze import (
    SimpleMazeNav,
    SimpleMazeUMazeConstraint,
    SimpleMazeCenterConstraint,
    SimpleMazeSingleSubgoal,
    SimpleMazeUSingleSubgoal,
    SimpleMazeTwoSubgoals,
    SimpleMazeUTwoSubgoals,
    SimpleMazeBranching1,
    SimpleMazeBranching2,
    SimpleMazeObligationConstraint1,
    SimpleMazeObligationConstraint2,
    SimpleMazeObligationConstraint3,
    SimpleMazeObligationConstraint4,
    SimpleMazeObligationConstraint5,
    SimpleMazeObligationConstraint6,
    SimpleMazeNotUntilAlwaysSubgoal,
    SimpleMazeUntil1,
)
from .simple_maze_3d import (
    SimpleMaze3DNav,
    SimpleMaze3DUMazeConstraint,
    SimpleMaze3DCenterConstraint,
    SimpleMaze3DSingleSubgoal,
    SimpleMaze3DTwoSubgoals,
    SimpleMaze3DBranching1,
    SimpleMaze3DObligationConstraint1,
    SimpleMaze3DObligationConstraint2,
    SimpleMaze3DObligationConstraint3,
)
from .ur5e import (
    UR5eReachTask,
    # UR5eReachShelfTask,
    # UR5eReachShelfEEFTask,
    # UR5eTranslateTask,
    UR5eEEFOneGoalTask,
    UR5eEEFObligationConstraint1Task,
    UR5eEEFObligationConstraint2Task,
    UR5eEEFEvenEasierScalabilityTestTask,
    UR5eEEFEasierScalabilityTestTask,
    UR5eEEFScalabilityTestTask,
    UR5eEEFBranchingScalabilityTestTask,
    UR5eEEFEasierBranchingScalabilityTestTask,
    UR5eEEFEvenEasierBranchingScalabilityTestTask,
    # UR5eEEFComplexBranchingScalabilityTestTask,
    UR5eEEFGraspTask,
)
from .panda import (
    PandaPushEasyTask
)
from .arm_eef import (
    ArmEEFBinpickEasyTask
)
from .x_point import (
    XPointFisacTask,
)


def get_task(robot_name: str, task_name: str) -> BraxTaskBase:
    _table = {
        "ant_maze": {
            "true": AntMazeNav,
            "umaze_constraint": AntMazeUMazeConstraint,
            "subgoal": AntMazeSingleSubgoal,
            "two_subgoals": AntMazeTwoSubgoals,
            "branching1": AntMazeBranching1,
            "branching2": AntMazeBranching2,
            "obligation1": AntMazeObligationConstraint1,
            "obligation2": AntMazeObligationConstraint2,
            "obligation3": AntMazeObligationConstraint3,
        },
        "simple_maze": {
            "true": SimpleMazeNav,
            "umaze_constraint": SimpleMazeUMazeConstraint,
            "center_constraint": SimpleMazeCenterConstraint,
            "subgoal": SimpleMazeSingleSubgoal,
            "two_subgoals": SimpleMazeTwoSubgoals,
            "branching1": SimpleMazeBranching1,
            "branching2": SimpleMazeBranching2,
            "obligation1": SimpleMazeObligationConstraint1,
            "obligation2": SimpleMazeObligationConstraint2,
            "obligation3": SimpleMazeObligationConstraint3,
            "obligation5": SimpleMazeObligationConstraint5,
        },
        "simple_maze_3d": {
            "true": SimpleMaze3DNav,
            "umaze_constraint": SimpleMaze3DUMazeConstraint,
            "center_constraint": SimpleMaze3DCenterConstraint,
            "subgoal": SimpleMaze3DSingleSubgoal,
            "two_subgoals": SimpleMaze3DTwoSubgoals,
            "branching1": SimpleMaze3DBranching1,
            "obligation1": SimpleMaze3DObligationConstraint1,
            "obligation2": SimpleMaze3DObligationConstraint2,
            "obligation3": SimpleMaze3DObligationConstraint3,
        },
        "ur5e": {
            "reach": UR5eReachTask,
            # "reach_shelf": UR5eReachShelfTask,
            # "reach_shelf_eef": UR5eReachShelfEEFTask,
            # "translate": UR5eTranslateTask,
            # main
            "one_goal": UR5eEEFOneGoalTask,
            "obligation1": UR5eEEFObligationConstraint1Task,
            "obligation2": UR5eEEFObligationConstraint2Task,
            "easier_scalability": UR5eEEFEasierScalabilityTestTask,
            "even_easier_scalability": UR5eEEFEvenEasierScalabilityTestTask,
            "scalability": UR5eEEFScalabilityTestTask,
            "branching": UR5eEEFBranchingScalabilityTestTask,
            "easier_branching": UR5eEEFEasierBranchingScalabilityTestTask,
            "even_easier_branching": UR5eEEFEvenEasierBranchingScalabilityTestTask,
            # "complex": UR5eEEFComplexBranchingScalabilityTestTask,
        },
        "x_point": {
            "fisac": XPointFisacTask,
        },
    }

    return _table[robot_name][task_name]()
