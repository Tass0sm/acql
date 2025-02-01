import random

import numpy as np
import jax.numpy as jnp

from achql.brax.envs.manipulation.ur5e_reach import UR5eReach
from achql.brax.envs.manipulation.ur5e_reach_shelf import UR5eReachShelf
from achql.brax.envs.manipulation.ur5e_reach_shelf_eef import UR5eReachShelfEEF
from achql.brax.envs.manipulation.ur5e_translate import UR5eTranslate
from achql.brax.envs.base import GoalConditionedEnv
from achql.brax.tasks.base import TaskBase
from achql.brax.tasks.templates import sequence, inside_circle, outside_circle, inside_box, true_exp
from achql.brax.tasks.mixins import *
from achql.hierarchy.ur5e.load import load_ur5e_options, load_ur5e_eef_options
from achql.hierarchy.option import FixedLengthTerminationPolicy

from achql.stl import Expression, Var
import achql.stl.expression_jax2 as stl


class UR5eTaskBase(TaskBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create_vars(self):
        self.wp_var = Var("wp", idx=0, dim=3)
        self.obs_var = Var("obs", idx=0, dim=self.env.observation_size, position=(0, 3))

    def get_options(self):
        return self.get_learned_options()

    def get_learned_options(self):
        # Tentative: option env size may change
        adapter = lambda x: x[..., :12]
        return load_ur5e_options(termination_policy=FixedLengthTerminationPolicy(1),
                                 adapter=adapter)

    def get_hard_coded_options(self):
        raise NotImplementedError()

    @property
    def hdqn_her_hps(self):
        return {
            "num_timesteps": 8_000_000,
            "reward_scaling": 1,
            "num_evals": 40,
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

    @property
    def hdcqn_her_hps(self):
        return {
            **self.hdqn_her_hps,
            "cost_scaling": 10.0,
            "safety_threshold": 0.0,
            "gamma_decay": 0.20, # the higher, the slower it reaches 1.0
            "gamma_end_value": 0.99,
        }

    @property
    def sac_her_hps(self):
        return {
            "num_timesteps": 2_000_000,
            "reward_scaling": 10,
            "num_evals": 50,
            "episode_length": 1000,
            "normalize_observations": True,
            "action_repeat": 1,
            "unroll_length": 62, # TODO: Reducing this increases time. What else does it affect?
            "multiplier_num_sgd_steps": 1,
            "max_devices_per_host": 1,
            "max_replay_size": 10000,
            # 8192, the default, causes the error "TypeError: broadcast_in_dim shape must have every element be nonnegative, got (-2, 50)."
            "min_replay_size": 1000,
            "use_her": False,
            "discounting": 0.97,
            # "learning_rate": 3e-4,
            "num_envs": 512,
            "batch_size": 256,
        }


class UR5eReachTask(UR5eTaskBase):
    def __init__(self, backend="mjx"):
        super().__init__(None, 1000, backend=backend)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = UR5eReach(
            backend=backend
        )
        return env

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        return true_exp(obs_var)


class UR5eReachShelfTask(UR5eTaskBase):
    def __init__(self, backend="mjx"):
        super().__init__(None, 1000, backend=backend)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = UR5eReachShelf(
            backend=backend
        )
        return env

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        return true_exp(obs_var)


class UR5eEEFTaskBase(UR5eTaskBase):
    def get_options(self):
        return self.get_hard_coded_options()

    def get_hard_coded_options(self):
        return load_ur5e_eef_options(termination_policy=FixedLengthTerminationPolicy(1),
                                     ctrl_scale=0.25)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = UR5eReachShelfEEF(
            backend=backend
        )
        return env


class UR5eEEFOneGoalTask(UR5eEEFTaskBase):
    def __init__(self, backend="mjx"):
        self.goal1_location = jnp.array([-0.225, 0.5, 0.1])
        self.goal2_location = jnp.array([0.225, 0.5, 0.5])
        self.goal3_location = jnp.array([-0.225, 0.5, 0.5])
        self.goal4_location = jnp.array([0.225, 0.5, 0.1])
        self.goal_radius = 0.10

        real_table_obs_corners = (jnp.array([-0.4572, 0.3952, 0.2844]), jnp.array([0.4572, 1.0048, 0.3444]))
        table_obs_padding = 0.00
        self.table_obs_corners = (real_table_obs_corners[0] - table_obs_padding,
                                  real_table_obs_corners[1] + table_obs_padding)

        real_wall_obs_corners = (jnp.array([-0.0150, 0.3952, 0.0000]), jnp.array([0.0150, 1.0048, 1.8288]))
        wall_obs_padding = 0.00
        self.wall_obs_corners = (real_wall_obs_corners[0] - wall_obs_padding,
                                 real_wall_obs_corners[1] + wall_obs_padding)

        super().__init__(None, 1000, backend=backend)

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        in_goal1 = inside_circle(obs_var.position, self.goal1_location, self.goal_radius, has_goal=True)
        phi_liveness = stl.STLUntimedEventually(in_goal1)

        # in_table_obs = inside_box(obs_var.position, *self.table_obs_corners)
        # in_wall_obs = inside_box(obs_var.position, *self.wall_obs_corners)
        # in_either_obs = stl.STLOr(in_table_obs, in_wall_obs)
        # phi_safety = stl.STLUntimedAlways(stl.STLNegation(in_either_obs))
        # phi_safety = stl.STLUntimedAlways(stl.STLNegation(in_wall_obs))

        # phi = stl.STLAnd(phi_liveness, phi_safety)
        phi = phi_liveness

        return phi

    @property
    def hdcqn_her_hps(self):
        return {
            **self.hdqn_her_hps,
            "cost_scaling": 15.0,
            "safety_threshold": -0.06,
            "gamma_decay": 0.20, # the higher, the slower it reaches 1.0
            "gamma_end_value": 0.98,
        }


class UR5eEEFObligationConstraint1Task(UR5eEEFTaskBase):
    def __init__(self, backend="mjx"):
        self.goal1_location = jnp.array([-0.225, 0.5, 0.1])
        self.goal2_location = jnp.array([0.225, 0.5, 0.5])
        self.goal3_location = jnp.array([-0.225, 0.5, 0.5])
        self.goal4_location = jnp.array([0.225, 0.5, 0.1])
        self.goal_radius = 0.10

        real_table_obs_corners = (jnp.array([-0.4572, 0.3952, 0.2844]), jnp.array([0.4572, 1.0048, 0.3444]))
        table_obs_padding = 0.00
        self.table_obs_corners = (real_table_obs_corners[0] - table_obs_padding,
                                  real_table_obs_corners[1] + table_obs_padding)

        real_wall_obs_corners = (jnp.array([-0.0150, 0.3952, 0.0000]), jnp.array([0.0150, 1.0048, 1.8288]))
        wall_obs_padding = 0.00
        self.wall_obs_corners = (real_wall_obs_corners[0] - wall_obs_padding,
                                 real_wall_obs_corners[1] + wall_obs_padding)

        super().__init__(None, 1000, backend=backend)

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        in_goal1 = inside_circle(obs_var.position, self.goal1_location, self.goal_radius, has_goal=True)
        phi_liveness = stl.STLUntimedEventually(in_goal1)

        in_table = inside_box(obs_var.position, *self.table_obs_corners)
        in_wall = inside_box(obs_var.position, *self.wall_obs_corners)
        in_either_obstacle = stl.STLOr(in_table, in_wall)
        phi_safety = stl.STLUntimedAlways(stl.STLNegation(in_either_obstacle))

        phi = stl.STLAnd(phi_liveness, phi_safety)
        return phi


class UR5eEEFObligationConstraint2Task(UR5eEEFTaskBase):
    def __init__(self, backend="mjx"):
        self.goal1_location = jnp.array([-0.225, 0.5, 0.1])
        self.goal2_location = jnp.array([0.225, 0.5, 0.5])
        self.goal3_location = jnp.array([-0.225, 0.5, 0.5])
        self.goal4_location = jnp.array([0.225, 0.5, 0.1])
        self.goal_radius = 0.10

        real_table_obs_corners = (jnp.array([-0.4572, 0.4552, 0.2844]), jnp.array([0.4572, 1.0048, 0.3444]))
        table_obs_padding = 0.00
        self.table_obs_corners = (real_table_obs_corners[0] - table_obs_padding,
                                  real_table_obs_corners[1] + table_obs_padding)

        real_wall_obs_corners = (jnp.array([-0.0150, 0.4552, 0.0000]), jnp.array([0.0150, 1.0048, 1.8288]))
        wall_obs_padding = 0.00
        self.wall_obs_corners = (real_wall_obs_corners[0] - wall_obs_padding,
                                 real_wall_obs_corners[1] + wall_obs_padding)

        super().__init__(None, 1000, backend=backend)

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        in_goal1 = inside_circle(obs_var.position, self.goal1_location, self.goal_radius, has_goal=True)
        in_goal2 = inside_circle(obs_var.position, self.goal2_location, self.goal_radius, has_goal=True)
        phi_liveness = stl.STLUntimedEventually(
            stl.STLAnd(in_goal1, stl.STLNext(stl.STLUntimedEventually(
                in_goal2))))

        in_table = inside_box(obs_var.position, *self.table_obs_corners)
        in_wall = inside_box(obs_var.position, *self.wall_obs_corners)
        in_either_obstacle = stl.STLOr(in_table, in_wall)
        phi_safety = stl.STLUntimedAlways(stl.STLNegation(in_either_obstacle))

        phi = stl.STLAnd(phi_liveness, phi_safety)
        return phi


class UR5eEEFEvenEasierScalabilityTestTask(UR5eEEFTaskBase):
    def __init__(self, backend="mjx"):
        self.goal1_location = jnp.array([0.225, 0.5, 0.5])
        self.goal2_location = jnp.array([-0.225, 0.5, 0.5])
        self.goal3_location = jnp.array([-0.225, 0.5, 0.1])
        self.goal_radius = 0.10

        real_wall_obs_corners = (jnp.array([-0.0150, 0.4552, 0.0000]), jnp.array([0.0150, 1.0048, 1.8288]))
        wall_obs_padding = 0.00
        self.wall_obs_corners = (real_wall_obs_corners[0] - wall_obs_padding,
                                 real_wall_obs_corners[1] + wall_obs_padding)

        super().__init__(None, 1000, backend=backend)

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        in_goal1 = inside_circle(obs_var.position, self.goal1_location, self.goal_radius, has_goal=True)
        in_goal2 = inside_circle(obs_var.position, self.goal2_location, self.goal_radius, has_goal=True)
        phi_liveness = stl.STLUntimedEventually(
            stl.STLAnd(in_goal1, stl.STLNext(stl.STLUntimedEventually(
                in_goal2))))

        in_wall_obs = inside_box(obs_var.position, *self.wall_obs_corners)
        phi_safety = stl.STLUntimedAlways(stl.STLNegation(in_wall_obs))

        phi = stl.STLAnd(phi_liveness, phi_safety)
        return phi


class UR5eEEFEasierScalabilityTestTask(UR5eEEFTaskBase):
    def __init__(self, backend="mjx"):
        self.goal1_location = jnp.array([0.225, 0.5, 0.5])
        self.goal2_location = jnp.array([-0.225, 0.5, 0.5])
        self.goal3_location = jnp.array([-0.225, 0.5, 0.1])
        self.goal_radius = 0.10

        real_table_obs_corners = (jnp.array([-0.4572, 0.4552, 0.2844]), jnp.array([0.4572, 1.0048, 0.3444]))
        table_obs_padding = 0.00
        self.table_obs_corners = (real_table_obs_corners[0] - table_obs_padding,
                                  real_table_obs_corners[1] + table_obs_padding)

        real_wall_obs_corners = (jnp.array([-0.0150, 0.4552, 0.0000]), jnp.array([0.0150, 1.0048, 1.8288]))
        wall_obs_padding = 0.00
        self.wall_obs_corners = (real_wall_obs_corners[0] - wall_obs_padding,
                                 real_wall_obs_corners[1] + wall_obs_padding)

        super().__init__(None, 1000, backend=backend)

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        in_goal1 = inside_circle(obs_var.position, self.goal1_location, self.goal_radius, has_goal=True)
        in_goal2 = inside_circle(obs_var.position, self.goal2_location, self.goal_radius, has_goal=True)
        phi_liveness = stl.STLUntimedEventually(
            stl.STLAnd(in_goal1, stl.STLNext(stl.STLUntimedEventually(
                in_goal2))))

        in_table_obs = inside_box(obs_var.position, *self.table_obs_corners)
        in_wall_obs = inside_box(obs_var.position, *self.wall_obs_corners)
        in_either_obs = stl.STLOr(in_table_obs, in_wall_obs)
        phi_safety = stl.STLUntimedAlways(stl.STLNegation(in_either_obs))

        phi = stl.STLAnd(phi_liveness, phi_safety)
        return phi


class UR5eEEFScalabilityTestTask(UR5eEEFTaskBase):
    def __init__(self, backend="mjx"):
        self.goal1_location = jnp.array([0.225, 0.5, 0.5])
        self.goal2_location = jnp.array([-0.225, 0.5, 0.5])
        self.goal3_location = jnp.array([-0.225, 0.5, 0.1])
        # self.goal4_location = jnp.array([0.225, 0.5, 0.1])
        self.goal_radius = 0.10

        real_table_obs_corners = (jnp.array([-0.4572, 0.3952, 0.2844]), jnp.array([0.4572, 1.0048, 0.3444]))
        table_obs_padding = 0.00
        self.table_obs_corners = (real_table_obs_corners[0] - table_obs_padding,
                                  real_table_obs_corners[1] + table_obs_padding)

        real_wall_obs_corners = (jnp.array([-0.0150, 0.3952, 0.0000]), jnp.array([0.0150, 1.0048, 1.8288]))
        wall_obs_padding = 0.00
        self.wall_obs_corners = (real_wall_obs_corners[0] - wall_obs_padding,
                                 real_wall_obs_corners[1] + wall_obs_padding)

        super().__init__(None, 1000, backend=backend)

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        in_goal1 = inside_circle(obs_var.position, self.goal1_location, self.goal_radius, has_goal=True)
        in_goal2 = inside_circle(obs_var.position, self.goal2_location, self.goal_radius, has_goal=True)
        in_goal3 = inside_circle(obs_var.position, self.goal3_location, self.goal_radius, has_goal=True)
        # in_goal4 = inside_circle(obs_var.position, self.goal4_location, self.goal_radius, has_goal=True)
        phi_liveness = stl.STLUntimedEventually(
            stl.STLAnd(in_goal1, stl.STLNext(stl.STLUntimedEventually(
                stl.STLAnd(in_goal2, stl.STLNext(stl.STLUntimedEventually(
                    in_goal3)))))))

        # , stl.STLNext(stl.STLUntimedEventually(in_goal4))))))))))

        in_table_obs = inside_box(obs_var.position, *self.table_obs_corners)
        in_wall_obs = inside_box(obs_var.position, *self.wall_obs_corners)
        in_either_obs = stl.STLOr(in_table_obs, in_wall_obs)
        phi_safety = stl.STLUntimedAlways(stl.STLNegation(in_either_obs))

        phi = stl.STLAnd(phi_liveness, phi_safety)
        # phi = phi_liveness
        return phi

class UR5eEEFEasierBranchingScalabilityTestTask(UR5eEEFTaskBase):
    def __init__(self, backend="mjx"):
        self.goal1_location = jnp.array([-0.225, 0.5, 0.1])
        self.goal2_location = jnp.array([0.3048, 0.5, 0.5])
        self.goal3_location = jnp.array([0.000, 0.5, 0.5])
        self.goal4_location = jnp.array([-0.3048, 0.5, 0.5])
        self.goal_radius = 0.05

        real_table_obs_corners = (jnp.array([-0.4572, 0.4552, 0.2844]), jnp.array([0.4572, 1.0048, 0.3444]))
        table_obs_padding = jnp.array([0.00, 0.01, 0.00])
        self.table_obs_corners = (real_table_obs_corners[0] - table_obs_padding,
                                  real_table_obs_corners[1] + table_obs_padding)

        wall1_x = -0.1524
        wall2_x = 0.1524
        base_wall_obs_corners = (jnp.array([-0.0150, 0.4552, 0.3144]), jnp.array([0.0150, 1.0048, 1.8288]))
        wall1_obs_corners = tuple(map(lambda p: p.at[0].add(wall1_x), base_wall_obs_corners))
        wall2_obs_corners = tuple(map(lambda p: p.at[0].add(wall2_x), base_wall_obs_corners))

        print(wall1_obs_corners)
        print(wall2_obs_corners)

        wall_obs_padding = jnp.array([0.00, 0.00, 0.00])
        self.wall1_obs_corners = (wall1_obs_corners[0] - wall_obs_padding,
                                  wall1_obs_corners[1] + wall_obs_padding)
        self.wall2_obs_corners = (wall2_obs_corners[0] - wall_obs_padding,
                                  wall2_obs_corners[1] + wall_obs_padding)

        super().__init__(None, 1000, backend=backend)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = UR5eReachShelfEEF(
            backend=backend,
        )
        env.possible_goals = jnp.array([[-0.2250, 0.5, 0.1],
                                        [ 0.2250, 0.5, 0.1],
                                        [-0.3048, 0.5, 0.5],
                                        [ 0.0000, 0.5, 0.5],
                                        [ 0.3048, 0.5, 0.5]])
        return env

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        # in_goal1 = inside_circle(obs_var.position, self.goal1_location, self.goal_radius, has_goal=True)
        in_goal2 = inside_circle(obs_var.position, self.goal2_location, self.goal_radius, has_goal=True)
        in_goal3 = inside_circle(obs_var.position, self.goal3_location, self.goal_radius, has_goal=True)
        in_goal4 = inside_circle(obs_var.position, self.goal4_location, self.goal_radius, has_goal=True)

        phi_liveness = stl.STLUntimedEventually(
            stl.STLAnd(in_goal2, stl.STLNext(
                stl.STLUntimedEventually(stl.STLDisjunction([in_goal3, in_goal4]))
            )))
        
        in_table_obs = inside_box(obs_var.position, *self.table_obs_corners)
        in_wall1_obs = inside_box(obs_var.position, *self.wall1_obs_corners)
        in_wall2_obs = inside_box(obs_var.position, *self.wall2_obs_corners)
        in_either_obs = stl.STLDisjunction([in_table_obs, in_wall2_obs, in_wall1_obs])
        phi_safety = stl.STLUntimedAlways(stl.STLNegation(in_either_obs))

        phi = stl.STLAnd(phi_liveness, phi_safety)
        return phi

    @property
    def hdcqn_her_hps(self):
        return {
            **self.hdqn_her_hps,
            "hidden_layer_sizes": (256, 256),
            "hidden_cost_layer_sizes": (128, 128, 128, 64),
            "cost_scaling": 15.0,
            "safety_threshold": -0.06,
            "gamma_decay": 0.10, # the higher, the slower it reaches 1.0
            "gamma_end_value": 0.99,
        }


class UR5eEEFEvenEasierBranchingScalabilityTestTask(UR5eEEFTaskBase):
    def __init__(self, backend="mjx"):
        self.goal1_location = jnp.array([-0.225, 0.5, 0.1])
        self.goal2_location = jnp.array([0.000, 0.5, 0.5])
        self.goal3_location = jnp.array([0.3048, 0.5, 0.5])
        self.goal4_location = jnp.array([-0.3048, 0.5, 0.5])
        self.goal_radius = 0.10

        real_table_obs_corners = (jnp.array([-0.4572, 0.4552, 0.2844]), jnp.array([0.4572, 1.0048, 0.3444]))
        table_obs_padding = jnp.array([0.00, 0.01, 0.00])
        self.table_obs_corners = (real_table_obs_corners[0] - table_obs_padding,
                                  real_table_obs_corners[1] + table_obs_padding)

        wall1_x = -0.1524
        wall2_x = 0.1524
        base_wall_obs_corners = (jnp.array([-0.0150, 0.4552, 0.3144]), jnp.array([0.0150, 1.0048, 1.8288]))
        wall1_obs_corners = tuple(map(lambda p: p.at[0].add(wall1_x), base_wall_obs_corners))
        wall2_obs_corners = tuple(map(lambda p: p.at[0].add(wall2_x), base_wall_obs_corners))

        print(wall1_obs_corners)
        print(wall2_obs_corners)

        wall_obs_padding = jnp.array([0.00, 0.00, 0.00])
        self.wall1_obs_corners = (wall1_obs_corners[0] - wall_obs_padding,
                                  wall1_obs_corners[1] + wall_obs_padding)
        self.wall2_obs_corners = (wall2_obs_corners[0] - wall_obs_padding,
                                  wall2_obs_corners[1] + wall_obs_padding)

        super().__init__(None, 1000, backend=backend)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = UR5eReachShelfEEF(
            backend=backend,
        )
        env.possible_goals = jnp.array([[-0.2250, 0.5, 0.1],
                                        [ 0.2250, 0.5, 0.1],
                                        [-0.3048, 0.5, 0.5],
                                        [ 0.0000, 0.5, 0.5],
                                        [ 0.3048, 0.5, 0.5]])
        return env

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        # in_goal1 = inside_circle(obs_var.position, self.goal1_location, self.goal_radius, has_goal=True)
        in_goal2 = inside_circle(obs_var.position, self.goal2_location, self.goal_radius, has_goal=True)
        in_goal3 = inside_circle(obs_var.position, self.goal3_location, self.goal_radius, has_goal=True)
        in_goal4 = inside_circle(obs_var.position, self.goal4_location, self.goal_radius, has_goal=True)

        # phi_liveness = stl.STLUntimedEventually(
        #     stl.STLAnd(in_goal1, stl.STLNext(
        #         stl.STLUntimedEventually(stl.STLDisjunction([in_goal3, in_goal4]))
        #     )))

        phi_liveness = stl.STLUntimedEventually(stl.STLDisjunction([in_goal2, in_goal3, in_goal4]))

        in_table_obs = inside_box(obs_var.position, *self.table_obs_corners)
        in_wall1_obs = inside_box(obs_var.position, *self.wall1_obs_corners)
        in_wall2_obs = inside_box(obs_var.position, *self.wall2_obs_corners)
        in_either_obs = stl.STLDisjunction([in_table_obs, in_wall2_obs, in_wall1_obs])
        phi_safety = stl.STLUntimedAlways(stl.STLNegation(in_either_obs))

        phi = stl.STLAnd(phi_liveness, phi_safety)
        return phi

    @property
    def hdcqn_her_hps(self):
        return {
            **self.hdqn_her_hps,
            "hidden_layer_sizes": (256, 256),
            "hidden_cost_layer_sizes": (128, 128, 128, 64),
            "cost_scaling": 15.0,
            "safety_threshold": -0.06,
            "gamma_decay": 0.10, # the higher, the slower it reaches 1.0
            "gamma_end_value": 0.99,
        }


class UR5eEEFBranchingScalabilityTestTask(UR5eEEFTaskBase):
    def __init__(self, backend="mjx"):
        self.goal1_location = jnp.array([-0.225, 0.5, 0.1])
        self.goal2_location = jnp.array([0.000, 0.4, 0.5])
        self.goal3_location = jnp.array([0.3048, 0.5, 0.5])
        self.goal4_location = jnp.array([-0.3048, 0.5, 0.5])
        self.goal_radius = 0.10

        real_table_obs_corners = (jnp.array([-0.4572, 0.4552, 0.2844]), jnp.array([0.4572, 1.0048, 0.3444]))
        table_obs_padding = jnp.array([0.00, 0.01, 0.00])
        self.table_obs_corners = (real_table_obs_corners[0] - table_obs_padding,
                                  real_table_obs_corners[1] + table_obs_padding)

        wall1_x = -0.1524
        wall2_x = 0.1524
        base_wall_obs_corners = (jnp.array([-0.0150, 0.4552, 0.3144]), jnp.array([0.0150, 1.0048, 1.8288]))
        wall1_obs_corners = tuple(map(lambda p: p.at[0].add(wall1_x), base_wall_obs_corners))
        wall2_obs_corners = tuple(map(lambda p: p.at[0].add(wall2_x), base_wall_obs_corners))

        print(wall1_obs_corners)
        print(wall2_obs_corners)

        wall_obs_padding = jnp.array([0.00, 0.00, 0.00])
        self.wall1_obs_corners = (wall1_obs_corners[0] - wall_obs_padding,
                                  wall1_obs_corners[1] + wall_obs_padding)
        self.wall2_obs_corners = (wall2_obs_corners[0] - wall_obs_padding,
                                  wall2_obs_corners[1] + wall_obs_padding)

        super().__init__(None, 1000, backend=backend)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = UR5eReachShelfEEF(
            backend=backend,
        )
        env.possible_goals = jnp.array([[-0.2250, 0.5, 0.1],
                                        [ 0.2250, 0.5, 0.1],
                                        [-0.3048, 0.5, 0.5],
                                        [ 0.0000, 0.5, 0.5],
                                        [ 0.3048, 0.5, 0.5]])
        return env

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        in_goal1 = inside_circle(obs_var.position, self.goal1_location, self.goal_radius, has_goal=True)
        in_goal2 = inside_circle(obs_var.position, self.goal2_location, self.goal_radius, has_goal=True)
        in_goal3 = inside_circle(obs_var.position, self.goal3_location, self.goal_radius, has_goal=True)
        in_goal4 = inside_circle(obs_var.position, self.goal4_location, self.goal_radius, has_goal=True)

        phi_liveness = stl.STLUntimedEventually(
            stl.STLAnd(in_goal1, stl.STLNext(stl.STLUntimedEventually(
                stl.STLAnd(in_goal2, stl.STLNext(
                    stl.STLUntimedEventually(stl.STLOr(in_goal3, in_goal4))
                ))))))

        # phi_liveness = stl.STLUntimedEventually(
        #     stl.STLAnd(in_goal2, stl.STLNext(
        #         stl.STLUntimedEventually(stl.STLOr(in_goal3, in_goal4))
        #     )))

        in_table_obs = inside_box(obs_var.position, *self.table_obs_corners)
        in_wall1_obs = inside_box(obs_var.position, *self.wall1_obs_corners)
        in_wall2_obs = inside_box(obs_var.position, *self.wall2_obs_corners)
        in_either_obs = stl.STLDisjunction([in_table_obs, in_wall2_obs, in_wall1_obs])
        phi_safety = stl.STLUntimedAlways(stl.STLNegation(in_either_obs))

        phi = stl.STLAnd(phi_liveness, phi_safety)
        return phi

    @property
    def hdcqn_her_hps(self):
        return {
            **self.hdqn_her_hps,
            "hidden_layer_sizes": (256, 256),
            "hidden_cost_layer_sizes": (128, 128, 128, 64),
            "cost_scaling": 15.0,
            "safety_threshold": -0.06,
            "gamma_decay": 0.10, # the higher, the slower it reaches 1.0
            "gamma_end_value": 0.99,
        }


# class UR5eTranslateTask(UR5eTaskBase):
#     def __init__(self, backend="mjx"):
#         super().__init__(None, 1000, backend=backend)

#     def _build_env(self, backend: str) -> GoalConditionedEnv:
#         env = UR5eTranslate(
#             backend=backend
#         )
#         return env

#     def _build_hi_spec(self, wp_var: Var) -> Expression:
#         pass

#     def _build_lo_spec(self, obs_var: Var) -> Expression:
#         return true_exp(obs_var)


# class UR5eSingleSubgoal(SingleSubgoalMixin, UR5eTaskBase):
#     def __init__(self, backend="mjx"):
#         self.goal1_location = jnp.array([12.0, 4.0])
#         self.goal1_radius = 2.0

#         super().__init__(None, 1000, backend=backend)

#     def _build_env(self, backend: str) -> GoalConditionedEnv:
#         env = UR5eTranslate(
#             backend=backend
#         )
#         return env

#     def _build_hi_spec(self, wp_var: Var) -> Expression:
#         pass

#     def _build_lo_spec(self, obs_var: Var) -> Expression:
#         return true_exp(obs_var)


# class UR5eReachTask(UR5eTaskBase):
#     def __init__(self, backend="mjx"):
#         super().__init__(None, 1000, backend=backend)

#     def _build_env(self, backend: str) -> GoalConditionedEnv:
#         env = UR5eReach(
#             backend=backend
#         )
#         return env

#     def _build_hi_spec(self, wp_var: Var) -> Expression:
#         pass

#     def _build_lo_spec(self, obs_var: Var) -> Expression:
#         return true_exp(obs_var)
