import numpy as np
import jax.numpy as jnp

# , outside_circle, inside_box, true_exp
from achql.navix.tasks.templates import inside_circle

from achql.stl import Expression, Var
import achql.stl.expression_jax2 as stl


# class CenterConstraintMixin:

#     def _build_hi_spec(self, wp_var: Var) -> Expression:
#         pass

#     def _build_lo_spec(self, obs_var: Var) -> Expression:
#         at_obs1 = inside_circle(obs_var.position, self.obs1_location, self.obs_radius)
#         phi = stl.STLUntimedAlways(stl.STLNegation(at_obs1))
#         return phi

#     @property
#     def rm_config(self) -> dict:
#         return {
#             "final_state": 0,
#             "terminal_states": [1],
#             "reward_functions": {
#                 (1, 1): lambda s_t, a_t, s_t1: 0.0,
#                 (0, 1): lambda s_t, a_t, s_t1: 0.0,
#                 (0, 0): lambda s_t, a_t, s_t1: 1.0,
#             },
#             "pruned_edges": [(0, 0)]
#         }

#     @property
#     def lof_task_state_costs(self) -> jnp.ndarray:
#         raise NotImplementedError()


# class UMazeConstraintMixin:

#     def _build_hi_spec(self, wp_var: Var) -> Expression:
#         pass

#     def _build_lo_spec(self, obs_var: Var) -> Expression:
#         at_obs1 = inside_box(obs_var.position, *self.obs_corners)
#         phi = stl.STLUntimedAlways(stl.STLNegation(at_obs1))
#         return phi

#     @property
#     def rm_config(self) -> dict:
#         return {
#             "final_state": 0,
#             "terminal_states": [1],
#             "reward_functions": {
#                 (1, 1): lambda s_t, a_t, s_t1: 0.0,
#                 (0, 1): lambda s_t, a_t, s_t1: 0.0,
#                 (0, 0): lambda s_t, a_t, s_t1: 1.0,
#             },
#             "pruned_edges": [(0, 0)]
#         }

#     @property
#     def lof_task_state_costs(self) -> jnp.ndarray:
#         raise NotImplementedError()


class SingleSubgoalMixin:

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        in_goal1 = inside_circle(obs_var, self.goal1_location, self.goal1_radius, has_goal=True)
        phi = stl.STLUntimedEventually(in_goal1)
        return phi

    @property
    def rm_config(self) -> dict:
        return {
            "final_state": 0,
            "terminal_states": [],
            "reward_functions": {
                (1, 1): lambda s_t, a_t, s_t1: 0.0,
                (1, 0): lambda s_t, a_t, s_t1: 1.0,
                (0, 0): lambda s_t, a_t, s_t1: 1.0,
            },
            "pruned_edges": [(1, 0), (0, 0)]
        }

    @property
    def lof_task_state_costs(self) -> jnp.ndarray:
        raise NotImplementedError()


# class TwoSubgoalsMixin:

#     def _build_hi_spec(self, wp_var: Var) -> Expression:
#         pass

#     def _build_lo_spec(self, obs_var: Var) -> Expression:
#         in_goal1 = inside_circle(obs_var.position, self.goal1_location, self.goal1_radius, has_goal=True)
#         in_goal2 = inside_circle(obs_var.position, self.goal2_location, self.goal2_radius, has_goal=True)
#         phi = stl.STLUntimedEventually(
#             stl.STLAnd(in_goal1, stl.STLNext(stl.STLUntimedEventually(in_goal2)))
#         )
#         return phi

#     @property
#     def rm_config(self) -> dict:
#         return {
#             "final_state": 0,
#             "terminal_states": [],
#             "reward_functions": {
#                 (1, 1): lambda s_t, a_t, s_t1: 0.0,
#                 (1, 2): lambda s_t, a_t, s_t1: 0.0,
#                 (2, 2): lambda s_t, a_t, s_t1: 0.0,
#                 (2, 0): lambda s_t, a_t, s_t1: 1.0,
#                 (0, 0): lambda s_t, a_t, s_t1: 1.0,
#             },
#             "pruned_edges": [(1, 2), (2, 0), (0, 0)]
#         }

#     @property
#     def lof_task_state_costs(self) -> jnp.ndarray:
#         return jnp.array([0.0, 1.0, 1.0])


# class Branching1Mixin:

#     def _build_hi_spec(self, wp_var: Var) -> Expression:
#         pass

#     def _build_lo_spec(self, obs_var: Var) -> Expression:
#         in_goal1 = inside_circle(obs_var.position, self.goal1_location, self.goal1_radius, has_goal=True)
#         in_goal2 = inside_circle(obs_var.position, self.goal2_location, self.goal2_radius, has_goal=True)
#         phi = stl.STLAnd(stl.STLUntimedEventually(in_goal1),
#                          stl.STLUntimedEventually(in_goal2))
#         return phi

#     @property
#     def rm_config(self) -> dict:
#         return {
#             "final_state": 0,
#             "terminal_states": [],
#             "reward_functions": {
#                 (3, 2): lambda s_t, a_t, s_t1: 0.0,
#                 (3, 1): lambda s_t, a_t, s_t1: 0.0,
#                 (3, 0): lambda s_t, a_t, s_t1: 1.0,
#                 (2, 0): lambda s_t, a_t, s_t1: 1.0,
#                 (1, 0): lambda s_t, a_t, s_t1: 1.0,
#                 (0, 0): lambda s_t, a_t, s_t1: 1.0,
#             },
#             "pruned_edges": [(3, 2), (3, 1), (3, 0), (2, 0), (1, 0), (0, 0)]
#         }

#     @property
#     def lof_task_state_costs(self) -> jnp.ndarray:
#         return jnp.array([0.0, 1.0, 1.0, 1.0])


# class Branching2Mixin:

#     def _build_hi_spec(self, wp_var: Var) -> Expression:
#         pass

#     def _build_lo_spec(self, obs_var: Var) -> Expression:
#         in_goal1 = inside_circle(obs_var.position, self.goal1_location, self.goal1_radius, has_goal=True)
#         in_goal2 = inside_circle(obs_var.position, self.goal2_location, self.goal2_radius, has_goal=True)
#         phi = stl.STLOr(stl.STLUntimedEventually(in_goal1),
#                         stl.STLUntimedEventually(in_goal2))
#         return phi

#     @property
#     def rm_config(self) -> dict:
#         return {
#             "final_state": 0,
#             "terminal_states": [],
#             "reward_functions": {
#                 (1, 1): lambda s_t, a_t, s_t1: 0.0,
#                 (1, 0): lambda s_t, a_t, s_t1: 1.0,
#                 (0, 0): lambda s_t, a_t, s_t1: 1.0,
#             },
#             "pruned_edges": [(1, 0), (0, 0)]
#         }

#     @property
#     def lof_task_state_costs(self) -> jnp.ndarray:
#         raise NotImplementedError()


# class ObligationConstraint1Mixin:

#     def _build_hi_spec(self, wp_var: Var) -> Expression:
#         pass

#     def _build_lo_spec(self, obs_var: Var) -> Expression:
#         at_obs1 = inside_box(obs_var.position, *self.obs1_corners)
#         phi_safety = stl.STLUntimedAlways(stl.STLNegation(at_obs1))

#         in_goal1 = inside_circle(obs_var.position, self.goal1_location, self.goal1_radius, has_goal=True)
#         phi_liveness = stl.STLUntimedEventually(in_goal1)

#         phi = stl.STLAnd(phi_liveness, phi_safety)
#         return phi

#     @property
#     def rm_config(self) -> dict:
#         return {
#             "final_state": 0,
#             "terminal_states": [2],
#             "reward_functions": {
#                 (1, 1): lambda s_t, a_t, s_t1: 0.0,
#                 (1, 0): lambda s_t, a_t, s_t1: 1.0,
#                 (1, 2): lambda s_t, a_t, s_t1: 0.0,
#                 (0, 0): lambda s_t, a_t, s_t1: 1.0,
#                 (0, 2): lambda s_t, a_t, s_t1: 0.0,
#                 (2, 2): lambda s_t, a_t, s_t1: 0.0,
#             },
#             "pruned_edges": [(1, 0), (0, 0)]
#         }

#     @property
#     def lof_task_state_costs(self) -> jnp.ndarray:
#         return jnp.array([0.0, 1.0, 1.0])


# class ObligationConstraint2Mixin:

#     def _build_hi_spec(self, wp_var: Var) -> Expression:
#         pass

#     def _build_lo_spec(self, obs_var: Var) -> Expression:
#         at_obs1 = inside_circle(obs_var.position, self.obs1_location, self.obs_radius)
#         phi_safety = stl.STLUntimedAlways(stl.STLNegation(at_obs1))

#         in_goal1 = inside_circle(obs_var.position, self.goal1_location, self.goal1_radius, has_goal=True)
#         phi_liveness = stl.STLUntimedEventually(in_goal1)

#         phi = stl.STLAnd(phi_liveness, phi_safety)
#         return phi

#     @property
#     def rm_config(self) -> dict:
#         return {
#             "final_state": 0,
#             "terminal_states": [2],
#             "reward_functions": {
#                 (1, 1): lambda s_t, a_t, s_t1: 0.0,
#                 (1, 0): lambda s_t, a_t, s_t1: 1.0,
#                 (1, 2): lambda s_t, a_t, s_t1: 0.0,
#                 (0, 0): lambda s_t, a_t, s_t1: 1.0,
#                 (0, 2): lambda s_t, a_t, s_t1: 0.0,
#                 (2, 2): lambda s_t, a_t, s_t1: 0.0,
#             },
#             "pruned_edges": [(1, 0), (0, 0)]
#         }

#     @property
#     def lof_task_state_costs(self) -> jnp.ndarray:
#         return jnp.array([0.0, 1.0, 1.0])


# class ObligationConstraint3Mixin(ObligationConstraint2Mixin):
#     pass
